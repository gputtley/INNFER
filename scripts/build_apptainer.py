#!/usr/bin/env python3
"""Freeze the current code and conda environment into a versioned Apptainer image."""
import argparse
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ['python/worker', 'python/runner', 'python/BayesFlow/bayesflow',
               'scripts', 'configs', 'tests', 'containers', 'profiles']
SOURCE_FILES = ['env.sh', 'README.md', 'LICENSE', 'python/BayesFlow/setup.py',
                'python/BayesFlow/setup.cfg', 'python/BayesFlow/pyproject.toml']
SOURCE_SUFFIXES = {'.py', '.sh', '.yaml', '.yml', '.json', '.txt', '.md', '.def', '.toml', '.cfg', '.ini'}
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', 'tmp', 'archive', 'build', 'dist'}


def sha256(path):
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
      digest.update(block)
  return digest.hexdigest()


def git_value(root, *arguments):
  result = subprocess.run(['git', '-C', str(root), *arguments], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
  return result.stdout.strip() if result.returncode == 0 else None


def source_files(root):
  files = set()
  for directory in SOURCE_DIRS:
    base = root/directory
    if not base.is_dir():
      continue
    for parent, dirs, names in os.walk(base, followlinks=False):
      dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
      for name in names:
        path = Path(parent)/name
        if path.suffix in SOURCE_SUFFIXES or path.relative_to(root).as_posix() == 'containers/innfer':
          files.add(path)
  files.update(root/name for name in SOURCE_FILES if (root/name).is_file())
  for path in sorted(files):
    if not path.resolve().is_relative_to(root.resolve()):
      raise ValueError(f'Source symlink points outside the checkout: {path}')
    yield path


def environment_inventory(prefix):
  if not (prefix/'conda-meta').is_dir() or not (prefix/'bin/python').is_file():
    raise ValueError(f'Not a conda environment: {prefix}')
  conda = []
  for path in sorted((prefix/'conda-meta').glob('*.json')):
    data = json.loads(path.read_text())
    conda.append({key: data.get(key) for key in ['name', 'version', 'build']})
  packages, editable = [], []
  for path in sorted(prefix.glob('lib/python*/site-packages/*.dist-info')):
    metadata = Parser().parsestr((path/'METADATA').read_text(errors='replace'), headersonly=True)
    name = metadata.get('Name', path.name)
    packages.append({'name': name, 'version': metadata.get('Version')})
    direct = path/'direct_url.json'
    if direct.exists() and json.loads(direct.read_text()).get('dir_info', {}).get('editable'):
      editable.append(name)
  unsupported = [name for name in editable if name.lower().replace('-', '') != 'bayesflow']
  if unsupported:
    raise ValueError(f'Editable packages need an explicit source snapshot: {unsupported}')
  return {'source_prefix': str(prefix), 'conda_packages': conda, 'python_packages': packages,
          'editable_packages': editable}


def replaced_conda_metadata(prefix, inventory):
  """Accept only obsolete dist-info files replaced by an installed pip version."""
  normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
  installed = {normalize(package['name']): package['version'] for package in inventory['python_packages']}
  accepted = []
  for path in sorted((prefix/'conda-meta').glob('*.json')):
    record = json.loads(path.read_text())
    missing = [name for name in record.get('files', []) if not os.path.lexists(prefix/name)]
    if not missing:
      continue
    name, version = record['name'], record['version']
    old_directory = name.replace('-', '_')+'-'+version+'.dist-info'
    metadata_only = all(len(Path(item).parts) == 5 and
      Path(item).parts[:1] == ('lib',) and Path(item).parts[2] == 'site-packages' and
      Path(item).parts[3] == old_directory for item in missing)
    replacement = installed.get(normalize(name))
    if not metadata_only or replacement is None or replacement == version:
      raise ValueError(f'Missing conda-managed files need repair in a separate environment: {name}: {missing[:5]}')
    accepted.append({'package': name, 'conda_version': version, 'installed_version': replacement,
                     'missing_obsolete_metadata': missing})
  return accepted


def snapshot(root, archive, metadata):
  entries, tree = [], hashlib.sha256()
  with tarfile.open(archive, 'w:gz') as output:
    for path in source_files(root):
      relative = path.relative_to(root).as_posix()
      content = path.read_bytes()
      digest = hashlib.sha256(content).hexdigest()
      tree.update(relative.encode()+b'\0'+digest.encode()+b'\n')
      entries.append({'path': relative, 'sha256': digest, 'size': len(content)})
      item = tarfile.TarInfo(relative)
      item.size = len(content)
      item.mode = 0o755 if os.access(path, os.X_OK) else 0o644
      output.addfile(item, io.BytesIO(content))
    manifest = {**metadata, 'source_sha256': tree.hexdigest(), 'source_files': entries}
    content = (json.dumps(manifest, indent=2)+'\n').encode()
    item = tarfile.TarInfo('container-manifest.json')
    item.size, item.mode = len(content), 0o644
    output.addfile(item, io.BytesIO(content))
  return manifest


def pack_command(context, prefix, audited_metadata=False):
  command = [str(context/'tools/bin/conda-pack'), '--prefix', str(prefix),
    '--output', str(context/'environment.tar.gz'), '--dest-prefix', '/opt/innfer/env',
    '--ignore-editable-packages', '--compress-level', '1', '--n-threads', '2',
    '--exclude', '*.pyc', '--exclude', '*/__pycache__/*']
  if audited_metadata:
    command.append('--ignore-missing-files')
  return command


def run(command, env):
  print('+ '+__import__('shlex').join(command), flush=True)
  subprocess.run(command, check=True, env=env)


def reusable_environment(context, inventory):
  """Reuse a completed package only with its matching environment inventory."""
  manifest = json.loads((context/'container-manifest.json').read_text())
  if manifest['environment'] != inventory:
    raise ValueError('The saved environment inventory differs from --env-prefix; package a fresh environment')
  archive = context/'environment.tar.gz'
  if not archive.is_file() or archive.stat().st_size == 0:
    raise FileNotFoundError(f'No completed environment archive: {archive}')
  return archive.resolve()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--env-prefix', type=Path,
    default=Path(os.environ.get('CONDA_PREFIX', ROOT/'miniconda/files/envs/innfer_env')))
  parser.add_argument('--work-dir', type=Path, default=ROOT/'tmp/apptainer-build',
    help='Build/cache parent; prefer node-local scratch with at least 30 GB free')
  parser.add_argument('--output', type=Path, help='New .sif path; existing images are never overwritten')
  parser.add_argument('--base-image', default='rockylinux:9', help='Docker image tag or digest')
  parser.add_argument('--prepare-only', action='store_true', help='Freeze code/metadata and write the build plan, without packing or building')
  parser.add_argument('--reuse-environment-from', type=Path,
    help='Reuse environment.tar.gz from a previous build context with a matching environment inventory; freeze current code afresh')
  parser.add_argument('--min-free-gb', type=float, default=30.)
  args = parser.parse_args()
  if args.min_free_gb <= 0:
    parser.error('--min-free-gb must be positive')
  prefix, parent = args.env_prefix.resolve(), args.work_dir.resolve()
  inventory = environment_inventory(prefix)
  inventory['replaced_conda_metadata'] = replaced_conda_metadata(prefix, inventory)
  reused_archive = reusable_environment(args.reuse_environment_from.resolve(), inventory) if args.reuse_environment_from else None
  required = ['scripts/innfer.py', 'python/BayesFlow/bayesflow/amortizers.py', 'containers/innfer.def']
  for name in required:
    if not (ROOT/name).is_file():
      raise FileNotFoundError(f'Required source missing (initialize BayesFlow submodule): {name}')
  parent.mkdir(parents=True, exist_ok=True)
  free_gb = shutil.disk_usage(parent).free/1024**3
  if not args.prepare_only and free_gb < args.min_free_gb:
    raise RuntimeError(f'Only {free_gb:.1f} GiB free at {parent}; choose a larger --work-dir')
  timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  context = Path(tempfile.mkdtemp(prefix=f'innfer-{timestamp}-', dir=parent))
  metadata = {'created_utc': timestamp, 'git_commit': git_value(ROOT, 'rev-parse', 'HEAD'),
    'git_tracked_changes': git_value(ROOT, 'status', '--porcelain', '--untracked-files=no'),
    'bayesflow_commit': git_value(ROOT/'python/BayesFlow', 'rev-parse', 'HEAD'),
    'environment': inventory, 'base_image': args.base_image}
  if reused_archive is not None:
    metadata['environment_reused_from'] = str(reused_archive)
  manifest = snapshot(ROOT, context/'code.tar.gz', metadata)
  output = args.output.resolve() if args.output else ROOT/'images'/f'innfer-{timestamp}-{manifest["source_sha256"][:12]}.sif'
  partial = output.with_name(output.name+'.partial')
  if output.exists() or partial.exists():
    raise FileExistsError(f'Choose a new output path: {output}')
  recipe = context/'innfer.def'
  shutil.copyfile(ROOT/'containers/innfer.def', recipe)
  build = ['apptainer', 'build', '--fakeroot', '--no-cleanup', '--build-arg', f'BASE_IMAGE={args.base_image}',
    '--build-arg', f'ENV_ARCHIVE={context}/environment.tar.gz', '--build-arg',
    f'CODE_ARCHIVE={context}/code.tar.gz', str(partial), str(recipe)]
  smoke = ['apptainer', 'exec', '--cleanenv', '--no-home', '--no-mount', 'cwd,hostfs,bind-paths',
    '--pwd', '/tmp', str(partial), 'python', '/opt/innfer/app/containers/smoke_test.py']
  plan = {'context': str(context), 'output': str(output), 'source_sha256': manifest['source_sha256'],
    'pack_command': pack_command(context, prefix, bool(inventory['replaced_conda_metadata'])), 'build_command': build,
    'isolated_smoke_command': smoke}
  (context/'build-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
  (context/'container-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
  print(json.dumps(plan, indent=2), flush=True)
  if args.prepare_only:
    return
  if shutil.which('apptainer') is None:
    raise RuntimeError('apptainer is required on the build host')
  build_env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
    'APPTAINER_TMPDIR': str(context/'temporary'), 'APPTAINER_CACHEDIR': str(context/'cache'),
    'TMPDIR': str(context/'temporary'), 'PIP_CACHE_DIR': str(context/'pip-cache')}
  for name in ['temporary', 'cache', 'pip-cache']:
    (context/name).mkdir()
  if reused_archive is not None:
    (context/'environment.tar.gz').symlink_to(reused_archive)
    print(f'Reusing packaged environment: {reused_archive}', flush=True)
  else:
    # Isolated packaging tools: never install into or change the running environment.
    run([sys.executable, '-m', 'venv', str(context/'tools')], build_env)
    run([str(context/'tools/bin/python'), '-m', 'pip', '--isolated',
         '--cache-dir', str(context/'pip-cache'), 'install', '--index-url', 'https://pypi.org/simple',
         'conda-pack==0.8.1', 'setuptools==80.9.0'], build_env)
    run(plan['pack_command'], build_env)
  output.parent.mkdir(parents=True, exist_ok=True)
  run(build, build_env)
  # Also test without host home/cwd/site bind mounts hiding missing dependencies.
  run(smoke, build_env)
  partial.replace(output)
  digest = sha256(output)
  (output.with_name(output.name+'.sha256')).write_text(f'{digest}  {output.name}\n')
  (output.with_name(output.name+'.json')).write_text(json.dumps({**manifest,
    'image_sha256': digest, 'environment_archive_sha256': sha256(context/'environment.tar.gz'),
    'cpu_smoke_passed': True, 'gpu_smoke_passed': False,
    'build_context': str(context)}, indent=2)+'\n')
  print(f'Built and CPU-tested: {output}', flush=True)
  print('GPU validation is separate: run containers/smoke_test.py --require-gpu with --nv on a GPU worker.', flush=True)


if __name__ == '__main__':
  main()
