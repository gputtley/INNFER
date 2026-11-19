"""Container builds must freeze code without changing a live environment."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_apptainer', ROOT/'scripts/build_apptainer.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
smoke_spec = importlib.util.spec_from_file_location('container_smoke_test', ROOT/'containers/smoke_test.py')
smoke = importlib.util.module_from_spec(smoke_spec)
smoke_spec.loader.exec_module(smoke)


class ApptainerBuildTests(unittest.TestCase):
  def test_snapshot_contains_current_code_and_ignored_configs_but_no_data_or_environment(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      included = ['scripts/innfer.py', 'python/worker/current.py',
        'python/BayesFlow/bayesflow/amortizers.py', 'configs/run/ignored.py',
        'containers/innfer', 'containers/innfer.def', 'profiles/cern/config.yaml']
      excluded = ['data/events.py', 'models/model.py', 'miniconda/secret.py',
        '.aws/credentials', 'python/worker/__pycache__/module.py',
        'python/runner/archive/obsolete.py', 'configs/setup/tmp/cache.json']
      for relative in included+excluded:
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative+'\n')
      archive = root/'code.tar.gz'
      manifest = builder.snapshot(root, archive, {'test': True})
      with tarfile.open(archive) as reader:
        self.assertEqual(set(reader.getnames()), set(included+['container-manifest.json']))
        for entry in manifest['source_files']:
          payload = reader.extractfile(entry['path']).read()
          self.assertEqual(hashlib.sha256(payload).hexdigest(), entry['sha256'])
        (root/'python/worker/current.py').write_text('later changes')
        self.assertEqual(reader.extractfile('python/worker/current.py').read(), b'python/worker/current.py\n')

  def test_external_source_symlinks_fail_instead_of_packaging_unrelated_files(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)/'repo'
      (root/'scripts').mkdir(parents=True)
      external = Path(directory)/'external.py'
      external.write_text('private')
      (root/'scripts/external.py').symlink_to(external)
      with self.assertRaisesRegex(ValueError, 'outside the checkout'):
        list(builder.source_files(root))

  def test_inventory_allows_frozen_bayesflow_but_rejects_other_editable_dependencies(self):
    with tempfile.TemporaryDirectory() as directory:
      prefix = Path(directory)
      (prefix/'bin').mkdir()
      (prefix/'bin/python').touch()
      (prefix/'conda-meta').mkdir()
      (prefix/'conda-meta/python.json').write_text(json.dumps({'name': 'python', 'version': '3.11', 'build': 'test'}))
      dist = prefix/'lib/python3.11/site-packages/bayesflow.dist-info'
      dist.mkdir(parents=True)
      (dist/'METADATA').write_text('Name: BayesFlow\nVersion: 1.0\n')
      (dist/'direct_url.json').write_text(json.dumps({'dir_info': {'editable': True}}))
      before = (dist/'METADATA').read_bytes()
      inventory = builder.environment_inventory(prefix)
      self.assertEqual(inventory['editable_packages'], ['BayesFlow'])
      self.assertEqual((dist/'METADATA').read_bytes(), before)
      (dist/'METADATA').write_text('Name: another-package\nVersion: 1.0\n')
      with self.assertRaisesRegex(ValueError, 'explicit source snapshot'):
        builder.environment_inventory(prefix)

  def test_packing_uses_fixed_image_prefix_and_keeps_missing_file_checks(self):
    command = builder.pack_command(Path('/build'), Path('/live/env'))
    self.assertIn('/live/env', command)
    self.assertEqual(command[command.index('--dest-prefix')+1], '/opt/innfer/env')
    self.assertIn('--ignore-editable-packages', command)
    self.assertNotIn('--ignore-missing-files', command)
    self.assertEqual(command[0], '/build/tools/bin/conda-pack')

  def test_only_replaced_dist_info_is_accepted_as_missing_conda_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      prefix = Path(directory)
      (prefix/'conda-meta').mkdir()
      path = prefix/'conda-meta/interface.json'
      record = {'name': 'snakemake-interface-common', 'version': '1.17.2',
        'files': ['lib/python3.11/site-packages/snakemake_interface_common-1.17.2.dist-info/METADATA']}
      path.write_text(json.dumps(record))
      inventory = {'python_packages': [{'name': 'snakemake-interface-common', 'version': '1.21.0'}]}
      result = builder.replaced_conda_metadata(prefix, inventory)
      self.assertEqual(result[0]['installed_version'], '1.21.0')
      self.assertIn('--ignore-missing-files', builder.pack_command(Path('/build'), prefix, bool(result)))
      record['files'].append('lib/python3.11/site-packages/snakemake_interface_common/__init__.py')
      path.write_text(json.dumps(record))
      with self.assertRaisesRegex(ValueError, 'Missing conda-managed files'):
        builder.replaced_conda_metadata(prefix, inventory)
      record['files'].pop()
      path.write_text(json.dumps(record))
      with self.assertRaises(ValueError):
        builder.replaced_conda_metadata(prefix, {'python_packages': []})

  def test_runtime_imports_use_image_code_and_preserve_explicit_data_locations(self):
    command = ['sh', '-c', '. "$1"; printf "%s\\n" "$PYTHONPATH" "$PREP_DATA_DIR" "$PYTHONNOUSERSITE" "$PYTHONDONTWRITEBYTECODE"',
               'test', str(ROOT/'containers/environment.sh')]
    result = subprocess.run(command, env={**os.environ, 'PYTHONPATH': '/live/checkout',
      'PREP_DATA_DIR': '/work/inputs'}, text=True, check=True, stdout=subprocess.PIPE)
    lines = result.stdout.splitlines()
    self.assertIn('/opt/innfer/app/python/BayesFlow', lines[0])
    self.assertNotIn('/live/checkout', lines[0])
    self.assertEqual(lines[1:], ['/work/inputs', '1', '1'])

  def test_environment_reuse_requires_saved_matching_inventory_and_archive(self):
    with tempfile.TemporaryDirectory() as directory:
      context = Path(directory)
      inventory = {'python_packages': [{'name': 'tensorflow', 'version': '2.15.1'}]}
      (context/'container-manifest.json').write_text(json.dumps({'environment': inventory}))
      with self.assertRaises(FileNotFoundError):
        builder.reusable_environment(context, inventory)
      archive = context/'environment.tar.gz'
      archive.write_bytes(b'completed archive')
      self.assertEqual(builder.reusable_environment(context, inventory), archive.resolve())
      with self.assertRaisesRegex(ValueError, 'inventory differs'):
        builder.reusable_environment(context, {'python_packages': []})

  def test_smoke_tool_failures_show_actual_output(self):
    stderr = io.StringIO()
    with redirect_stderr(stderr), self.assertRaises(subprocess.CalledProcessError):
      smoke.run_checked([sys.executable, '-c',
        "import sys; print('LaTeX diagnostic', file=sys.stderr); sys.exit(1)"])
    self.assertIn('LaTeX diagnostic', stderr.getvalue())
    result = smoke.run_checked([sys.executable, '-c', "print('passed')"])
    self.assertEqual(result.stdout.strip(), 'passed')


if __name__ == '__main__':
  unittest.main()
