"""Prepare immutable job configs and a self-contained Apptainer batch launcher."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import tarfile
import tempfile
from functools import lru_cache


def _Sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
      digest.update(block)
  return digest.hexdigest()


@lru_cache(maxsize=8)
def _ImageInfo(path, signature, verify=False):
  path = Path(path)
  checksum = Path(str(path)+'.sha256')
  if checksum.is_file():
    words = checksum.read_text().split()
    digest = words[0].lower() if words else ''
    if not re.fullmatch('[0-9a-f]{64}', digest):
      raise ValueError(f'Invalid image checksum: {checksum}')
  else:
    digest = _Sha256(path)
  if verify and checksum.is_file() and _Sha256(path) != digest:
    raise ValueError(f'Container image checksum mismatch: {path}')
  metadata = Path(str(path)+'.json')
  source = json.loads(metadata.read_text()).get('source_sha256') if metadata.is_file() else None
  return {'image': str(path), 'image_sha256': digest, 'image_source_sha256': source}


def _SnapshotConfigs(project, directory, config_files):
  files = set()
  # Setup assets and unrelated run configs can be very large. These directories
  # contain supporting architectures, translations and plotting conventions.
  for name in ['configs/architecture', 'configs/other']:
    base = project/name
    if not base.is_dir():
      continue
    for parent, dirs, names in os.walk(base, followlinks=False):
      dirs[:] = sorted(d for d in dirs if d not in ['archive', '__pycache__', 'tmp'] and not d.startswith('.'))
      files.update(Path(parent)/name for name in names
        if Path(name).suffix in ['.py', '.yaml', '.yml', '.json', '.txt', '.pkl'])
  for value in config_files:
    if value is not None and Path(value).is_file():
      files.add(Path(os.path.abspath(value)))

  entries, bindings, tree = [], [], hashlib.sha256()
  for path in sorted(files):
    path = Path(os.path.abspath(path))
    if path.is_relative_to(project/'configs'):
      name = path.relative_to(project).as_posix()
    else:
      name = f'extra/{len(bindings)}/{path.name}'
      bindings.append({'source': name, 'target': str(path)})
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    tree.update(name.encode()+b'\0'+str(path).encode()+b'\0'+digest.encode()+b'\n')
    entries.append((name, data, digest))

  source_digest = tree.hexdigest()
  directory.mkdir(parents=True, exist_ok=True)
  archive = directory/f'configs-{source_digest}.tar.gz'
  if not archive.exists():
    fd, temporary = tempfile.mkstemp(prefix='.configs-', dir=directory)
    os.close(fd)
    temporary = Path(temporary)
    try:
      with tarfile.open(temporary, 'w:gz') as output:
        for name, data, digest in entries:
          item = tarfile.TarInfo(name)
          item.size, item.mode = len(data), 0o644
          output.addfile(item, io.BytesIO(data))
        data = json.dumps({'bindings': bindings,
          'files': {name: digest for name, _, digest in entries}}, indent=2).encode()
        item = tarfile.TarInfo('config-manifest.json')
        item.size, item.mode = len(data), 0o644
        output.addfile(item, io.BytesIO(data))
      # Concurrent workflow generation must not replace an already pinned archive.
      try:
        os.link(temporary, archive)
      except FileExistsError:
        pass
    finally:
      temporary.unlink(missing_ok=True)
  return {'config_archive': str(archive), 'config_sha256': _Sha256(archive),
    'config_source_sha256': source_digest, 'config_bindings': bindings,
    'config_size_bytes': archive.stat().st_size}


def PrepareContainer(image, project, directory, config_files=(), image_mode='direct'):
  """Pin the image checksum, execution mode and submission-time configuration."""
  if image_mode not in ['copy', 'direct']:
    raise ValueError(f'Invalid container image mode: {image_mode}')
  image, project, directory = Path(image).resolve(), Path(project).resolve(), Path(directory).resolve()
  if not image.is_file():
    raise FileNotFoundError(f'Container image not found: {image}')
  signature = []
  for path in [image, Path(str(image)+'.sha256'), Path(str(image)+'.json')]:
    signature.append((path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None)
  return {**_ImageInfo(str(image), tuple(signature), verify=image_mode == 'direct'),
    'project': str(project), 'image_mode': image_mode, 'image_size_bytes': signature[0][1],
    **_SnapshotConfigs(project, directory/'containers', config_files)}


def ContainerSubmitOptions(options, container):
  """Request runtime scratch and, in copy mode, space for the local image."""
  lines = list(options.get('extra_lines', []))
  if not any(line.partition('=')[0].strip().lower().replace('_', '') == 'requestdisk' for line in lines):
    # Condor RequestDisk is in KiB; leave 1 GiB for runtime caches and temporary files.
    image_bytes = container['image_size_bytes'] if container.get('image_mode', 'direct') == 'copy' else 0
    required = (image_bytes+container['config_size_bytes']+1023)//1024 + 1024**2
    lines.append(f'RequestDisk = {required}')
  return {**options, 'extra_lines': lines}


def RequestsGPU(options):
  """Mirror a submission file's RequestGpus setting, including expressions."""
  for line in options.get('extra_lines', []):
    key, separator, value = line.partition('=')
    if separator and key.strip().lower().replace('_', '') == 'requestgpus':
      try:
        return float(value.strip()) > 0
      except ValueError:
        return value.strip().lower() not in ['', 'false', 'undefined']
  return False


_SETUP = r'''
command -v apptainer >/dev/null || { echo 'apptainer is required on the worker' >&2; exit 1; }
innfer_scratch="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
if [[ "$innfer_image_mode" == "direct" ]]; then
  [[ -f "$innfer_image_source" && -r "$innfer_image_source" ]] || {
    echo "Container image is not readable: $innfer_image_source" >&2; exit 1;
  }
  innfer_runtime_image="$innfer_image_source"
else
  innfer_cache="${INNFER_CONTAINER_CACHE:-$innfer_scratch/innfer-images-$UID}"
  mkdir -p "$innfer_cache"
  innfer_runtime_image="$innfer_cache/$innfer_image_sha.sif"
  (
    flock -x 9
    if [[ ! -f "$innfer_runtime_image" ]] || ! printf '%s  %s\n' "$innfer_image_sha" "$innfer_runtime_image" | sha256sum --check --status; then
      innfer_partial=$(mktemp "$innfer_cache/.image.XXXXXX")
      trap 'rm -f -- "$innfer_partial"' EXIT
      cp -- "$innfer_image_source" "$innfer_partial"
      printf '%s  %s\n' "$innfer_image_sha" "$innfer_partial" | sha256sum --check --status || {
        echo 'Container image checksum mismatch' >&2; exit 1;
      }
      chmod 444 "$innfer_partial"
      mv -f -- "$innfer_partial" "$innfer_runtime_image"
    fi
  ) 9>"$innfer_cache/$innfer_image_sha.lock"
fi
innfer_work=$(mktemp -d "$innfer_scratch/innfer-runtime.XXXXXX")
trap 'rm -rf -- "$innfer_work"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$innfer_work/snapshot/configs" "$innfer_work/tmp"
cp -- "$innfer_config_source" "$innfer_work/configs.tar.gz"
printf '%s  %s\n' "$innfer_config_sha" "$innfer_work/configs.tar.gz" | sha256sum --check --status || {
  echo 'Container config checksum mismatch' >&2; exit 1;
}
tar -xzf "$innfer_work/configs.tar.gz" -C "$innfer_work/snapshot"
innfer_options=(--cleanenv --bind "$innfer_project:$innfer_project" --bind "$innfer_work:$innfer_work"
  --bind "$innfer_work/snapshot/configs:$innfer_project/configs" --pwd "$innfer_project"
  --env "TMPDIR=$innfer_work/tmp" --env "MPLCONFIGDIR=$innfer_work/matplotlib"
  --env "XDG_CACHE_HOME=$innfer_work/cache")
'''


_RUN = r'''
innfer_container() {
  local innfer_name innfer_ticket_file='' innfer_krb_config
  local -a innfer_environment=()
  local -a innfer_auth_bindings=() innfer_krb_configs=()
  for innfer_name in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES OMP_NUM_THREADS TF_NUM_INTEROP_THREADS TF_NUM_INTRAOP_THREADS WANDB_MODE WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY KRB5CCNAME KRB5_CONFIG; do
    if [[ -v "$innfer_name" ]]; then
      innfer_environment+=(--env "$innfer_name=${!innfer_name}")
    fi
  done
  # Use credentials available on this worker, not a login-node path captured
  # during submission. In particular, /run/user caches are not default mounts.
  case "${KRB5CCNAME:-}" in
    FILE:/*) innfer_ticket_file="${KRB5CCNAME#FILE:}" ;;
    /*) innfer_ticket_file="$KRB5CCNAME" ;;
  esac
  if [[ -n "$innfer_ticket_file" && -f "$innfer_ticket_file" && -r "$innfer_ticket_file" ]]; then
    innfer_auth_bindings+=(--bind "$innfer_ticket_file:$innfer_ticket_file:ro")
  fi
  IFS=: read -r -a innfer_krb_configs <<< "${KRB5_CONFIG:-}"
  for innfer_krb_config in "${innfer_krb_configs[@]}"; do
    if [[ "$innfer_krb_config" = /* && -f "$innfer_krb_config" && -r "$innfer_krb_config" ]]; then
      innfer_auth_bindings+=(--bind "$innfer_krb_config:$innfer_krb_config:ro")
    fi
  done
  apptainer run "${innfer_environment[@]}" "${innfer_options[@]}" "${innfer_auth_bindings[@]}" "$innfer_runtime_image" "$@"
}
printf 'INNFER container: %s\nImage mode: %s\nImage SHA256: %s\nConfig SHA256: %s\n' "$innfer_image_source" "$innfer_image_mode" "$innfer_image_sha" "$innfer_config_sha"
cd "$innfer_project"
'''


def ContainerJob(options, commands, job_name, submit_options):
  """Create a launcher that needs no host Python or conda environment on the worker."""
  project = Path(options['project'])
  directories = {name: os.environ.get(name) or default for name, default in {
    'PREP_DATA_DIR': 'data', 'EVAL_DATA_DIR': 'data', 'MODELS_DIR': 'models',
    'PLOTS_DIR': 'plots', 'JOBS_DIR': 'jobs'}.items()}
  gpu = RequestsGPU(submit_options)
  metadata = {**options, 'gpu': gpu, 'commands': commands, 'directories': directories}
  manifest = Path(str(Path(job_name).with_suffix(''))+'.container.json')
  manifest.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(metadata, indent=2)+'\n'
  if not manifest.exists() or manifest.read_text() != text:
    manifest.write_text(text)

  lines = ['#!/bin/bash', 'set -euo pipefail', 'umask 077', 'ulimit -s unlimited']
  for name, key in [('project', 'project'), ('image_source', 'image'),
      ('image_sha', 'image_sha256'), ('config_source', 'config_archive'), ('config_sha', 'config_sha256')]:
    lines.append(f'innfer_{name}={shlex.quote(options[key])}')
  lines.append(f'innfer_image_mode={shlex.quote(options.get("image_mode", "direct"))}')
  lines.append(_SETUP)
  if gpu:
    lines.append('innfer_options+=(--nv)')
  else:
    lines.append('innfer_options+=(--env CUDA_VISIBLE_DEVICES=-1)')
  for binding in options['config_bindings']:
    source = shlex.quote(binding['source'])
    target = shlex.quote(binding['target'])
    lines.append(f'innfer_options+=(--bind "$innfer_work/snapshot/"{source}:{target}:ro)')
  for name, value in directories.items():
    lines.append(f'innfer_options+=(--env {shlex.quote(name+"="+value)})')
  # Bind external roots as well as the project. Their existing parent may be
  # needed when this job is the first to create an output directory.
  roots = set()
  for value in directories.values():
    path = Path(value)
    if not path.is_absolute() or path.is_relative_to(project):
      continue
    while not path.exists() and path != path.parent:
      path = path.parent
    if path != Path('/'):
      roots.add(str(path))
  for path in sorted(roots):
    lines.append(f'innfer_options+=(--bind {shlex.quote(path+":"+path)})')
  lines += [_RUN, *commands]
  return lines
