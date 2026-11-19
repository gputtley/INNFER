"""Controller-side submission settings, independent of the worker environment."""
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=16)
def _ReadSubmitMap(path, signature):
  mapping = yaml.safe_load(Path(path).read_text())
  if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
    raise ValueError(f'Submission map must contain submission YAML path pairs: {path}')
  return {str(Path(key).resolve()): value for key, value in mapping.items()}


def ResolveSubmitConfig(path, mapping_path=None):
  """Require an explicit site mapping for every selected submission config."""
  if mapping_path is None:
    return path
  mapping_file = Path(mapping_path).resolve()
  info = mapping_file.stat()
  mapping = _ReadSubmitMap(str(mapping_file), (info.st_mtime_ns, info.st_ctime_ns, info.st_size))
  key = str(Path(path).resolve())
  if key not in mapping:
    raise ValueError(f'No submission mapping for {path} in {mapping_path}')
  target = mapping[key]
  if not Path(target).is_file():
    raise FileNotFoundError(f'Mapped submission config not found: {target}')
  return target


def SnakeMakeCommand(args, snakefile, unlock=False):
  """Build the controller command without passing shell syntax through paths."""
  command = ['snakemake', '--cores', 'all']
  command += ['--executor', 'local'] if args.snakemake_local else ['--profile', args.snakemake_profile]
  command += ['-s', str(snakefile)]
  if args.snakemake_directory is not None:
    command += ['--directory', args.snakemake_directory]
  if unlock:
    command.append('--unlock')
  else:
    command += ['--shared-fs-usage', *args.snakemake_shared_fs_usage]
    if args.snakemake_rerun_incomplete:
      command.append('--rerun-incomplete')
    if args.snakemake_force:
      command.append('--forceall')
  return command
