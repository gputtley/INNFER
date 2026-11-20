"""Both submission paths use frozen configs, local verified images and image code."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python/worker'))
from batch import Batch
from container_submission import PrepareContainer, RequestsGPU, ContainerSubmitOptions
from module import Module
from useful_functions import GetSnakeMakeStepLoop


@contextmanager
def working_directory(path):
  previous = Path.cwd()
  try:
    os.chdir(path)
    yield
  finally:
    os.chdir(previous)


def parse(*options):
  tree = ast.parse((ROOT/'scripts/innfer.py').read_text())
  function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
  namespace = {'argparse': argparse, 'AdjustArgs': lambda args: args}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<CLI>', 'exec'), namespace)
  with patch.object(sys, 'argv', ['innfer.py', *options]):
    return namespace['parse_args']()


class ContainerSubmissionTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.base = Path(self.temporary.name)
    self.project = self.base/'project space'
    self.project.mkdir()
    self.cfg = self.project/'configs/run/test.py'
    self.cfg.parent.mkdir(parents=True)
    self.cfg.write_text('config = {"name": "original"}\n')
    arch = self.project/'configs/architecture/default.yaml'
    arch.parent.mkdir(parents=True)
    arch.write_text('type: FCNN\n')
    self.image = self.base/'image space.sif'
    self.image.write_bytes(b'test image')
    self.digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
    Path(str(self.image)+'.sha256').write_text(self.digest+'  image space.sif\n')
    self.jobs = self.base/'jobs'
    self.jobs.mkdir()
    self.environment = {**os.environ, '_CONDOR_SCRATCH_DIR': str(self.base/'scratch'),
      'INNFER_CONTAINER_CACHE': str(self.base/'cache'), 'TMPDIR': str(self.base/'scratch'),
      'FAKE_APPTAINER_LOG': str(self.base/'calls.jsonl')}
    Path(self.environment['_CONDOR_SCRATCH_DIR']).mkdir()
    binary = self.base/'bin'
    binary.mkdir()
    self.environment['PATH'] = str(binary)+os.pathsep+os.environ['PATH']
    app = binary/'apptainer'
    app.write_text('#!'+sys.executable+'\n'+'''
import json, os, pathlib, sys
args = sys.argv[1:]
config = None
for index, arg in enumerate(args[:-1]):
  if arg == '--bind' and '/snapshot/configs:' in args[index+1]:
    source = args[index+1].split(':')[0]
    config = (pathlib.Path(source)/'run/test.py').read_text()
with open(os.environ['FAKE_APPTAINER_LOG'], 'a') as stream:
  stream.write(json.dumps({'args': args, 'config': config})+'\\n')
sys.exit(7 if '--fail' in args else 0)
''')
    app.chmod(0o755)

  def prepare(self):
    with working_directory(self.project):
      return PrepareContainer(self.image, self.project, self.jobs, [str(self.cfg)])

  def job(self, commands=None, gpu=False):
    options = self.prepare()
    path = self.jobs/('gpu.sh' if gpu else 'cpu.sh')
    batch = Batch({'container_image': str(self.image), 'container_options': options,
      'job_name': str(path), 'options': {'extra_lines': ['RequestGpus = 1']} if gpu else {}})
    batch._CreateBatchJob(commands or ['innfer_container --help'])
    subprocess.run(['bash', '-n', str(path)], check=True)
    return path, options

  def run_job(self, path):
    return subprocess.run(['bash', str(path)], env=self.environment, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  def calls(self):
    return [json.loads(line) for line in Path(self.environment['FAKE_APPTAINER_LOG']).read_text().splitlines()]

  def test_runtime_uses_local_verified_image_and_frozen_config(self):
    job, options = self.job()
    self.cfg.write_text('changed after submission')
    result = self.run_job(job)
    self.assertEqual(result.returncode, 0, result.stdout)
    call = self.calls()[0]
    self.assertEqual(call['config'], 'config = {"name": "original"}\n')
    self.assertIn(str(self.base/'cache'/f'{self.digest}.sif'), call['args'])
    self.assertNotIn('--nv', call['args'])
    self.assertIn('CUDA_VISIBLE_DEVICES=-1', call['args'])
    self.assertIn('--cleanenv', call['args'])
    self.assertNotIn('source env.sh', job.read_text())
    self.assertFalse(list(Path(self.environment['_CONDOR_SCRATCH_DIR']).glob('innfer-runtime.*')))
    metadata = json.loads(job.with_suffix('.container.json').read_text())
    self.assertEqual(metadata['image_sha256'], self.digest)
    self.assertEqual(metadata['config_sha256'], options['config_sha256'])

  def test_gpu_allocation_and_worker_failure_are_preserved(self):
    job, _ = self.job(['innfer_container --fail', 'innfer_container --must-not-run'], gpu=True)
    self.environment['CUDA_VISIBLE_DEVICES'] = 'GPU-allocated'
    result = self.run_job(job)
    self.assertEqual(result.returncode, 7, result.stdout)
    self.assertEqual(len(self.calls()), 1)
    self.assertIn('--nv', self.calls()[0]['args'])
    self.assertIn('CUDA_VISIBLE_DEVICES=GPU-allocated', self.calls()[0]['args'])
    self.assertFalse(list(Path(self.environment['_CONDOR_SCRATCH_DIR']).glob('innfer-runtime.*')))

  def test_runtime_kerberos_cache_and_configuration_are_forwarded_without_packaging(self):
    job, options = self.job()
    tickets = self.base/'worker ticket cache'
    tickets.mkdir()
    cache = tickets/'krb5cc'
    cache.write_bytes(b'test ticket cache')
    config = tickets/'krb5.conf'
    config.write_text('[libdefaults]\n default_realm = EXAMPLE.INVALID\n')
    self.environment['KRB5CCNAME'] = 'FILE:'+str(cache)
    self.environment['KRB5_CONFIG'] = str(config)
    result = self.run_job(job)
    self.assertEqual(result.returncode, 0, result.stdout)
    arguments = self.calls()[0]['args']
    self.assertIn('KRB5CCNAME=FILE:'+str(cache), arguments)
    self.assertIn('KRB5_CONFIG='+str(config), arguments)
    self.assertIn(f'{cache}:{cache}:ro', arguments)
    self.assertIn(f'{config}:{config}:ro', arguments)
    self.assertNotIn(str(cache), job.read_text())
    self.assertNotIn(str(cache), job.with_suffix('.container.json').read_text())
    with tarfile.open(options['config_archive']) as archive:
      self.assertFalse(any('krb5cc' in name or 'krb5.conf' in name for name in archive.getnames()))

  def test_non_file_kerberos_cache_is_forwarded_without_a_file_binding(self):
    job, _ = self.job()
    self.environment['KRB5CCNAME'] = 'KEYRING:persistent:12345'
    result = self.run_job(job)
    self.assertEqual(result.returncode, 0, result.stdout)
    arguments = self.calls()[0]['args']
    self.assertIn('KRB5CCNAME=KEYRING:persistent:12345', arguments)
    self.assertFalse(any(arg.endswith(':12345:ro') for arg in arguments))

  def test_concurrent_cache_population_and_reuse(self):
    job, _ = self.job()
    processes = [subprocess.Popen(['bash', str(job)], env=self.environment,
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    for process in processes:
      output, _ = process.communicate(timeout=20)
      self.assertEqual(process.returncode, 0, output)
    cached = self.base/'cache'/f'{self.digest}.sif'
    self.assertEqual(cached.read_bytes(), b'test image')
    self.image.unlink()
    self.assertEqual(self.run_job(job).returncode, 0)
    self.assertFalse(list((self.base/'cache').glob('.image.*')))

  def test_corrupt_source_image_and_changed_config_archive_fail_before_execution(self):
    job, options = self.job()
    self.image.write_bytes(b'wrong image')
    result = self.run_job(job)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('Container image checksum mismatch', result.stdout)
    self.assertFalse(Path(self.environment['FAKE_APPTAINER_LOG']).exists())
    self.image.write_bytes(b'test image')
    Path(options['config_archive']).write_bytes(b'wrong configs')
    result = self.run_job(job)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('Container config checksum mismatch', result.stdout)
    self.assertFalse(Path(self.environment['FAKE_APPTAINER_LOG']).exists())

  def test_explicit_external_configs_are_frozen_and_bound(self):
    external = self.base/'external config.yaml'
    external.write_text('architecture: original\n')
    with working_directory(self.project):
      options = PrepareContainer(self.image, self.project, self.jobs, [str(self.cfg), str(external)])
    binding = options['config_bindings'][0]
    self.assertEqual(binding['target'], str(external))
    external.write_text('edited')
    with tarfile.open(options['config_archive']) as archive:
      self.assertEqual(archive.extractfile(binding['source']).read(), b'architecture: original\n')

  def test_cli_and_workflow_inheritance_and_both_submission_paths(self):
    defaults, _ = parse()
    self.assertIsNone(defaults.container_image)
    workflow = [{'workflow': 'child.yaml', 'run_options': {'container_image': str(self.image)}}]
    child = [{'step': 'LikelihoodDebug', 'submit': 'submit.yaml'}]
    nested = self.base/'workflow.yaml'
    nested.write_text(yaml.safe_dump(child))
    workflow[0]['workflow'] = str(nested)
    self.assertEqual(GetSnakeMakeStepLoop(workflow)[0]['run_options']['container_image'], str(self.image))

    submission = self.base/'submit.yaml'
    submission.write_text(yaml.safe_dump({'name': 'condor', 'options': {
      'extra_lines': ['RequestGpus = 1', '+MaxRuntime = 36000']}}))
    mapping = self.base/'map.yaml'
    original_submission = self.base/'old_site.yaml'
    mapping.write_text(yaml.safe_dump({str(original_submission): str(submission)}))
    dangerous = 'mass=$(touch SHOULD_NOT_EXIST);literal=`echo untouched`'
    for snake in [False, True]:
      directory = self.jobs/('snake' if snake else 'direct')
      directory.mkdir()
      args, default_args = parse('--cfg='+str(self.cfg), '--step=LikelihoodDebug',
        '--container-image='+str(self.image), '--submit='+str(original_submission),
        '--submit-config-map='+str(mapping), '--snakemake-profile=profiles/cern',
        '--debug-input='+dangerous, '--dry-run')
      args.make_snakemake_inputs = snake
      with working_directory(self.project):
        module = Module(['scripts/innfer.py'], args, default_args, job_name=str(directory/'innfer_Debug'))
        command = module._SetupBatchCommand({'file_name': 'combined', 'val_ind': 2})
        arguments = shlex.split(command)
        self.assertIn('--debug-input='+dangerous, arguments)
        self.assertEqual(arguments[0], 'innfer_container')
        self.assertFalse(any(arg.startswith(('--container-image', '--submit=', '--make-snakemake-inputs',
          '--submit-config-map', '--snakemake-profile', '--snakemake-shared-fs-usage')) for arg in arguments))
        module.cmd_store, module.extra_names = [command], ['_combined']
        module.input_store = ['events.parquet']
        module.output_store = [] if snake else ['result.yaml']
        with patch('batch.os.system') as system:
          module.Sweep()
        self.assertFalse(any('condor_submit' in str(call) for call in system.call_args_list))
      jobs = list((directory/'LikelihoodDebug').glob('*/innfer_Debug_0.sh'))
      self.assertEqual(len(jobs), 1)
      job = jobs[0]
      self.assertIn('innfer_options+=(--nv)', job.read_text())
      subprocess.run(['bash', '-n', str(job)], check=True)
      if snake:
        rule = (directory/'innfer_SnakeMake.txt').read_text()
        self.assertIn(str(self.image), rule)
        self.assertIn('configs-', rule)
        self.assertIn('request_gpus=1', rule)
        self.assertIn('request_disk=', rule)
        self.assertIn('  output: []', rule)
        self.assertIn('+MaxRuntime=36000', rule)
        self.assertIn(str(job), rule)
        self.assertIn('rule innfer_Debug_0:', rule)
      else:
        self.assertIn('RequestGpus = 1', job.with_suffix('.sub').read_text())
        self.assertIn('RequestDisk = ', job.with_suffix('.sub').read_text())
    self.assertFalse((self.project/'SHOULD_NOT_EXIST').exists())

  def test_many_container_jobs_fit_small_directory_entry_limits(self):
    args, defaults = parse('--container-image='+str(self.image),
      '--step=PreProcessParallelBinnedFitInputsParameters')
    module = Module([], args, defaults)
    parent = self.jobs/'many'
    parent.mkdir()
    original_open = Path.open

    def limited_open(path, mode='r', *arguments, **keywords):
      # AFS can reject creation in a full directory with EFBIG even for tiny files.
      if any(flag in mode for flag in ['w', 'a', 'x']) and not path.exists():
        if sum(1 for _ in path.parent.iterdir()) >= 6:
          raise OSError(27, 'File too large', str(path))
      return original_open(path, mode, *arguments, **keywords)

    generated = []
    options = self.prepare()
    for index in [0, 499, 500, 6114]:
      name = str(parent/f'innfer_PreProcessParallelBinnedFitInputsParameters_{index}')
      name = module._JobPath(name)
      self.assertEqual(name, module._JobPath(str(parent/Path(name).name)))
      generated.append(name)
      with patch.object(Path, 'open', limited_open):
        Batch({'container_image': str(self.image), 'container_options': options,
          'job_name': name+'.sh', 'dry_run': True}).RunCondor()
    self.assertEqual(len(set(generated)), 4)
    self.assertEqual(len(list(parent.rglob('*.container.json'))), 4)
    self.assertGreater(len({Path(name).parent for name in generated}), 1)

  def test_non_container_jobs_use_step_directories_in_both_submission_paths(self):
    submission = self.base/'submit.yaml'
    submission.write_text(yaml.safe_dump({'name': 'condor', 'options': {}}))
    for snake in [False, True]:
      directory = self.jobs/('snake' if snake else 'direct')
      args, defaults = parse('--step=TrainDensity', '--submit='+str(submission), '--dry-run')
      args.make_snakemake_inputs = snake
      module = Module(['scripts/innfer.py'], args, defaults,
        job_name=str(directory/'innfer_TrainDensity'))
      module.cmd_store, module.extra_names = ['python3 scripts/innfer.py --help'], ['_test']
      module.Sweep()
      job = directory/'TrainDensity/00000/innfer_TrainDensity_0.sh'
      self.assertIn('source env.sh', job.read_text())
      if snake:
        rule = (directory/'innfer_SnakeMake.txt').read_text()
        self.assertIn(str(job), rule)
        self.assertIn('rule innfer_TrainDensity_0:', rule)
      else:
        self.assertIn(str(job), job.with_suffix('.sub').read_text())

  def test_named_jobs_keep_distinct_stable_paths_within_the_step(self):
    args, defaults = parse('--step=TrainDensity')
    module = Module([], args, defaults, options={'use_number_for_jobs': False})
    names = [str(self.jobs/f'innfer_TrainDensity_{suffix}') for suffix in ['sample_a', 'sample_b']]
    paths = [module._JobPath(name) for name in names]
    self.assertEqual(len(set(paths)), 2)
    self.assertEqual(paths, [module._JobPath(name) for name in names])
    for path, name in zip(paths, names):
      self.assertEqual(Path(path).relative_to(self.jobs).parts[0], 'TrainDensity')
      self.assertEqual(Path(path).name, Path(name).name)

  def test_missing_image_invalid_checksum_and_gpu_request_zero(self):
    with self.assertRaises(FileNotFoundError):
      PrepareContainer(self.base/'missing.sif', self.project, self.jobs)
    Path(str(self.image)+'.sha256').write_text('invalid')
    with self.assertRaisesRegex(ValueError, 'Invalid image checksum'):
      self.prepare()
    self.assertFalse(RequestsGPU({'extra_lines': ['RequestGpus = 0']}))
    self.assertTrue(RequestsGPU({'extra_lines': ['request_gpus = 1']}))

  def test_external_data_root_is_forwarded_and_bound(self):
    raw = self.base/'raw input'
    raw.mkdir()
    with patch.dict(os.environ, {'PREP_DATA_DIR': str(raw)}):
      job, options = self.job()
    content = job.read_text()
    self.assertIn(shlex.quote('PREP_DATA_DIR='+str(raw)), content)
    self.assertIn(shlex.quote(str(raw)+':'+str(raw)), content)
    metadata = json.loads(job.with_suffix('.container.json').read_text())
    self.assertEqual(metadata['directories']['PREP_DATA_DIR'], str(raw))

  def test_disabled_container_keeps_host_environment_setup(self):
    job = self.jobs/'legacy.sh'
    Batch({'job_name': str(job)})._CreateBatchJob(['python3 scripts/innfer.py --help'])
    self.assertIn('source env.sh', job.read_text())
    self.assertNotIn('apptainer', job.read_text())

  def test_scratch_request_accounts_for_image_and_preserves_explicit_disk(self):
    container = {'image_size_bytes': 3*1024**3, 'config_size_bytes': 1024}
    original = {'extra_lines': ['RequestGpus = 1']}
    options = ContainerSubmitOptions(original, container)
    self.assertIn('RequestDisk = 4194305', options['extra_lines'])
    self.assertEqual(original['extra_lines'], ['RequestGpus = 1'])
    explicit = {'extra_lines': ['RequestDisk = 20GB', 'RequestGpus = 1']}
    self.assertEqual(ContainerSubmitOptions(explicit, container), explicit)


if __name__ == '__main__':
  unittest.main()
