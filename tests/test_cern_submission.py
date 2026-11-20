"""CERN profile translation, Condor lifecycle and portable input configuration."""
import ast
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python/worker'))
from submission_config import ResolveSubmitConfig, SnakeMakeCommand

spec = importlib.util.spec_from_file_location('cern_condor', ROOT/'profiles/cern/condor.py')
cern = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cern)


def parse(*options):
  tree = ast.parse((ROOT/'scripts/innfer.py').read_text())
  function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
  namespace = {'argparse': argparse, 'AdjustArgs': lambda args: args}
  exec(compile(ast.Module(body=[function], type_ignores=[]), '<CLI>', 'exec'), namespace)
  with patch.object(sys, 'argv', ['innfer.py', *options]):
    return namespace['parse_args']()


def workflow_functions():
  tree = ast.parse((ROOT/'python/worker/useful_functions.py').read_text())
  functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'GetSnakeMakeStepLoop']
  namespace = {'copy': copy, 'yaml': yaml}
  exec(compile(ast.Module(body=functions, type_ignores=[]), '<workflow>', 'exec'), namespace)
  return namespace['GetSnakeMakeStepLoop']


class CERNSubmissionTests(unittest.TestCase):
  def test_site_mapping_covers_every_binned_step_and_retains_physics_options(self):
    expand = workflow_functions()
    old = Path.cwd()
    try:
      os.chdir(ROOT)
      original = expand(yaml.safe_load((ROOT/'configs/snakemake/btm_binned.yaml').read_text()))
      migrated = expand(yaml.safe_load((ROOT/'configs/snakemake/btm_binned_cern.yaml').read_text()))
      self.assertEqual(len(migrated), len(original))
      for before, after in zip(original, migrated):
        self.assertEqual(before['step'], after['step'])
        options = dict(after['run_options'])
        mapping = options.pop('submit_config_map')
        self.assertEqual(options, before.get('run_options', {}))
        self.assertEqual(after['retry_runtime'], before.get('retry_runtime', 1440))
        target = ResolveSubmitConfig(after['submit'], mapping)
        config = yaml.safe_load(Path(target).read_text())
        description = cern.submit_description('job.sh', ROOT, {'params': {
          'submit_options': config['options']['extra_lines']}})
        self.assertIn('+JobFlavour = "workday"', description)
        self.assertNotIn('RequestGpus', description)
        self.assertNotIn('request_gpus', description)
        self.assertNotIn('has_avx', description)
    finally:
      os.chdir(old)

  def test_missing_mapping_is_an_error(self):
    with tempfile.TemporaryDirectory() as temporary:
      mapping = Path(temporary)/'map.yaml'
      mapping.write_text('unrelated: other.yaml\n')
      with self.assertRaisesRegex(ValueError, 'No submission mapping'):
        ResolveSubmitConfig('unmapped.yaml', mapping)
      original = mapping.stat()
      mapping.write_text('- not a mapping\n')
      os.utime(mapping, ns=(original.st_atime_ns, original.st_mtime_ns))
      with self.assertRaisesRegex(ValueError, 'path pairs'):
        ResolveSubmitConfig('unmapped.yaml', mapping)
    self.assertEqual(ResolveSubmitConfig('ic.yaml'), 'ic.yaml')

  def test_controller_command_keeps_retry_and_force_flags_and_quotes_paths(self):
    args, _ = parse('--snakemake-profile=profiles/cern', '--snakemake-directory=project space',
      '--snakemake-rerun-incomplete', '--snakemake-force')
    cmd = SnakeMakeCommand(args, 'file space.smk')
    self.assertIn('profiles/cern', cmd)
    self.assertIn('project space', cmd)
    self.assertIn('file space.smk', cmd)
    self.assertIn('--rerun-incomplete', cmd)
    self.assertIn('--forceall', cmd)
    self.assertIn('input-output', cmd)
    self.assertNotIn('--forceall', SnakeMakeCommand(args, 'file.smk', unlock=True))
    args.snakemake_local = True
    self.assertNotIn('--profile', SnakeMakeCommand(args, 'file.smk'))
    self.assertEqual(parse()[0].snakemake_profile, 'profiles/ic')

  def test_condor_options_preserve_expressions_and_retry_overrides(self):
    properties = {'params': {'submit_options': ['request_cpus=4', 'request_memory=12000',
      'requirements=(TARGET.Arch == "X86_64") && (TARGET.Memory >= 16000)',
      '+JobFlavour="workday"', '+MaxRuntime=10800']},
      'resources': {'retry_runtime': 1440, 'retry_memory': 24000}}
    description = cern.submit_description("job 'with' spaces.sh", ROOT, properties)
    self.assertIn('request_cpus = 4', description)
    self.assertIn('request_memory = 24000', description)
    self.assertIn('requirements = (TARGET.Arch == "X86_64") && (TARGET.Memory >= 16000)', description)
    self.assertIn('+JobFlavour = "tomorrow"', description)
    self.assertNotIn('+MaxRuntime', description)
    self.assertIn('should_transfer_files = NO', description)
    for limit, name in cern.FLAVOURS:
      self.assertEqual(cern.flavour(limit), name)
    self.assertEqual(cern.flavour(600), 'tomorrow')
    with self.assertRaises(ValueError):
      cern.flavour(10081)

  @unittest.skipUnless(shutil.which('condor_submit'), 'Condor command line tools are not installed')
  def test_native_condor_parser_accepts_paths_arguments_and_resources(self):
    with tempfile.TemporaryDirectory(prefix='cern profile ') as temporary:
      root = Path(temporary)
      script = root/"payload 'quoted'.sh"
      script.write_text('true\n')
      description = root/'job.sub'
      description.write_text(cern.submit_description(script, root, {'params': {
        'submit_options': ['request_cpus=4', 'request_memory=16000']}}))
      result = subprocess.run(['condor_submit', '-dry-run', str(root/'job.ad'), str(description)],
        text=True, capture_output=True)
      self.assertEqual(result.returncode, 0, result.stderr)
      # Do not print the full ClassAd: getenv includes the controller environment.
      lines = (root/'job.ad').read_text().splitlines()
      self.assertTrue(any(line.replace(' ', '') == 'RequestCpus=4' for line in lines))
      self.assertTrue(any(line.replace(' ', '') == 'RequestMemory=16000' for line in lines))
      self.assertTrue(any(line.replace(' ', '') == 'ShouldTransferFiles="NO"' for line in lines))

  def test_status_uses_history_and_rejects_signal_or_nonzero_exit(self):
    for code, signal, expected in [(0, False, 'success'), (2, False, 'failed'), (0, True, 'failed')]:
      with patch.object(cern, 'query', side_effect=[[], [{'JobStatus': 4, 'ExitCode': code, 'ExitBySignal': signal}]]) as query:
        self.assertEqual(cern.status('123.0'), expected)
      self.assertEqual(query.call_args.args[0][0], 'condor_history')
    for state in [1, 2, 6, 7]:
      with patch.object(cern, 'query', return_value=[{'JobStatus': state}]):
        self.assertEqual(cern.status('123.0'), 'running')
    with patch.object(cern, 'query', side_effect=[[], []]):
      self.assertEqual(cern.status('123.0'), 'running')
    with patch.object(cern, 'query', side_effect=subprocess.CalledProcessError(1, 'condor_q')):
      self.assertEqual(cern.status('123.0'), 'running')

  def test_empty_successful_queue_output_falls_back_to_history(self):
    for empty in ['', ' \n\t']:
      with self.subTest(output=repr(empty)):
        results = [subprocess.CompletedProcess([], 0, empty, ''),
          subprocess.CompletedProcess([], 0, json.dumps([
            {'JobStatus': 4, 'ExitCode': 0, 'ExitBySignal': False}]), '')]
        with patch.object(cern.subprocess, 'run', side_effect=results) as run:
          self.assertEqual(cern.status('123.0'), 'success')
        self.assertEqual([call.args[0][0] for call in run.call_args_list],
          ['condor_q', 'condor_history'])

  def test_empty_queue_and_history_remain_running_without_parse_warnings(self):
    results = [subprocess.CompletedProcess([], 0, '', ''),
      subprocess.CompletedProcess([], 0, '\n', '')]
    with patch.object(cern.subprocess, 'run', side_effect=results), \
        patch('builtins.print') as output:
      self.assertEqual(cern.status('123.0'), 'running')
      output.assert_not_called()

  def test_invalid_nonempty_condor_output_remains_unknown_with_command_context(self):
    for stdout in ['not JSON', '{}', 'null', '[1]']:
      with self.subTest(output=stdout), \
          patch.object(cern.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout, '')) as run, \
          patch('builtins.print') as output:
        self.assertEqual(cern.status('123.0'), 'running')
        self.assertEqual(run.call_count, 1)
        message = output.call_args.args[0]
        self.assertIn('Condor status temporarily unavailable', message)
        self.assertIn('condor_q', message)

  def test_held_job_is_removed_before_retry_and_reason_is_printed(self):
    with patch.object(cern, 'query', return_value=[{'JobStatus': 5, 'HoldReason': 'runtime limit'}]), \
        patch.object(cern.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as remove:
      self.assertEqual(cern.status('123.0'), 'failed')
      self.assertEqual(remove.call_args.args[0], ['condor_rm', '123.0'])
    with patch.object(cern, 'query', return_value=[{'JobStatus': 5}]), \
        patch.object(cern.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, stderr='unavailable')):
      self.assertEqual(cern.status('123.0'), 'running')

  def test_jobscript_propagates_payload_failure(self):
    template = (ROOT/'profiles/cern/jobscript.sh').read_text()
    script = template.format(properties='{}', exec_job='false\necho must-not-run')
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertNotIn('must-not-run', result.stdout)

  def test_submit_prints_only_condor_id_and_retains_submit_record(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      script = root/'script.sh'
      script.write_text('# properties = '+json.dumps({'params': {'submit_options': []}})+'\ntrue\n')
      with patch.dict(os.environ, {'INNFER_CONDOR_LOG_DIR': str(root/'logs')}), \
          patch.object(cern.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123.0 - 123.0\n')) as run, \
          patch('builtins.print') as output:
        cern.submit(script)
        output.assert_called_once_with('123.0')
        self.assertEqual(run.call_args.args[0][0:2], ['condor_submit', '-terse'])
      records = list((root/'logs').glob('*/submission.json'))
      self.assertEqual(json.loads(records[0].read_text())['job_id'], '123.0')
      with patch.dict(os.environ, {'INNFER_CONDOR_LOG_DIR': str(root/'logs')}), \
          patch.object(cern.subprocess, 'run') as run, patch('builtins.print'):
        cern.submit(script, dry_run=True)
        run.assert_not_called()


if __name__ == '__main__':
  unittest.main()
