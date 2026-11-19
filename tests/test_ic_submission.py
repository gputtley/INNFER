"""IC settings, retry resources and job-log handling without submitting jobs."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ic_condor', ROOT/'profiles/ic/condor.py')
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)

TOKEN = 'a'*32
EXTERNAL_ID = f'7_{TOKEN}_123'
TYPES = SimpleNamespace(JOB_TERMINATED=5, JOB_ABORTED=9, JOB_HELD=12,
  EXECUTABLE_ERROR=2, JOB_RELEASED=13)


class Event(dict):
  def __init__(self, kind, cluster=123, proc=0, **values):
    super().__init__(values)
    self.type, self.cluster, self.proc = kind, cluster, proc


class ICSubmissionTests(unittest.TestCase):
  def fake_bindings(self, events=()):
    event_log = Mock()
    event_log.events.return_value = iter(events)
    htcondor = SimpleNamespace(JobEventType=TYPES, JobEventLog=Mock(return_value=event_log),
      JobAction=SimpleNamespace(Remove='Remove'), Submit=Mock(side_effect=lambda data: data), Schedd=Mock())
    htcondor.Schedd.return_value.act.return_value = {'TotalSuccess': 1, 'TotalError': 0}
    htcondor.Schedd.return_value.submit.return_value.cluster.return_value = 123
    return htcondor, event_log

  def test_workflow_requests_preserved_and_only_retry_resources_override_them(self):
    options = ['+MaxRuntime=10800', 'request_cpus=4', 'request_memory=8000',
      'request_gpus=1', 'requirements=(TARGET.Memory >= 8000) && (TARGET.Arch == "X86_64")']
    props = {'params': {'submit_options': options}}
    original = ic.submit_settings('worker.sh', Path('/tmp/logs'), props)
    self.assertEqual(original['+MaxRuntime'], '10800')
    self.assertEqual(original['request_cpus'], '4')
    self.assertEqual(original['request_memory'], '8000')
    self.assertEqual(original['request_gpus'], '1')
    self.assertEqual(original['requirements'], options[-1].partition('=')[2])
    self.assertNotIn('+JobFlavour', original)
    self.assertNotIn('max_retries', original)
    props['resources'] = {'retry_runtime': 600, 'retry_memory': 16000}
    retried = ic.submit_settings('worker.sh', Path('/tmp/logs'), props)
    self.assertEqual(retried['+MaxRuntime'], '36000')
    self.assertEqual(retried['request_memory'], '16000')
    self.assertEqual(retried['request_cpus'], '4')
    self.assertEqual(retried['request_gpus'], '1')

  def test_submission_returns_condor_cluster_number_and_keeps_portable_record(self):
    htcondor, _ = self.fake_bindings()
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      script = root/'worker.sh'
      script.write_text('# properties = '+json.dumps({'jobid': 7, 'params': {'submit_options': ['+MaxRuntime=10800']}})+'\ntrue\n')
      output = io.StringIO()
      with patch.dict(os.environ, {'INNFER_CONDOR_LOG_DIR': str(root/'logs')}), \
          patch.object(ic, 'uuid4', return_value=SimpleNamespace(hex=TOKEN)), \
          patch.object(ic, 'bindings', return_value=htcondor), redirect_stdout(output):
        ic.submit(script)
      self.assertEqual(output.getvalue().strip(), EXTERNAL_ID)
      record = json.loads((root/'logs'/f'7_{TOKEN}'/'submission.json').read_text())
      self.assertEqual(record['job_id'], EXTERNAL_ID)
      self.assertEqual(record['settings']['log'], str(root/'logs'/f'7_{TOKEN}'/'condor.log'))
      self.assertEqual(htcondor.Schedd.return_value.submit.call_args.kwargs, {'count': 1})

  def test_dry_run_needs_no_condor_connection(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      script = root/'worker.sh'
      script.write_text('# properties = {"jobid": 7}\ntrue\n')
      with patch.dict(os.environ, {'INNFER_CONDOR_LOG_DIR': str(root/'logs')}), \
          patch.object(ic, 'bindings') as bindings, redirect_stdout(io.StringIO()):
        ic.submit(script, dry_run=True)
      bindings.assert_not_called()
      self.assertEqual(len(list((root/'logs').glob('*/submission.json'))), 1)

  def test_status_is_nonblocking_and_distinguishes_failure_and_signals(self):
    for normal, code, expected in [(True, 0, 'success'), (True, 3, 'failed'), (False, 0, 'failed')]:
      event = Event(TYPES.JOB_TERMINATED, TerminatedNormally=normal, ReturnValue=code)
      htcondor, log = self.fake_bindings([event])
      with patch.object(ic, 'bindings', return_value=htcondor):
        self.assertEqual(ic.status(EXTERNAL_ID), expected)
      log.events.assert_called_once_with(stop_after=0)
      log.close.assert_called_once()
    htcondor, _ = self.fake_bindings([Event(TYPES.JOB_ABORTED)])
    with patch.object(ic, 'bindings', return_value=htcondor):
      self.assertEqual(ic.status(EXTERNAL_ID), 'failed')

  def test_unrelated_events_and_unavailable_logs_do_not_trigger_retry(self):
    htcondor, _ = self.fake_bindings([
      Event(TYPES.JOB_TERMINATED, cluster=999, TerminatedNormally=True, ReturnValue=0)])
    with patch.object(ic, 'bindings', return_value=htcondor):
      self.assertEqual(ic.status(EXTERNAL_ID), 'running')
    htcondor.JobEventLog.side_effect = OSError('log not yet visible')
    with patch.object(ic, 'bindings', return_value=htcondor):
      self.assertEqual(ic.status(EXTERNAL_ID), 'running')
    htcondor.Schedd.assert_not_called()

  def test_held_jobs_are_removed_before_retry_and_released_jobs_are_left_running(self):
    htcondor, _ = self.fake_bindings([Event(TYPES.JOB_HELD, HoldReason='runtime limit')])
    with patch.object(ic, 'bindings', return_value=htcondor):
      self.assertEqual(ic.status(EXTERNAL_ID), 'failed')
    htcondor.Schedd.return_value.act.assert_called_once_with('Remove', ['123.0'])
    htcondor, _ = self.fake_bindings([Event(TYPES.JOB_HELD), Event(TYPES.JOB_RELEASED)])
    with patch.object(ic, 'bindings', return_value=htcondor):
      self.assertEqual(ic.status(EXTERNAL_ID), 'running')
    htcondor.Schedd.assert_not_called()
    htcondor, _ = self.fake_bindings([Event(TYPES.JOB_HELD)])
    htcondor.Schedd.return_value.act.side_effect = RuntimeError('scheduler unavailable')
    with patch.object(ic, 'bindings', return_value=htcondor):
      self.assertEqual(ic.status(EXTERNAL_ID), 'running')
    for failure in ['TotalError', 'TotalPermissionDenied', 'TotalBadStatus']:
      htcondor, _ = self.fake_bindings([Event(TYPES.JOB_HELD)])
      htcondor.Schedd.return_value.act.return_value = {failure: 1}
      with patch.object(ic, 'bindings', return_value=htcondor):
        self.assertEqual(ic.status(EXTERNAL_ID), 'running')

  def test_cancel_converts_ids_and_rejects_invalid_ids_before_removal(self):
    htcondor, _ = self.fake_bindings()
    with patch.object(ic, 'bindings', return_value=htcondor):
      ic.cancel([EXTERNAL_ID, f'8_{TOKEN}_124'])
    htcondor.Schedd.return_value.act.assert_called_once_with('Remove', ['123.0', '124.0'])
    with self.assertRaises(ValueError):
      ic.split_id('../../other_job')

  def test_default_log_root_uses_jobs_dir(self):
    with patch.dict(os.environ, {'JOBS_DIR': '/analysis/jobs', 'INNFER_CONDOR_LOG_DIR': ''}):
      self.assertEqual(ic.log_root(), Path('/analysis/jobs/condor'))

  @unittest.skipUnless(importlib.util.find_spec('htcondor'), 'HTCondor Python bindings are not installed')
  def test_real_bindings_parse_success_and_signal_events(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      directory = root/f'7_{TOKEN}'
      directory.mkdir()
      log = directory/'condor.log'
      for termination, expected in [('(1) Normal termination (return value 0)', 'success'),
                                    ('(0) Abnormal termination (signal 9)', 'failed')]:
        core = '\t(0) No core file\n' if expected == 'failed' else ''
        usage = ''.join(f'\t\tUsr 0 00:00:00, Sys 0 00:00:00  -  {kind} Usage\n'
          for kind in ['Run Remote', 'Run Local', 'Total Remote', 'Total Local'])
        transfer = ''.join(f'\t0  -  {kind}\n' for kind in ['Run Bytes Sent By Job',
          'Run Bytes Received By Job', 'Total Bytes Sent By Job', 'Total Bytes Received By Job'])
        log.write_text('005 (123.000.000) 11/19 12:00:00 Job terminated.\n\t'+termination+'\n'+core+usage+transfer+'...\n')
        with patch.dict(os.environ, {'INNFER_CONDOR_LOG_DIR': str(root)}):
          self.assertEqual(ic.status(EXTERNAL_ID), expected)


if __name__ == '__main__':
  unittest.main()
