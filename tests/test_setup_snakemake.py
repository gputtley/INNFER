"""Exercise site setup without installing packages or touching real profiles."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SnakeMakeSetupTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.base = Path(self.temporary.name)
    self.project = self.base/'project'
    self.project.mkdir()
    shutil.copyfile(ROOT/'setup.sh', self.project/'setup.sh')
    shutil.copytree(ROOT/'profiles/cern', self.project/'profiles/cern',
      ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'profiles/ic', self.project/'profiles/ic',
      ignore=shutil.ignore_patterns('__pycache__'))
    self.binary = self.base/'bin'
    self.binary.mkdir()
    # Restrict PATH so absent CERN tools cannot be supplied by the test host.
    (self.binary/'mkdir').symlink_to(shutil.which('mkdir'))
    self.log = self.base/'calls.txt'
    conda = self.base/'conda install/files/etc/profile.d/conda.sh'
    conda.parent.mkdir(parents=True)
    conda.write_text('conda() { printf "%s\\n" "conda $*" >> "$FAKE_SETUP_LOG"; }\n')
    self.environment = {**os.environ, 'PATH': str(self.binary),
      'MINICONDA_BASE': str(self.base/'conda install'),
      'HOME': str(self.base/'home'), 'XDG_CONFIG_HOME': str(self.base/'config space'),
      'FAKE_SETUP_LOG': str(self.log), 'FAKE_PLUGIN_MISSING': '0'}
    self.executable('python3', '''
import os, pathlib, sys
with open(os.environ['FAKE_SETUP_LOG'], 'a') as stream:
    stream.write('python3 '+ ' '.join(sys.argv[1:])+'\\n')
args = sys.argv[1:]
if args and args[0] == '-c':
    if 'distribution(' in args[1]:
        sys.exit(int(os.environ['FAKE_PLUGIN_MISSING']))
    if args[1] == 'import htcondor':
        sys.exit(0)
if args[:2] == ['-m', 'pip']:
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *args])
''')
    for tool in ['condor_submit', 'condor_q', 'condor_history', 'condor_rm', 'apptainer']:
      self.executable(tool, 'raise SystemExit("Site setup must not execute batch commands")\n')

  def executable(self, name, code):
    path = self.binary/name
    path.write_text('#!'+sys.executable+'\n'+code)
    path.chmod(0o755)

  def run_setup(self, target, sourced=False):
    if sourced:
      command = ['/bin/bash', '-c',
        'source setup.sh "$1"; setup_result=$?; echo SOURCE_RETURNED; exit "$setup_result"', 'test', target]
    else:
      command = ['/bin/bash', 'setup.sh', target]
    return subprocess.run(command, cwd=self.project, env=self.environment,
      text=True, capture_output=True)

  def test_cern_validates_checkout_without_creating_or_changing_ic_profile(self):
    existing = Path(self.environment['XDG_CONFIG_HOME'])/'snakemake/htcondor/config.yaml'
    existing.parent.mkdir(parents=True)
    existing.write_text('existing IC customization\n')
    before = {path.name: path.read_bytes() for path in (self.project/'profiles/cern').iterdir()}
    result = self.run_setup('snakemake_cern')
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('--snakemake-profile=profiles/cern', result.stdout)
    self.assertEqual(existing.read_text(), 'existing IC customization\n')
    self.assertEqual(before, {path.name: path.read_bytes() for path in (self.project/'profiles/cern').iterdir()})
    calls = self.log.read_text()
    self.assertIn('conda activate innfer_env', calls)
    self.assertNotIn('-m pip', calls)
    self.assertNotIn('import htcondor', calls)
    self.assertNotIn('cookiecutter', calls)

  def test_missing_plugin_is_installed_by_site_setup(self):
    self.environment['FAKE_PLUGIN_MISSING'] = '1'
    result = self.run_setup('snakemake_cern')
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('python3 -m pip install snakemake-executor-plugin-cluster-generic==1.0.9', self.log.read_text())

  def test_ic_and_old_alias_use_bundled_profile_and_preserve_personal_profile(self):
    profile = Path(self.environment['XDG_CONFIG_HOME'])/'snakemake/htcondor/config.yaml'
    profile.parent.mkdir(parents=True)
    profile.write_text('preserve customized profile\n')
    result = self.run_setup('snakemake_ic')
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('--snakemake-profile=profiles/ic', result.stdout)
    self.assertIn('IC profile ready:', result.stdout)
    result = self.run_setup('snakemake_condor', sourced=True)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('--snakemake-profile=profiles/ic', result.stdout)
    self.assertIn('SOURCE_RETURNED', result.stdout)
    self.assertEqual(profile.read_text(), 'preserve customized profile\n')
    self.assertNotIn('cookiecutter', self.log.read_text())

  def test_ic_setup_needs_no_personal_profile_directory(self):
    result = self.run_setup('snakemake_ic')
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertFalse(Path(self.environment['XDG_CONFIG_HOME']).exists())
    self.assertNotIn('cookiecutter', self.log.read_text())

  def test_missing_condor_command_fails_without_exiting_sourcing_shell(self):
    (self.binary/'condor_history').unlink()
    result = self.run_setup('snakemake_cern', sourced=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('Missing required command: condor_history', result.stderr)
    self.assertIn('SOURCE_RETURNED', result.stdout)
    self.assertNotIn('CERN profile ready', result.stdout)

  def test_missing_or_invalid_profile_fails_setup(self):
    for site in ['ic', 'cern']:
      config = self.project/f'profiles/{site}/config.yaml'
      config.write_text('executor: local\n')
      result = self.run_setup('snakemake_'+site)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn(f'Invalid {site.upper()} profile', result.stderr)
      config.unlink()
      result = self.run_setup('snakemake_'+site)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn(f'Missing {site.upper()} profile file', result.stderr)


if __name__ == '__main__':
  unittest.main()
