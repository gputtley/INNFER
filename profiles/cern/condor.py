#!/usr/bin/env python3
"""Snakemake cluster-generic adapter using the lxplus Condor command line tools."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


FLAVOURS = [(20, 'espresso'), (60, 'microcentury'), (120, 'longlunch'),
            (480, 'workday'), (1440, 'tomorrow'), (4320, 'testmatch'), (10080, 'nextweek')]


def flavour(minutes):
    for limit, name in FLAVOURS:
        if 0 < minutes <= limit:
            return name
    raise ValueError('CERN retry runtime must be between 1 minute and 7 days')


def read_properties(jobscript):
    for line in Path(jobscript).read_text().splitlines():
        if line.startswith('# properties = '):
            return json.loads(line.split(' = ', 1)[1])
    raise ValueError(f'No Snakemake properties in {jobscript}')


def submit_description(jobscript, directory, properties):
    # Arguments use Condor's new argument syntax: double quotes around the
    # whole argument list and single quotes around each argument with spaces.
    script = str(Path(jobscript).resolve())
    arguments = '"' + ("'" + script.replace("'", "''") + "'").replace('"', '""') + '"'
    settings = {
        'universe': 'vanilla', 'executable': '/bin/bash', 'arguments': arguments,
        'initialdir': str(Path.cwd()), 'getenv': 'True',
        'should_transfer_files': 'NO',
        'output': str(directory/'job.$(ClusterId).$(ProcId).out'),
        'error': str(directory/'job.$(ClusterId).$(ProcId).err'),
        'log': str(directory/'job.$(ClusterId).log'),
        'request_cpus': '1', 'request_memory': '4000', '+JobFlavour': '"workday"',
    }
    for option in properties.get('params', {}).get('submit_options', []):
        key, separator, value = option.partition('=')
        if not separator or not re.fullmatch(r'\+?[A-Za-z][A-Za-z0-9_]*', key.strip()) or '\n' in value or '\r' in value:
            raise ValueError(f'Invalid submission option: {option!r}')
        key = key.strip()
        key = key if key.startswith('+') else key.lower()
        aliases = {'requestcpus': 'request_cpus', 'requestmemory': 'request_memory',
                   'requestdisk': 'request_disk', 'requestgpus': 'request_gpus'}
        key = aliases.get(key, key)
        settings[key] = value.strip()
    resources = properties.get('resources', {})
    runtime = int(resources.get('retry_runtime', 0))
    if runtime:
        # A CERN flavour replaces the original runtime on subsequent attempts.
        for key in list(settings):
            if key.lower() in ['+maxruntime', '+jobflavour']:
                del settings[key]
        settings['+JobFlavour'] = json.dumps(flavour(runtime))
    memory = int(resources.get('retry_memory', 0))
    if memory:
        if memory < 0:
            raise ValueError('retry_memory must be positive')
        settings['request_memory'] = str(memory)
    if settings['should_transfer_files'].upper() != 'NO':
        raise ValueError('The CERN profile uses shared filesystems; should_transfer_files must be NO')
    # nproc can report the entire worker, rather than the CPUs requested by
    # this payload. Bound library thread pools to the actual allocation.
    cpus = int(settings['request_cpus'])
    if cpus <= 0:
        raise ValueError('request_cpus must be positive')
    settings.setdefault('environment', json.dumps(' '.join(
        f'{name}={cpus}' for name in ['OMP_NUM_THREADS', 'TF_NUM_INTEROP_THREADS', 'TF_NUM_INTRAOP_THREADS'])))
    return '\n'.join(f'{key} = {value}' for key, value in settings.items()) + '\nqueue 1\n'


def log_root():
    return Path(os.environ.get('INNFER_CONDOR_LOG_DIR', str(Path.home()/'.cache/innfer/condor'))).resolve()


def submit(jobscript, dry_run=False):
    root = log_root()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='job-', dir=root))
    description = directory/'job.sub'
    description.write_text(submit_description(jobscript, directory, read_properties(jobscript)))
    if dry_run:
        print(description)
        return
    result = subprocess.run(['condor_submit', '-terse', str(description)], check=True, text=True, capture_output=True)
    match = re.fullmatch(r'\s*(\d+\.\d+)(?:\s*-\s*\d+\.\d+)?\s*', result.stdout)
    if not match:
        raise RuntimeError(f'Unexpected condor_submit output: {result.stdout!r}')
    job_id = match.group(1)
    (directory/'submission.json').write_text(json.dumps({'job_id': job_id, 'jobscript': str(Path(jobscript).resolve())}, indent=2))
    print(job_id)


def query(command):
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def status(job_id):
    if not re.fullmatch(r'\d+\.\d+', job_id):
        raise ValueError(f'Invalid Condor job ID: {job_id}')
    attributes = 'JobStatus,ExitCode,ExitBySignal,HoldReason,RemoveReason'
    try:
        jobs = query(['condor_q', job_id, '-json', '-attributes', attributes])
        if not jobs:
            jobs = query(['condor_history', job_id, '-match', '1', '-json', '-attributes', attributes])
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f'Condor status temporarily unavailable for {job_id}: {error}', file=sys.stderr)
        return 'running'
    if not jobs:
        # Queue removal can precede the record becoming visible in history.
        return 'running'
    job = jobs[0]
    state = job.get('JobStatus')
    if state == 4:
        return 'success' if job.get('ExitCode') == 0 and not job.get('ExitBySignal', False) else 'failed'
    if state == 3:
        print(f'{job_id} removed: {job.get("RemoveReason", "unknown reason")}', file=sys.stderr)
        return 'failed'
    if state == 5:
        print(f'{job_id} held: {job.get("HoldReason", "unknown reason")}', file=sys.stderr)
        # Remove the held copy before Snakemake submits a replacement, avoiding
        # two jobs later writing to the same outputs.
        removed = subprocess.run(['condor_rm', job_id], text=True, capture_output=True)
        if removed.returncode:
            print(removed.stderr, file=sys.stderr)
            return 'running'
        return 'failed'
    return 'running'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    submission = commands.add_parser('submit')
    submission.add_argument('--dry-run', action='store_true', help='Write the submit file without submitting')
    submission.add_argument('jobscript')
    commands.add_parser('status').add_argument('job_id')
    args = parser.parse_args()
    if args.command == 'submit':
        submit(args.jobscript, args.dry_run)
    else:
        print(status(args.job_id))


if __name__ == '__main__':
    main()
