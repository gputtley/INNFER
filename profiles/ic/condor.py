#!/usr/bin/env python3
"""Bundled IC profile using HTCondor bindings and nonblocking event-log polling."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4


def bindings():
    import htcondor
    return htcondor


def log_root():
    default = Path(os.environ.get('JOBS_DIR') or 'jobs')/'condor'
    return Path(os.environ.get('INNFER_CONDOR_LOG_DIR') or default).resolve()


def read_properties(jobscript):
    for line in Path(jobscript).read_text().splitlines():
        if line.startswith('# properties = '):
            return json.loads(line.split(' = ', 1)[1])
    raise ValueError(f'No Snakemake properties in {jobscript}')


def submit_settings(jobscript, directory, properties):
    script = str(Path(jobscript).resolve())
    arguments = '"' + ("'" + script.replace("'", "''") + "'").replace('"', '""') + '"'
    settings = {
        'executable': '/bin/bash', 'arguments': arguments,
        'initialdir': str(Path.cwd()), 'getenv': 'True',
        'should_transfer_files': 'NO',
        'log': str(directory/'condor.log'),
        'output': str(directory/'condor.out'),
        'error': str(directory/'condor.err'),
    }
    for option in properties.get('params', {}).get('submit_options', []):
        key, separator, value = option.partition('=')
        if not separator or not re.fullmatch(r'\+?[A-Za-z][A-Za-z0-9_]*', key.strip()) or '\n' in value or '\r' in value:
            raise ValueError(f'Invalid submission option: {option!r}')
        settings[key.strip()] = value.strip()
    resources = properties.get('resources', {})
    for resource, key, multiplier in [('retry_runtime', '+MaxRuntime', 60), ('retry_memory', 'request_memory', 1)]:
        value = int(resources.get(resource, 0))
        if value < 0:
            raise ValueError(f'{resource} must be positive')
        if value:
            for old_key in list(settings):
                if old_key.lower().replace('_', '') == key.lower().replace('_', ''):
                    del settings[old_key]
            settings[key] = str(value*multiplier)
    # Snakemake owns retries, so every new attempt gets its requested retry
    # resources. Condor must not independently restart a failed copy as well.
    return settings


def split_id(job_id):
    match = re.fullmatch(r'(\d+)_([0-9a-f]{32})_(\d+)', job_id)
    if not match:
        raise ValueError(f'Invalid IC job ID: {job_id}')
    return match.groups()


def submit(jobscript, dry_run=False):
    properties = read_properties(jobscript)
    job_id = int(properties['jobid'])
    token = uuid4().hex
    directory = log_root()/f'{job_id}_{token}'
    directory.mkdir(parents=True, mode=0o700)
    settings = submit_settings(jobscript, directory, properties)
    record = directory/'submission.json'
    record.write_text(json.dumps({'jobscript': str(Path(jobscript).resolve()), 'settings': settings}, indent=2))
    if dry_run:
        print(record)
        return
    htcondor = bindings()
    result = htcondor.Schedd().submit(htcondor.Submit(settings), count=1)
    cluster = result.cluster()
    external_id = f'{job_id}_{token}_{cluster}'
    # Persist the accepted ID before printing it to Snakemake.
    record.write_text(json.dumps({'job_id': external_id, 'settings': settings}, indent=2))
    print(external_id)


def status(job_id):
    job, token, cluster = split_id(job_id)
    htcondor = bindings()
    state, reason, event_log = 'running', '', None
    try:
        event_log = htcondor.JobEventLog(str(log_root()/f'{job}_{token}'/'condor.log'))
        for event in event_log.events(stop_after=0):
            if event.cluster != int(cluster) or event.proc != 0:
                continue
            if event.type == htcondor.JobEventType.JOB_TERMINATED:
                state = 'success' if event.get('TerminatedNormally', False) and event.get('ReturnValue') == 0 else 'failed'
            elif event.type == htcondor.JobEventType.JOB_ABORTED:
                state = 'failed'
            elif event.type in [htcondor.JobEventType.JOB_HELD, htcondor.JobEventType.EXECUTABLE_ERROR]:
                state = 'remove'
                reason = event.get('HoldReason', event.get('Reason', 'held or executable error'))
            elif event.type == htcondor.JobEventType.JOB_RELEASED:
                state = 'running'
    except (OSError, RuntimeError) as error:
        # A missing/new/incompletely written log is not a failed payload.
        print(f'IC job log temporarily unavailable for {job_id}: {error}', file=sys.stderr)
        return 'running'
    finally:
        if event_log is not None:
            event_log.close()
    if state == 'remove':
        print(f'{job_id}: {reason}', file=sys.stderr)
        try:
            removed = htcondor.Schedd().act(htcondor.JobAction.Remove, [f'{cluster}.0'])
            if any(removed.get(key, 0) for key in ['TotalError', 'TotalPermissionDenied', 'TotalBadStatus']):
                print(f'Could not remove held IC job {cluster}.0', file=sys.stderr)
                return 'running'
        except RuntimeError as error:
            print(f'Could not remove held IC job {cluster}.0: {error}', file=sys.stderr)
            return 'running'
        return 'failed'
    return state


def cancel(job_ids):
    clusters = [f'{split_id(job_id)[2]}.0' for job_id in job_ids]
    htcondor = bindings()
    removed = htcondor.Schedd().act(htcondor.JobAction.Remove, clusters)
    if any(removed.get(key, 0) for key in ['TotalError', 'TotalPermissionDenied', 'TotalBadStatus']):
        raise RuntimeError('HTCondor could not cancel every requested IC job')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    submission = commands.add_parser('submit')
    submission.add_argument('--dry-run', action='store_true', help='Write submission settings without submitting')
    submission.add_argument('jobscript')
    commands.add_parser('status').add_argument('job_id')
    commands.add_parser('cancel').add_argument('job_ids', nargs='+')
    args = parser.parse_args()
    if args.command == 'submit':
        submit(args.jobscript, args.dry_run)
    elif args.command == 'status':
        print(status(args.job_id))
    else:
        cancel(args.job_ids)


if __name__ == '__main__':
    main()
