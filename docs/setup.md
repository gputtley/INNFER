---
layout: page
title: "Setup"
---

## Cloning from github

The INNFER repository can be directly cloned from github with the following command

```bash
git clone --recurse-submodules https://github.com/gputtley/INNFER.git
```

## Installing conda and the environment

Running on the code relies on multiple python packages which are installed using a conda environment. Firstly, conda needs to set up. To install this you can run the following command. You may need to click `enter` and `yes` through all prompts. If you already have conda installed, this initial command may not be needed. 
```bash
source setup.sh conda
```

Next, you will need to set up the environment, with the following command.
```bash
source setup.sh env
```

## Setup SnakeMake

After creating the environment, choose the setup target for your batch service.
Run it from the repository root. Both targets activate `innfer_env` and ensure
the shared Snakemake executor plugin is installed.

For Imperial College HTCondor:

```bash
bash setup.sh snakemake_ic
```

This validates the bundled `profiles/ic` profile and ensures the HTCondor
Python bindings are available. Select it with `--snakemake-profile=profiles/ic`,
which is also INNFER's default. The old `snakemake_condor` setup target remains
an alias for IC setup. Running setup with no target retains the IC default.
No cookiecutter installation or external profile repository is required.

The IC profile preserves the existing limits (5000 jobs, 100 submissions and
status checks per second, and five Snakemake retries). Scheduler logs default
to `${JOBS_DIR:-jobs}/condor`; set `INNFER_CONDOR_LOG_DIR` to use another
location. Workflow `retry_runtime` remains minutes, converted to IC's
`+MaxRuntime` in seconds; `retry_memory` remains MB. Snakemake manages retries
so that each attempt receives its requested resources. Status polling reads
available log events without waiting, and held jobs are removed before a
replacement is submitted.

Existing personal profiles are left untouched. To keep using one, explicitly
pass its name or path, for example `--snakemake-profile=htcondor`. Already
running Snakemake controllers continue using the profile with which they
were launched.

For CERN lxplus:

```bash
bash setup.sh snakemake_cern
```

This checks the Condor commands and Apptainer, and validates the bundled CERN
profile in `profiles/cern`. Select it with
`--snakemake-profile=profiles/cern`. See [CERN submission](cern_submission.md)
for workflow submission examples and container requirements.

Both setup targets print the profile argument to use. They also work with
`source setup.sh <target>` when you want to retain environment activation in
your current terminal. Otherwise, run `source env.sh` after setup to activate
the environment for subsequent INNFER commands.

## Apptainer at IC and CERN

See [Apptainer at IC and CERN](container.md) for the image contents, separate
build commands, submission examples and failed-build recovery. Set `--work-dir`
to local scratch with sufficient space for the unpacked image and temporary
build files. You can run the command from an AFS checkout and save the finished
SIF on AFS, as in the CERN example.

The controller still needs the host INNFER environment and the appropriate
Snakemake profile above. Submitted workers use the image's environment and code.
For CERN workflow setup and submission, follow [CERN Condor submission](cern_submission.md).

<br>

---

Next: [Running INNFER](running.md).
