---
layout: page
title: "CERN Condor submission"
---

The CERN profile runs INNFER workflows through lxplus Condor and the Apptainer
launcher. It can be used with any run config and workflow. Inputs, models and
outputs remain on shared POSIX filesystems, including your mounted EOS paths. No dataset transfers or
staging are added. Jobs copy the SIF and frozen job configuration to local
scratch before starting Apptainer, as in the existing container launcher.

## Prepare the checkout and image

You can build the image directly on lxplus, or copy an image built at IC.
In either case, keep the checkout and a current SIF accessible at CERN. The
image must include the Python modules and dependencies used by your config.
When copying an image, include its `.sif.sha256` and `.sif.json`
sidecars. When building at CERN, no image transfer is needed.

To build on lxplus, run from the checkout with the INNFER environment installed:

```bash
source env.sh
container_build_dir=$(mktemp -d /tmp/innfer-apptainer.XXXXXX)

python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir "$container_build_dir" \
  --base-image quay.io/rockylinux/rockylinux:9 \
  --output "$PWD/images/innfer-20261120-cern-local.sif"
```

Use a local filesystem with at least 30 GiB free for `--work-dir`. The checkout
and final SIF may stay on AFS, but extracting the base image beneath AFS failed
with `invalid cross-device link`. Quay avoids the Docker Hub HTTP 429 error
encountered on the first CERN build.

If an earlier attempt already produced `environment.tar.gz`, add
`--reuse-environment-from /absolute/path/to/previous-build-directory` to reuse
it. Keep the old archive available and select a new output name if a previous
`.sif` or `.partial` file exists. See [IC/CERN container builds and troubleshooting](container.md)
for the full retry command and an explanation of the fakeroot messages.

On lxplus, the controller needs the normal INNFER Python environment, Snakemake
and its `cluster-generic` executor plugin. The controller runs outside the
container and must have `condor_submit`, `condor_q`, `condor_history`, and
`condor_rm` available. Workers need Apptainer and access to the shared paths;
they use Python and dependencies from the SIF. No Python `htcondor` binding is
required by the new profile. After creating the environment, run the CERN
setup target from the repository root:

```bash
bash setup.sh snakemake_cern
source env.sh
```

This ensures the shared executor plugin is installed, checks the Condor
commands and Apptainer, validates `profiles/cern`, and prints the profile
argument to use. No separate pip command or IC profile installation is
needed. Full installation instructions are in [Setup](setup.md).
You can also check the installation with:

```bash
snakemake --version
snakemake --help | grep cluster-generic-submit-cmd
command -v condor_submit condor_q condor_history condor_rm apptainer
```

## Configure inputs and submission resources

Copy or make available the inputs required by your analysis: raw files, models,
calibrations and other auxiliary files. Configure their CERN paths in your run
config and any supporting configs before generating the workflow. Relative
paths resolve against the project working directory; absolute mounted EOS paths
can also be used. The selected run config is snapshotted for submitted jobs.

The container launcher has no analysis-specific path settings. External paths
must be visible inside the container; use the site's mounts or Apptainer's
standard [bind settings](https://apptainer.org/docs/user/main/bind_paths_and_mounts.html),
when an extra mount is needed. For input files on EOS, set this before starting
the controller so the setting is passed to newly submitted jobs:

```bash
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}/eos:/eos"
```

Use a literal colon, with no backslash before it. `APPTAINER_BINDPATH=/eos` is
equivalent when no other extra bindings are needed. The EOS path must also be
accessible on the host worker. Existing queued jobs retain their submission-time
environment; editing this variable in another terminal does not update them.

### EOS authentication inside the container

If the EOS path is accessible directly on lxplus but gives `PermissionError`
inside the image, check the Kerberos cache on the same machine:

```bash
klist
printf 'KRB5CCNAME=%s\n' "${KRB5CCNAME:-unset}"
```

EOS can use `KRB5CCNAME` from the process accessing its FUSE mount. The
container's `--cleanenv` removes host variables unless they are passed
explicitly, and a cache under `/run/user/...` needs a bind mount to be visible.
The generated launcher forwards the current worker's `KRB5CCNAME` and
`KRB5_CONFIG`, and mounts readable file-based caches and explicitly selected
configuration files read-only at their original paths. Credentials are read
at runtime and are not included in the SIF, config archive or job manifest.

Regenerate job scripts after updating `python/worker/container_submission.py`
to pick up this behaviour; rebuilding the SIF is unnecessary. For an existing
script, a local test with a `FILE:` cache can use:

```bash
APPTAINERENV_KRB5CCNAME="$KRB5CCNAME" \
APPTAINER_BINDPATH="/eos,${KRB5CCNAME#FILE:}:${KRB5CCNAME#FILE:}:ro" \
bash /path/to/generated/job.sh
```

Use these command-scoped settings for the local test. Batch workers need their
own usable credentials; binding the cache from a login node does not delegate
its tickets to another machine. If EOS access also fails outside the container,
check the host ticket and EOS permissions first.
See the [EOS authentication explanation](https://eos-community.web.cern.ch/t/fusex-kerberos-authentication-inside-docker-for-version-v4-4-18/326)
and [Apptainer environment controls](https://apptainer.org/docs/user/main/environment_and_metadata.html).

### Submission resources

Each selected workflow step needs CERN submission settings. You can set its
`submit` entry directly to a suitable submission YAML, or retain an existing
workflow and provide `--submit-config-map`. For example, save this mapping as
`configs/submit/cern_map.yaml`, extending it for every submission YAML used by
your workflow:

```yaml
configs/submit/condor.yaml: configs/submit/cern_1_cpu.yaml
configs/submit/condor_2_cpus_avx.yaml: configs/submit/cern_2_cpus.yaml
configs/submit/condor_4_cpus_avx.yaml: configs/submit/cern_4_cpus.yaml
```

The bundled `cern_1_cpu.yaml`, `cern_2_cpus.yaml` and `cern_4_cpus.yaml` are CPU
examples. For a GPU step, create a CERN submission YAML with `RequestGpus` and
the queue, CPU, memory and disk settings needed by that step, then select it
directly or add it to the map. Preserve the resource needs of the computation
when translating between sites.

Missing map entries fail during workflow generation; unmapped steps do not
silently retain their IC settings. Paths in the map are resolved from the
project working directory. The profile alone does not translate IC submission
settings into CERN settings.

## Run your workflow

From the project root on lxplus, activate the controller environment and source
`env.sh` (use `source env.sh no_conda` if your environment is already active).
Replace `your_analysis.py` and `your_workflow.yaml` with your run config and
workflow, and create the submission map described above. These placeholder
paths and `cern_map.yaml` are examples to adapt, not generated setup files.

```bash
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}/eos:/eos"

python3 scripts/innfer.py \
  --cfg=configs/run/your_analysis.py \
  --step=SnakeMake \
  --snakemake-cfg=configs/snakemake/your_workflow.yaml \
  --snakemake-profile=profiles/cern \
  --submit-config-map=configs/submit/cern_map.yaml \
  --container-image="$PWD/images/innfer-20261120-cern-local.sif" \
  --snakemake-rerun-incomplete
```

Omit `--submit-config-map` if your workflow already selects CERN submission
YAMLs directly. Keep the controller running in tmux. The selected workflow
determines which processing, training, fitting or plotting steps run; choosing
the CERN profile does not add analysis steps.

To generate the job files without submitting, append `--snakemake-dry-run`.
This INNFER flag generates files only; it is not a Snakemake DAG dry run. Then
check dependencies with the generated Snakefile path printed by INNFER. For
the default `JOBS_DIR=jobs` and a config whose `name` is `your_run_name`:

```bash
snakemake --profile profiles/cern \
  -s jobs/your_run_name/innfer_SnakeMake.txt \
  --shared-fs-usage input-output --dry-run
```

## Settings and retries

The bundled CPU submission configs request 1, 2 or 4 CPUs, respectively 4000,
8000 or 16000 MB of memory, and `+JobFlavour = "workday"` (eight hours). Adjust these resource
settings after observing usage. Library thread pools are limited to the
requested CPU count. The container launcher adds its existing scratch disk
request based on the SIF size when no explicit disk request is present.

The profile allows two retries. To request more runtime on retries, set
`retry_runtime` in minutes on the relevant step or workflow wrapper. For example,
save a wrapper selecting your existing workflow and map:

```yaml
- workflow: configs/snakemake/your_workflow.yaml
  retry_runtime: 1440
  run_options:
    submit_config_map: configs/submit/cern_map.yaml
```

Select that wrapper with `--snakemake-cfg`. Here `retry_runtime: 1440` requests
`tomorrow` (one day) on retries, while first attempts use the original submission
YAML. `retry_memory` can similarly request memory in MB on retries; it is not
enabled by the profile automatically. Held jobs are removed before being
reported as failed so that a later release cannot duplicate the replacement.
The hold reason is printed to the controller's stderr. Completed jobs must
have zero exit status and no terminating signal to count as successful.
Temporary query failures and the queue-to-history visibility gap remain
`running`, rather than triggering duplicate submissions.

A successful Condor query may return blank output when there are no matching
jobs. The status checker treats this as an empty result and checks history
when the queue no longer contains the job. If both are empty, it waits for the
next poll. Nonempty malformed responses still produce a warning identifying
which Condor command failed to return job records. This checker runs on the
host for each poll, so updating `profiles/cern/condor.py` takes effect in an
existing workflow without rebuilding the image or regenerating the Snakefile.

Condor submit files, scheduler output/error logs and submission records go to
`$HOME/.cache/innfer/condor/job-*` by default. Set `INNFER_CONDOR_LOG_DIR` to
another suitable shared controller/worker log directory before starting the
workflow if needed. Job scripts, container manifests and application logs are
distributed under `jobs/<run>/<step>/<group>/`, for both container and ordinary
submissions. Each group holds up to 500 numbered jobs, avoiding AFS
per-directory entry limits within large steps. The generated Snakefile remains
at `jobs/<run>/innfer_SnakeMake.txt`.
The profile launches Apptainer through INNFER; it does not request a second
container layer from Condor.

The generic controller flags are `--snakemake-profile` (default `profiles/ic`),
`--snakemake-shared-fs-usage` (default `input-output`) and `--submit-config-map`.
They are not forwarded to worker INNFER commands. The profile's helper can
write an individual submit file without sending it to Condor:

```bash
python3 profiles/cern/condor.py submit --dry-run /path/to/snakemake-jobscript.sh
```

The shared filesystem behavior follows [HTCondor's `should_transfer_files = NO`
mode](https://htcondor.readthedocs.io/en/main/users-manual/file-transfer.html).
The adapter uses the supported [Snakemake cluster-generic profile
hooks](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/cluster-generic.html).
