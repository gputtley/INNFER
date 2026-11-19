---
layout: page
title: "CERN Condor submission"
---

The CERN profile runs the existing binned physics workflow through lxplus
Condor and the Apptainer launcher. Inputs, models and outputs remain on shared
POSIX filesystems, including your mounted EOS paths. No dataset transfers or
staging are added. Jobs copy the SIF and frozen job configuration to local
scratch before starting Apptainer, as in the existing container launcher.

## Prepare the checkout and image

Copy the updated checkout and a newly built SIF, its `.sha256` and `.json`
sidecars to CERN. Rebuild the image after these changes: the old image's common
BTM configuration explicitly rejects `.cern.ch` hosts. The image now includes
the portable common configuration and the versioned profile.

For example, build on the machine with the existing environment archive:

```bash
python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir /dev/shm/innfer-apptainer-$USER \
  --reuse-environment-from "$PWD/tmp/apptainer-build/innfer-20261118T113737Z-jgd0y1rw" \
  --output "$PWD/images/innfer-20261119-cern.sif"
```

Omit `--reuse-environment-from` if that archive is unavailable. Choose a local
build directory with adequate space; see [the container instructions](container.md).

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

Copy the raw inputs yourself. Either retain the project layout
`data/top_reco/270726_parquet_with_psweights`, or set `TOP_RECO_DIR` to their
new location, e.g. an absolute mounted EOS directory. This value is captured
when the jobs are generated and is forwarded through the container's clean
environment. External absolute input roots are also bound into the container.
Leave the auxiliary BDT files in the project at:

```text
data/merged_ttbar_bdt/ttbar_bdt_model_<era>.json
data/merged_ttbar_bdt/ttbar_bdt_features_<era>.yaml
data/merged_ttbar_bdt/ttbar_bdt_working_points_<era>.yaml
```

## Run the full binned workflow

From the project root on lxplus, activate the controller environment and source
`env.sh` (use `source env.sh no_conda` if your environment is already active).
If the raw inputs are elsewhere, set `TOP_RECO_DIR` after sourcing it.

```bash
python3 scripts/innfer.py \
  --cfg=configs/run/btm_full_171126_binned.py \
  --step=SnakeMake \
  --snakemake-cfg=configs/snakemake/btm_binned_cern.yaml \
  --snakemake-profile=profiles/cern \
  --container-image="$PWD/images/innfer-20261119-cern.sif" \
  --snakemake-rerun-incomplete
```

Keep this controller running in tmux. The workflow includes preprocessing,
input validation, prefit plots, and the combined and per-era Asimov fits,
scans, impacts and postfit plots already defined by `btm_binned.yaml`.

To generate the job files without submitting, append `--snakemake-dry-run`.
This INNFER flag generates files only; it is not a Snakemake DAG dry run. Then
check dependencies with:

```bash
snakemake --profile profiles/cern \
  -s jobs/BTM_Full_171126_Binned/innfer_SnakeMake.txt \
  --shared-fs-usage input-output --dry-run
```

## Settings and retries

`configs/snakemake/btm_binned_cern.yaml` includes the existing workflow and
adds `submit_config_map`. The map in `configs/submit/cern_binned_map.yaml`
replaces every selected IC submission config with a CERN config. Missing map
entries fail during generation rather than silently inheriting IC settings.
The GPU submission used by the dataset split is mapped to four CPUs; no GPU
is requested anywhere in this binned workflow. IC workflows remain unchanged.

The CERN configs request 1, 2 or 4 CPUs, respectively 4000, 8000 or 16000 MB
of memory, and `+JobFlavour = "workday"` (eight hours). Adjust these resource
settings after observing usage. Library thread pools are limited to the
requested CPU count. The container launcher adds its existing scratch disk
request based on the SIF size when no explicit disk request is present.

The profile allows two retries. The wrapper's `retry_runtime: 1440` requests
`tomorrow` (one day) on retries. Existing `retry_memory` settings are also
supported; the workflow does not add one. Held jobs are removed before being
reported as failed so that a later release cannot duplicate the replacement.
The hold reason is printed to the controller's stderr. Completed jobs must
have zero exit status and no terminating signal to count as successful.
Temporary query failures and the queue-to-history visibility gap remain
`running`, rather than triggering duplicate submissions.

Condor submit files, scheduler output/error logs and submission records go to
`$HOME/.cache/innfer/condor/job-*` by default. Set `INNFER_CONDOR_LOG_DIR` to
another suitable shared controller/worker log directory before starting the
workflow if needed. Application logs retain the usual `jobs/<run>/` paths.
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
