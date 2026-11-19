---
layout: page
title: "Building the Apptainer image"
---

The image contains **both the current software environment and a frozen source/configuration snapshot**. It does not contain datasets, trained models, job outputs, credentials, the host checkout's `.git` directory, or the live conda package cache. Building an image does not change existing jobs. Opt-in batch submission is described below; dataset staging remains separate.

For the CERN lxplus binned workflow, see [CERN Condor submission](cern_submission.md).

## Build

Run from the repository, with the working environment selected explicitly:

```bash
python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir "$PWD/tmp/apptainer-build" \
  --output "$PWD/images/innfer-20261118.sif"
```

Prefer a node-local `--work-dir` when available. The builder requires 30 GiB free by default, and large environments can need more. The build/cache/unpacked files all use this directory; a small system `/tmp` is unsuitable. The example uses shared storage for this one-off build, not for future job execution. `--min-free-gb` controls the initial space check.

The builder creates an isolated virtual environment for conda-pack 0.8.1 and setuptools 80.9.0. It reads and packages the selected environment without installing into or modifying it. Network access is needed for these small build tools, the base OS image and OS packages. The default base is `rockylinux:9`; `--base-image` can select a tag or Docker digest. The completed SIF is immutable, but rebuilding against a mutable tag/repository is not guaranteed to reproduce identical bytes.

The source snapshot includes ignored/uncommitted run and architecture configs, active worker/runner modules, scripts, tests, and the actual local `python/BayesFlow/bayesflow` source. Each file's hash is recorded. The environment is relocated to `/opt/innfer/env`; BayesFlow imports use the snapshot rather than editable paths back to `/vols/...`. Additional editable packages cause a clear error until their sources are explicitly included.

The builder audits conda's expected files before packaging. Old `.dist-info` metadata left by a pip upgrade is accepted only when a different version of that same Python distribution is installed; each exception is recorded in the manifest. Other missing conda-managed files fail the build. This handles the working environment's Snakemake interface upgrades without repairing or modifying the environment used by running jobs.

Inspect the source snapshot and plan without packing/building:

```bash
python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir "$PWD/tmp/apptainer-build" \
  --prepare-only
```

The printed build directory contains `code.tar.gz`, `container-manifest.json`, the copied definition and `build-plan.json`. Full builds additionally contain the environment archive and isolated build tools. Nothing deletes these directories automatically. Existing output images are never overwritten; choose a new image filename for a new version.

If the environment was packaged successfully but a later build/test failed, retry with its saved archive to avoid packaging it again:

```bash
python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir "$PWD/tmp/apptainer-build" \
  --reuse-environment-from "$PWD/tmp/apptainer-build/<previous-build-directory>" \
  --output "$PWD/images/innfer-20261118.sif"
```

This requires the saved environment inventory to match the selected environment. It freezes the current code/recipe afresh and records the reused archive's location; it does not reuse the old source snapshot. If you changed package contents without changing package versions, package a fresh environment instead.

`bash setup.sh container` is also wired to the builder. Its work directory can be selected with `INNFER_CONTAINER_BUILD_DIR`; the explicit Python command above is preferable when choosing the source environment and output filename. The disabled cleanup at the end of setup.sh stays disabled.

## Validation and image contents

The build installs the OS libraries, fonts and LaTeX packages needed by the application. It runs a CPU smoke test that checks imports from the image, both BayesFlow and KerasFlow checkpoint round trips/density evaluation/sampling, parquet I/O, PDF table compilation and `innfer --help`. A second test disables host home/current-directory/site bind mounts, so missing dependencies cannot be satisfied accidentally by the original checkout.

The recipe creates the system `pdflatex.fmt` during the writable build phase. Generating it on first use would fail when the finished image is read-only, especially under fakeroot where TeX tries to write beneath `/root`.

After both tests pass the `.partial` image is published to the requested `.sif` path, alongside `.sif.sha256` and `.sif.json` provenance files. The image itself contains `/opt/innfer/app/container-manifest.json` and an OS-package list. Failed builds retain their build context and any partial image for diagnosis; they do not replace an existing image.

Failed builds also retain Apptainer's bundle under the build context's `temporary` directory. Failed LaTeX/CLI smoke checks print the tool's captured output before raising an error, so the build log includes the underlying failure.

Run the CLI and inspect the frozen version:

```bash
apptainer run --cleanenv images/innfer-20261118.sif --help
apptainer exec --cleanenv images/innfer-20261118.sif \
  cat /opt/innfer/app/container-manifest.json
```

On an allocated GPU node, validate NVIDIA access separately:

```bash
apptainer exec --nv --cleanenv images/innfer-20261118.sif \
  python /opt/innfer/app/containers/smoke_test.py --require-gpu
```

The host supplies the NVIDIA driver; TensorFlow/CUDA userspace dependencies and CUDA libdevice are packaged in the environment. A successful CPU build does not claim GPU compatibility with every execution node.

The run entry point uses `/opt/innfer/app/scripts/innfer.py`. When running actual analyses, select frozen configs via `/opt/innfer/app/configs/...`; a relative `configs/...` argument resolves against the runtime working directory and can select a mounted host config instead. Data/model/output paths remain external and are controlled by the existing directory variables. This image alone does not create a local data sandbox, publish results or change the Snakemake shared-filesystem assumptions.

## Submit jobs with an image

Add `--container-image` to the normal host-side submission command. It is disabled by default and works for direct Condor submission and generated Snakemake jobs. The host removes this submission-only option from worker arguments, so the existing image can be used without rebuilding it for the launcher.

Prepare a 2223 likelihood diagnostic without submitting it:

```bash
source env.sh
JOBS_DIR=jobs/container_check python3 scripts/innfer.py \
  --cfg=configs/run/btm_full_171126_sbi.py \
  --step=LikelihoodDebug \
  --specific-file-name=combined \
  --specific-combined-default-val \
  --specific-category=2223 \
  --data-type=asimov \
  --extra-asimov-input-dir-name=StatOnlyLocalScan10 \
  --classifier-divide-by-nominal \
  --prune-classifier-models=q75:1.01 \
  --prune-from=ClassifierNuisanceVariationsTestInf \
  --integrate-density-with-ratios \
  --debug-input='bw_mass=172.5;bw_mass=172.6' \
  --container-image=images/innfer-20261118.sif \
  --submit=configs/submit/condor_gpu.yaml \
  --dry-run
```

Inspect the `.sh`, `.sub` and `.container.json` in `jobs/container_check/BTM_Full_171126/`. Remove `--dry-run` to submit. To test locally, execute the generated `.sh` with `bash` on a machine with the requested GPU and sufficient local scratch.

For a workflow, pass `--container-image` alongside `--step=SnakeMake` and `--snakemake-cfg`, or set it in `run_options`:

```yaml
- workflow: configs/snakemake/subworkflow/inference_asimov_condor_ic.yaml
  run_options:
    container_image: images/innfer-20261118.sif
```

A child can use `container_image: null` to retain ordinary host execution. Regenerate the Snakefile and scripts for changes to take effect; existing running workflows are not modified.

The launcher copies the image to a checksum-verified cache in worker scratch, using a lock and atomic publication for concurrent jobs. Its default location is under `_CONDOR_SCRATCH_DIR`, then `TMPDIR`, then `/tmp`. `INNFER_CONTAINER_CACHE` in the worker environment can select a persistent node-local cache. Local space must accommodate the image and runtime files.

The selected run/architecture files, supporting architecture configs and `configs/other` are archived at job generation. Each worker checks and extracts a private copy. Thus jobs use submission-time configs with the image's code. Private config copies permit runtime architecture overrides without editing the archived snapshot. Rebuild the image when changing Python worker code that jobs need.

GPU requests in the submission YAML automatically enable `--nv`; the worker's GPU visibility allocation is preserved. CPU jobs disable CUDA visibility. Existing GPU counts, memory, runtime and retry settings are retained. If the submit file has no disk request, the generated submission requests space for the image/config archive plus 1 GiB of runtime scratch; an explicit `RequestDisk` takes precedence. The launcher invokes image code without activating host conda. Its normal home mount remains available for existing credentials; credentials are not packaged into the image or config archive.

Each job gets a `.container.json` recording the image/config hashes, commands, directory settings and GPU mode. Snakemake declares the image and config archive as inputs. Worker failures stop grouped commands and propagate to the scheduler.

**Data, models and outputs remain on shared storage**, using the existing directory variables. This launcher does not implement dataset staging or copying completed results from a full scratch workspace.

Implementation: [builder](../scripts/build_apptainer.py), [recipe](../containers/innfer.def), [runtime environment](../containers/environment.sh), [smoke test](../containers/smoke_test.py), [submission launcher](../python/worker/container_submission.py).

References: [Apptainer definition files](https://apptainer.org/docs/user/1.5/definition_files.html), [GPU support](https://apptainer.org/docs/user/main/gpu.html), [conda-pack relocation](https://conda.github.io/conda-pack/cli.html).
