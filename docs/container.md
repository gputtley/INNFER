---
layout: page
title: "Apptainer at IC and CERN"
---

The image contains **both the current software environment and a frozen source/configuration snapshot**. It does not contain datasets, trained models, job outputs, credentials, the host checkout's `.git` directory, or the live conda package cache. Building an image does not change existing jobs. Opt-in batch submission is described below; dataset staging remains separate.

Build and submit from a checkout with the usual INNFER controller environment.
The controller runs outside the image; submitted workers use the environment
and code inside it. The same image can be copied between IC and CERN, provided
the required input paths are configured for the destination site.

| Setting | Imperial College | CERN lxplus / HTCondor |
| --- | --- | --- |
| Profile setup | `bash setup.sh snakemake_ic` | `bash setup.sh snakemake_cern` |
| Snakemake profile | `profiles/ic` (default) | `profiles/cern` |
| Base image | Default `rockylinux:9`; Quay is also available | Prefer `quay.io/rockylinux/rockylinux:9` to avoid Docker Hub throttling |
| Temporary build files (`--work-dir`) | Node-local disk with sufficient space | Local scratch such as `/tmp`; the checkout and final image can stay on AFS |
| Finished image | Shared storage accessible to workers | AFS or another shared path accessible to workers |
| Workflow submission settings | Your workflow's IC submission YAMLs | CERN submission YAMLs, selected directly or through `--submit-config-map` |

For CERN workflow setup and submission examples, see
[CERN Condor submission](cern_submission.md).

## Build

### Imperial College

Run from the repository, with the working environment selected explicitly:

```bash
source env.sh
container_build_dir=$(mktemp -d /tmp/innfer-apptainer.XXXXXX)

python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir "$container_build_dir" \
  --output "$PWD/images/innfer-20261120-ic.sif"
```

The builder requires 30 GiB free by default, and large environments can need
more. Check the chosen filesystem with `df -h "$container_build_dir"`. If
`/tmp` is too small, select a larger node-local scratch directory with
`--work-dir`; a batch job's `_CONDOR_SCRATCH_DIR` is another option when it has
enough space. `/dev/shm` consumes memory and is suitable only when enough is
available. `--min-free-gb` controls the initial space check.

The build/cache/unpacked files all use `--work-dir`. The builder sets
`APPTAINER_TMPDIR`, `APPTAINER_CACHEDIR` and `TMPDIR` beneath that directory,
so setting `APPTAINER_TMPDIR` separately does not override this choice.

### CERN lxplus / HTCondor

Build directly on CERN using local disk for the workspace and Quay for the
base image. The checkout and final SIF may remain on AFS:

```bash
source env.sh
container_build_dir=$(mktemp -d /tmp/innfer-apptainer.XXXXXX)

python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir "$container_build_dir" \
  --base-image quay.io/rockylinux/rockylinux:9 \
  --output "$PWD/images/innfer-20261120-cern-local.sif"
```

Building from an AFS checkout and saving the finished image on AFS worked with
the command above. Its `--work-dir` points to `/tmp`, so Apptainer extracts the
base image and creates temporary build files on local scratch. The earlier
attempt used the AFS checkout's `tmp/apptainer-build` for those temporary files
and failed during OCI layer extraction with `invalid cross-device link` and
extended-attribute warnings. This distinction concerns the temporary workspace;
it does not require moving the checkout, packaged environment or finished SIF
off AFS. Building with an EOS temporary workspace has not been tested here.

If building inside an HTCondor job, keep the build process on that worker and
write the finished SIF to shared storage before the job exits; local scratch
does not persist across workers. Keep a terminal build running in tmux.

### Packaging and rebuilds

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

If the environment was packaged successfully but a later build/test failed,
retry with its saved archive to avoid packaging it again. Use the context
directory printed by the failed build; the example below reads an earlier
archive on shared storage while creating a new workspace on local disk:

```bash
container_build_dir=$(mktemp -d /tmp/innfer-apptainer.XXXXXX)

python3 scripts/build_apptainer.py \
  --env-prefix "$PWD/miniconda/files/envs/innfer_env" \
  --work-dir "$container_build_dir" \
  --reuse-environment-from "$PWD/tmp/apptainer-build/<previous-build-directory>" \
  --base-image quay.io/rockylinux/rockylinux:9 \
  --output "$PWD/images/innfer-20261120-cern-local.sif"
```

This requires the saved environment inventory to match the selected environment. It freezes the current code/recipe afresh and records the reused archive's location; it does not reuse the old source snapshot. If you changed package contents without changing package versions, package a fresh environment instead.

Keep the old context and its `environment.tar.gz` until the retry succeeds.
Do not delete a local context while its build is running. Temporary directories
can be removed by site cleanup policies, so preserve the packaged environment
and manifests on shared storage if they will be needed for future rebuilds.

### Runtime dependency checks and environment repairs

The builder checks CPU ONNX inference and numerical differentiation using the
selected environment before packing it. These checks also run inside the image
as part of `%test` and the isolated smoke test. They catch missing
`onnxruntime`/`numdifftools` packages, incompatible dependency versions and ONNX shared-library failures without
external datasets or model downloads. The image smoke test also checks that
`python.worker` imports resolve inside the frozen source tree.

Both setup environment files declare `onnxruntime==1.28.0` and
`numdifftools==0.9.42`. The ONNX Runtime pin matches the working IC image;
`protobuf==4.25.8` satisfies its dependency while retaining the protobuf 4 series
used with TensorFlow 2.15. Existing environments need an explicit repair before
rebuilding, since editing the YAML does not install packages into them:

```bash
python3 -m pip install \
  onnxruntime==1.28.0 protobuf==4.25.8 numdifftools==0.9.42
python3 containers/smoke_test.py --dependencies-only
```

Run this with the environment you intend to package active. Build a new,
versioned SIF after the checks pass and select its path for the regenerated
workflow. Existing SIF files do not change when their source environment is
repaired. Package the updated environment afresh; do not reuse an archive made
before the dependency repair.

The combined controller/training environment retains the workaround in
`setup.sh`: TensorFlow 2.15.1 uses `wrapt==1.14.1`; Snakemake 9.9.0 and its
storage-interface package 4.2.3 are installed with `--no-deps`. The dependency
validator accepts that exact combination, reports the exception, and records
the pre-build check output in the image manifest. Other dependency mismatches,
including incompatible protobuf versions or a different wrapt combination,
still fail validation. No metadata in installed third-party packages is edited.

The image smoke tests run both flow implementations and a small local Snakemake
workflow to check the combined environment works in practice. `pip check` will
still report the storage-interface package's declared `wrapt>=1.15` requirement;
this is the explicitly accepted setup workaround. Changing these pinned
versions requires revisiting the exception and rerunning the runtime tests.

### Build failures seen at CERN

| Message | Meaning and next action |
| --- | --- |
| `docker.io/... StatusCode: 429` | Docker Hub throttled the download. Retry with `--base-image quay.io/rockylinux/rockylinux:9`; reuse the completed environment archive. |
| `unpack entry ... invalid cross-device link` under `/afs/...` | Move `--work-dir` to local disk. Reuse the environment archive; the final `--output` can stay on shared storage. |
| `User not listed in /etc/subuid, trying root-mapped namespace` | Apptainer is selecting its fallback fakeroot mode. This was informational in the reported failures; inspect the subsequent `FATAL` message for the actual cause. |
| `Choose a new output path` | The requested SIF or its `.partial` file already exists. Choose a new versioned output name and use that same name when submitting. |

See the upstream [build filesystem guidance](https://apptainer.org/docs/admin/main/installation.html),
[temporary build directories](https://apptainer.org/docs/user/main/build_env.html)
and [Docker Hub throttling documentation](https://docs.docker.com/docker-hub/troubleshoot/).

`bash setup.sh container` is also wired to the builder. Its work directory can be selected with `INNFER_CONTAINER_BUILD_DIR`; the explicit Python command above is preferable when choosing the source environment and output filename. The disabled cleanup at the end of setup.sh stays disabled.

## Validation and image contents

The build installs the OS libraries, fonts and LaTeX packages needed by the application. It runs a CPU smoke test that checks ONNX inference, numerical differentiation, frozen source imports from the image, both BayesFlow and KerasFlow checkpoint round trips/density evaluation/sampling, parquet I/O, a local Snakemake workflow, PDF table compilation and `innfer --help`. A second test disables host home/current-directory/site bind mounts, so missing dependencies cannot be satisfied accidentally by the original checkout.

The recipe creates the system `pdflatex.fmt` during the writable build phase. Generating it on first use would fail when the finished image is read-only, especially under fakeroot where TeX tries to write beneath `/root`.

After both tests pass the `.partial` image is published to the requested `.sif` path, alongside `.sif.sha256` and `.sif.json` provenance files. The image itself contains `/opt/innfer/app/container-manifest.json` and an OS-package list. Failed builds retain their build context and any partial image for diagnosis; they do not replace an existing image.

Failed builds also retain Apptainer's bundle under the build context's `temporary` directory. Failed LaTeX/CLI smoke checks print the tool's captured output before raising an error, so the build log includes the underlying failure.

Run the CLI and inspect the frozen version. Select the image you just built
(at CERN use `images/innfer-20261120-cern-local.sif`):

```bash
container_image="$PWD/images/innfer-20261120-ic.sif"
apptainer run --cleanenv "$container_image" --help
apptainer exec --cleanenv "$container_image" \
  cat /opt/innfer/app/container-manifest.json
```

On an allocated GPU node, validate NVIDIA access separately:

```bash
apptainer exec --nv --cleanenv "$container_image" \
  python /opt/innfer/app/containers/smoke_test.py --require-gpu
```

The host supplies the NVIDIA driver; TensorFlow/CUDA userspace dependencies and CUDA libdevice are packaged in the environment. A successful CPU build does not claim GPU compatibility with every execution node.

The run entry point uses `/opt/innfer/app/scripts/innfer.py`. When running actual analyses, select frozen configs via `/opt/innfer/app/configs/...`; a relative `configs/...` argument resolves against the runtime working directory and can select a mounted host config instead. Data/model/output paths remain external and are controlled by the existing directory variables. This image alone does not create a local data sandbox, publish results or change the Snakemake shared-filesystem assumptions.

## Submit jobs with an image

### Imperial College

Set up the bundled IC profile once, then start the controller in tmux.
Replace `your_analysis.py` and `your_workflow.yaml` with your run config and
workflow. These are example paths, not files supplied by the container setup.

```bash
bash setup.sh snakemake_ic
source env.sh

python3 scripts/innfer.py \
  --cfg=configs/run/your_analysis.py \
  --step=SnakeMake \
  --snakemake-cfg=configs/snakemake/your_workflow.yaml \
  --snakemake-profile=profiles/ic \
  --container-image="$PWD/images/innfer-20261120-ic.sif" \
  --snakemake-rerun-incomplete
```

The IC workflow keeps its existing IC submission YAMLs and resource requests.
For CERN, use `profiles/cern` and CERN submission settings for each step, either
directly in the workflow or through a submission map. The profile selects the
scheduler adapter; it does not convert the workflow's resource requests by
itself. See [CERN Condor submission](cern_submission.md) for a generic map and
submission command.

### Direct jobs and launcher behaviour

Add `--container-image` to the normal host-side submission command. It is disabled by default and works for direct Condor submission and generated Snakemake jobs. The host removes container submission options from worker arguments, so the existing image can be used without rebuilding it for the launcher.

For example, prepare density training jobs without submitting them. Select a
different step and its usual options as needed for your config:

```bash
source env.sh
JOBS_DIR=jobs/container_check python3 scripts/innfer.py \
  --cfg=configs/run/your_analysis.py \
  --step=TrainDensity \
  --density-architecture=configs/architecture/density_default.yaml \
  --container-image=images/innfer-20261120-ic.sif \
  --submit=configs/submit/condor_gpu.yaml \
  --dry-run
```

Inspect the `.sh`, `.sub` and `.container.json` in
`jobs/container_check/<config-name>/TrainDensity/00000/`, where
`<config-name>` is the run config's `name`. All submitted jobs, including jobs
without containers, have their own step directory and groups of 500 numbered
jobs. Group `00000` contains indices 0--499, `00001` contains 500--999, and so
on. Jobs using descriptive names instead of indices use stable `named_*`
subdirectories. The generated Snakefile stays in
`jobs/container_check/<config-name>/`; its rule names are unchanged.
Remove `--dry-run` to submit. To test locally, execute the generated
`.sh` with `bash` on a machine with the requested GPU and sufficient local
scratch. The direct submission YAML in this example is for IC; on CERN use a
submission YAML with the appropriate CERN queue and resources.

For a workflow, pass `--container-image` alongside `--step=SnakeMake` and `--snakemake-cfg`, or set it in `run_options`:

```yaml
- workflow: configs/snakemake/your_workflow.yaml
  run_options:
    container_image: images/innfer-20261120-ic.sif
    container_image_mode: direct  # Default; use copy for worker scratch staging.
```

A child can use `container_image: null` to retain ordinary host execution. Regenerate the Snakefile and scripts for changes to take effect; existing running workflows are not modified.

With `--container-image-mode=copy`, the launcher copies the image to a checksum-verified cache in worker scratch, using a lock and atomic publication for concurrent jobs. Its default location is under `_CONDOR_SCRATCH_DIR`, then `TMPDIR`, then `/tmp`. `INNFER_CONTAINER_CACHE` in the worker environment can select a persistent node-local cache. Local space must accommodate the image and runtime files.

By default (`--container-image-mode=direct`), the launcher runs the SIF directly
from the path supplied by `--container-image`. Keep it in the checkout
(e.g. `--container-image=images/innfer-version.sif`) or select another shared
location, including AFS or EOS. There is no worker image copy. For example:

```bash
python3 scripts/innfer.py \
  --cfg=configs/run/your_analysis.py \
  --step=SnakeMake \
  --snakemake-cfg=configs/snakemake/your_workflow.yaml \
  --snakemake-profile=profiles/cern \
  --container-image=/eos/user/u/username/images/innfer-version.sif
```

The mode is also accepted as `container_image_mode: direct` in Snakemake
`run_options`. It applies to ordinary batch submissions and Snakemake at both
sites. Direct mode verifies the image checksum on the submitting machine;
verification is reused across steps while the image and provenance file
metadata remain unchanged. Workers check that the SIF is readable and use the
shared path without a full image checksum read. Keep each submitted SIF at a
versioned path and do not replace or edit it while jobs are queued or running.
Workers still copy and verify the small config archive in scratch.

EOS must be accessible on the host worker to open an EOS image, and datasets
inside the container still require the appropriate mounts and credentials.
See [CERN authentication](cern_submission.md#eos-authentication-inside-the-container).
Direct mode avoids full image copies; its runtime performance depends on the
shared filesystem. Updating the host generator and regenerating jobs is enough
to enable this mode; the existing image does not need rebuilding.

The selected run/architecture files, supporting architecture configs and `configs/other` are archived at job generation. Each worker checks and extracts a private copy. Thus jobs use submission-time configs with the image's code. Private config copies permit runtime architecture overrides without editing the archived snapshot. Rebuild the image when changing Python worker code that jobs need.

GPU requests in the submission YAML automatically enable `--nv`; the worker's GPU visibility allocation is preserved. CPU jobs disable CUDA visibility. Existing GPU counts, memory, runtime and retry settings are retained. If the submit file has no disk request, the generated submission requests space for the config archive plus 1 GiB of runtime scratch, and adds the image size in copy mode; an explicit `RequestDisk` takes precedence. The launcher invokes image code without activating host conda. Its normal home mount remains available. For Kerberos access it forwards the worker's current `KRB5CCNAME` and `KRB5_CONFIG`, binding readable file-based caches and selected configuration files read-only at runtime. Credentials are not packaged into the image or config archive. See [CERN authentication](cern_submission.md#eos-authentication-inside-the-container) for EOS checks and local testing.

Each job gets a `.container.json` recording the image/config hashes, commands, directory settings, image execution mode and GPU mode. Snakemake declares the image and config archive as inputs. Worker failures stop grouped commands and propagate to the scheduler.

If workflow generation on AFS raises `OSError: [Errno 27] File too large` while
opening a small new job file, the containing directory may have reached its
entry limit. This limit depends on filename lengths, as described in the
[OpenAFS directory limits](https://docs.openafs.org/Reference/8/dafileserver.html).
Job files use the per-step and group subdirectories above to avoid concentrating
every script, manifest and log in one directory. Updating this host-side generator
and regenerating the workflow does not require rebuilding the image. An old
flat jobs directory that is already full still needs space for the new
subdirectories; use a fresh shared `JOBS_DIR` or archive inactive generated
files first. Keep files referenced by running jobs in place.

**Data, models and outputs remain on shared storage**, using the existing directory variables. This launcher does not implement dataset staging or copying completed results from a full scratch workspace.

Implementation: [builder](../scripts/build_apptainer.py), [recipe](../containers/innfer.def), [runtime environment](../containers/environment.sh), [smoke test](../containers/smoke_test.py), [submission launcher](../python/worker/container_submission.py).

References: [Apptainer definition files](https://apptainer.org/docs/user/1.5/definition_files.html), [GPU support](https://apptainer.org/docs/user/main/gpu.html), [conda-pack relocation](https://conda.github.io/conda-pack/cli.html).
