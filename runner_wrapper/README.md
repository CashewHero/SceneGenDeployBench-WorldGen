# WorldGen DeployBench runner

This wrapper provides two generator runners for one full 360 by 180 degree equirectangular RGB `image`:

- `worldgen-panorama@0.1.1` converts WorldGen's DA-2 spherical RGB-D prediction directly to 3D Gaussians.
- `worldgen-panorama-sharp@0.1.1` runs WorldGen's experimental Sharp path over six cubemap faces, aligns each face to DA-2 depth, and merges the results.

Both return a Graphdeco-compatible `3dgs` PLY. Neither runs FLUX panorama generation, background inpainting, or mesh export.

The generated scene uses `RDF` coordinates, where positive X points right, positive Y points down, and positive Z points forward. DA-2 normalizes each prediction to a maximum radial distance of 20, so the runner reports relative scene units and an initial `scene_scale` of 1.0. Use the DeployBench `3dgs_scale_calibration` evaluator when metric camera displacement matters.

The distributable catalog is [config/runners/worldgen.yaml](config/runners/worldgen.yaml). The direct runner uses a batch size of 10. Sharp uses 4 because each result contains about 7.1 million Gaussians and its PLY is roughly 460 MiB. The server executes jobs sequentially in fresh child processes, so these values amortize startup while periodically recycling CUDA and Python state. Both use two attempts to allow one retry for transient downloads, container failures, or GPU failures.

WorldGen, DA-2, and Sharp run each inference job on one `torch.device`; Sharp processes its six faces sequentially. The upstream inference code has no multi-GPU execution path, so both runners request one GPU. To pin a deployment to a specific GPU, set `launcher.gpus` to `device=<GPU UUID>` in the deployed catalog. For a local run, set `RUNNER_GPUS=device=<GPU UUID>`.

## Model assets

Both runners download the public `haodongli/DA-2` checkpoint through Hugging Face on first use. They pin snapshot `0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3` and report that revision in each job's model metrics. The catalog stores Hugging Face, Torch, and XDG caches below `/data/model_cache/worldgen`. `HF_TOKEN` is optional and passes through from the deployment environment.

Sharp additionally downloads Apple's 2.81 GB `sharp_2572gikvuh.pt` checkpoint on first use. The adapter validates its byte size and multipart ETag under a file lock before atomically publishing it to the shared cache. It pins ml-sharp source revision `1eaa046834b81852261262b41b0919f5c1efdd2e` and reports both identities in the job metrics. The Sharp model weights are subject to Apple's [research-only, non-commercial model license](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL); confirm that your use complies before deploying this runner.

Initialize the repository-pinned DA-2 and ml-sharp sources before building. The Dockerfile fetches PyTorch3D at its pinned commit because the upstream fork listed it in `.gitmodules` without committing a corresponding gitlink.

```bash
git submodule update --init submodules/DA-2 submodules/ml-sharp
```

The Docker build fails with a direct message when either submodule is missing.

## Build and test

Run contract and adapter tests without a GPU:

```bash
runner_wrapper/localtest.sh test
```

Build the CUDA image from the repository root:

```bash
runner_wrapper/localtest.sh build
```

The smoke request expects a real 2:1 panorama at `datasets/smoke/image.png` below the selected data root. The standard local DeployBench data root already has this input:

```bash
RUNNER_DATA_DIR=/mnt/sata1/deploybench runner_wrapper/localtest.sh smoke
```

The first direct smoke run downloads DA-2. The result contains `3DGS-rgbd-<hash>.ply`, a runner log, and a metrics JSON file under `output/worldgen-panorama@0.1.1/smoke/sample-1`.

Run the Sharp smoke with its runner identity, adapter, and request:

```bash
RUNNER_DATA_DIR=/mnt/sata1/deploybench \
RUNNER_NAME=worldgen-panorama-sharp \
RUNNER_VERSION=0.1.1 \
RUNNER_ADAPTER=runner_wrapper.sharp_adapter:run_job \
RUNNER_REQUEST_FILE=runner_wrapper/examples/generator_sharp_job_request.json \
runner_wrapper/localtest.sh smoke
```

Its first run also downloads the Sharp checkpoint. The result is written below `output/worldgen-panorama-sharp@0.1.1/smoke/sample-1` and is named `3DGS-sharp-<hash>.ply`.

## Contract

The request must contain exactly one primary sample in `inputs.data`, with an `image` path. Missing projection metadata defaults to `equirectangular`. An explicit projection must be `equirectangular`, an explicit field of view must be `[360, 180]`, and the decoded image must have a 2:1 aspect ratio.

The adapter converts raw alpha to Graphdeco opacity logits, canonicalizes signed pole scales and clamps them to a small positive value, normalizes quaternions, and rejects non-finite output before publication. This makes WorldGen's PLY compatible with the DeployBench 3DGS renderer.

The HTTP and result contract is defined in [docs/api.md](docs/api.md).
