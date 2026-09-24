# WorldGen DeployBench runner

`worldgen-panorama` is a generator runner for one full 360 by 180 degree equirectangular RGB `image`. It runs WorldGen's DA-2 spherical depth path and returns a Graphdeco-compatible `3dgs` PLY. It does not run FLUX panorama generation, Sharp, background inpainting, or mesh export.

The generated scene uses `RDF` coordinates, where positive X points right, positive Y points down, and positive Z points forward. DA-2 normalizes each prediction to a maximum radial distance of 20, so the runner reports relative scene units and an initial `scene_scale` of 1.0. Use the DeployBench `3dgs_scale_calibration` evaluator when metric camera displacement matters.

The distributable catalog is [config/runners/worldgen.yaml](config/runners/worldgen.yaml). Its batch size is 10 because the server executes jobs sequentially and each job gets a fresh child process. This amortizes container startup while periodically recycling CUDA and Python state. Two attempts allow one retry for transient downloads, container failures, or GPU failures.

WorldGen and DA-2 run each inference job on one `torch.device`. DA-2's supplied Accelerate inference configs disable distributed execution and use one process. The runner therefore requests one GPU rather than exposing every host GPU. To pin a deployment to a specific GPU, set `launcher.gpus` to `device=<GPU UUID>` in the deployed catalog. For a local run, set `RUNNER_GPUS=device=<GPU UUID>`.

## Model assets

The runner downloads the public `haodongli/DA-2` checkpoint through Hugging Face on first use. It pins snapshot `0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3` and reports that revision in each job's model metrics. The catalog stores Hugging Face, Torch, and XDG caches below `/data/model_cache/worldgen`. `HF_TOKEN` is optional and passes through from the deployment environment.

Initialize the repository-pinned DA-2 source before building. The Dockerfile fetches PyTorch3D at its pinned commit because the upstream fork listed it in `.gitmodules` without committing a corresponding gitlink.

```bash
git submodule update --init submodules/DA-2
```

The Docker build fails with a direct message when the DA-2 submodule is missing.

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

The first smoke run downloads DA-2. The result contains `3DGS-rgbd-<hash>.ply`, a runner log, and a metrics JSON file under `output/worldgen-panorama@0.1.0/smoke/sample-1`.

## Contract

The request must contain exactly one primary sample in `inputs.data`, with an `image` path. Missing projection metadata defaults to `equirectangular`. An explicit projection must be `equirectangular`, an explicit field of view must be `[360, 180]`, and the decoded image must have a 2:1 aspect ratio.

The adapter converts raw alpha to Graphdeco opacity logits, canonicalizes signed pole scales and clamps them to a small positive value, normalizes quaternions, and rejects non-finite output before publication. This makes WorldGen's PLY compatible with the DeployBench 3DGS renderer.

The HTTP and result contract is defined in [docs/api.md](docs/api.md).
