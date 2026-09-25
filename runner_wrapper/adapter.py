from __future__ import annotations

"""WorldGen equirectangular panorama-to-3DGS adapter."""

import hashlib
import json
import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.adapter")

RUNNER_NAME = "worldgen-panorama"
DA2_MODEL_ID = "haodongli/DA-2"
DA2_REVISION = "0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3"
OUTPUT_METADATA = {
    "scene_scale": 1.0,
    "scene_coordinate_system": "RDF",
    "scene_units": "relative",
    "scene_origin": "primary_viewpoint",
}


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def utc_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _variant_key(parameters: dict[str, Any], splat_mode: str = "rgbd") -> str:
    if splat_mode not in {"rgbd", "sharp"}:
        raise ValueError(f"unknown splat mode: {splat_mode}")
    digest = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:10]
    return f"{splat_mode}-{digest}"


def _parameters(raw: object) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("job.parameters must be an object")
    if raw:
        raise ValueError(f"unknown job parameters: {', '.join(sorted(map(str, raw)))}")
    return {}


def _normalize_inputs(raw_inputs: object) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(raw_inputs, dict):
        raise ValueError("inputs must be an object")
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_role, raw_samples in raw_inputs.items():
        role = str(raw_role).strip()
        if not role or not isinstance(raw_samples, dict):
            raise ValueError("each input role must contain a sample mapping")
        samples: dict[str, dict[str, Any]] = {}
        for raw_sample, raw_data in raw_samples.items():
            sample = str(raw_sample).strip()
            if not sample or not isinstance(raw_data, dict):
                raise ValueError(f"inputs.{role} must map sample ids to data mappings")
            samples[sample] = {
                str(data_type).strip(): value.strip() if isinstance(value, str) else value
                for data_type, value in raw_data.items()
                if str(data_type).strip()
            }
        normalized[role] = samples
    return normalized


def _full_panorama_metadata(job: dict[str, Any]) -> None:
    metadata = job.get("primary_sample_metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("job.primary_sample_metadata must be an object")
    projection = str(metadata.get("projection") or "equirectangular").strip().lower()
    if projection != "equirectangular":
        raise ValueError(
            "primary_sample_metadata.projection must be equirectangular; "
            f"received {projection!r}"
        )
    fov = metadata.get("fov")
    if fov is not None:
        if (
            not isinstance(fov, (list, tuple))
            or len(fov) != 2
            or any(type(value) not in (int, float) for value in fov)
            or not math.isclose(float(fov[0]), 360.0, abs_tol=1e-3)
            or not math.isclose(float(fov[1]), 180.0, abs_tol=1e-3)
        ):
            raise ValueError("WorldGen requires a full 360 by 180 degree panorama")


def _prepare_input(job_request: dict[str, Any], destination: Path) -> tuple[str, Path, tuple[int, int]]:
    from PIL import Image, UnidentifiedImageError

    job = job_request.get("job")
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    if job.get("job_type") not in ("generation", "generator"):
        raise ValueError("WorldGen accepts generation jobs only")
    primary = str(job.get("primary_sample") or "").strip()
    if not primary:
        raise ValueError("job.primary_sample is required")

    inputs = _normalize_inputs(job_request.get("inputs"))
    samples = inputs.get("data", {})
    if set(samples) != {primary}:
        raise ValueError("WorldGen requires exactly one primary sample in inputs.data")
    if inputs.get("candidate") or inputs.get("references"):
        raise ValueError("WorldGen does not consume candidate or reference inputs")
    image_value = samples[primary].get("image")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"inputs.data.{primary}.image must be a file path")
    source = Path(image_value)
    if not source.is_file():
        raise FileNotFoundError(f"input image not found: {source}")

    _full_panorama_metadata(job)
    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            if width != 2 * height:
                raise ValueError(
                    "WorldGen requires a 2:1 equirectangular image; "
                    f"received {width}x{height}"
                )
            if width < 16 or height < 8:
                raise ValueError("input panorama is too small")
            image.convert("RGB").save(destination, format="PNG")
    except UnidentifiedImageError as exc:
        raise ValueError(f"input image is not readable: {source}") from exc
    return primary, source, (width, height)


def _configure_model_cache() -> Path:
    cache_root = Path(os.getenv("PATH_MODEL_CACHE", "/data/model_cache")) / "worldgen"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    return cache_root


def _generate_splat(image_path: Path, splat_mode: str = "rgbd") -> Any:
    import torch
    from da2.model.spherevit import SphereViT
    from PIL import Image
    from worldgen.pano_depth import DA2_CONFIG, pred_pano_depth

    if not torch.cuda.is_available():
        raise RuntimeError("WorldGen requires an NVIDIA CUDA GPU")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (7, 5):
        raise RuntimeError("WorldGen requires CUDA compute capability 7.5 or newer")
    device = torch.device("cuda")
    depth_model = SphereViT.from_pretrained(
        DA2_MODEL_ID,
        config=DA2_CONFIG,
        revision=DA2_REVISION,
    )
    depth_model.eval()
    depth_model = depth_model.to(device)
    with Image.open(image_path) as image:
        panorama = image.convert("RGB")
        predictions = pred_pano_depth(depth_model, panorama)
        del depth_model
        if splat_mode == "sharp":
            from runner_wrapper.sharp_adapter import generate_sharp_splat

            return generate_sharp_splat(panorama, predictions, device)
    if splat_mode == "rgbd":
        from worldgen.utils.splat_utils import convert_rgbd_to_gs

        return convert_rgbd_to_gs(
            predictions["rgb"],
            predictions["distance"],
            predictions["rays"],
        )
    raise ValueError(f"unknown splat mode: {splat_mode}")


def _write_graphdeco_ply(splat: Any, destination: Path) -> int:
    """Write a finite Graphdeco PLY with opacity logits, not raw alpha."""

    import numpy as np
    from plyfile import PlyData, PlyElement

    centers = np.asarray(splat.centers, dtype=np.float32)
    colors = np.asarray(splat.rgbs, dtype=np.float32)
    alpha = np.asarray(splat.opacities, dtype=np.float32).reshape(-1)
    scales = np.asarray(splat.scales, dtype=np.float32)
    rotations = np.asarray(splat.rotations, dtype=np.float32)
    count = len(centers)
    expected_shapes = {
        "centers": (count, 3),
        "colors": (count, 3),
        "scales": (count, 3),
        "rotations": (count, 4),
    }
    arrays = {
        "centers": centers,
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
    }
    if count == 0:
        raise ValueError("WorldGen produced no Gaussians")
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(f"invalid {name} shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"WorldGen produced non-finite {name}")
    if alpha.shape != (count,) or not np.isfinite(alpha).all():
        raise ValueError("WorldGen produced invalid opacities")
    colors = np.clip(colors, 0.0, 1.0)
    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    opacity_logits = np.log(alpha / (1.0 - alpha))
    # The equirectangular south pole can produce a tiny negative sigma from
    # floating-point sin(pi). Scale signs do not affect the covariance, so
    # canonicalize them before converting to Graphdeco's log-scale encoding.
    log_scales = np.log(np.maximum(np.abs(scales), 1e-7))
    rotation_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if (rotation_norms < 1e-12).any():
        raise ValueError("WorldGen produced a zero-length Gaussian rotation")
    rotations = rotations / rotation_norms
    sh_dc = (colors - 0.5) / 0.28209479177387814

    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    vertices = np.empty(count, dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = centers[:, index]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0.0
    for index, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        vertices[name] = sh_dc[:, index]
    vertices["opacity"] = opacity_logits
    for index, name in enumerate(("scale_0", "scale_1", "scale_2")):
        vertices[name] = log_scales[:, index]
    for index, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        vertices[name] = rotations[:, index]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(destination)
    return count


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run_job(job_request: dict[str, Any], splat_mode: str) -> dict[str, Any]:
    if splat_mode not in {"rgbd", "sharp"}:
        raise ValueError(f"unknown splat mode: {splat_mode}")
    started_at = time.time()
    runtime = job_request.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("workspace_dir"):
        raise ValueError("runtime.workspace_dir is required")
    workspace = Path(runtime["workspace_dir"])
    workspace.mkdir(parents=True, exist_ok=True)
    raw_parameters = (
        job_request.get("job", {}).get("parameters")
        if isinstance(job_request.get("job"), dict)
        else None
    )
    try:
        variant = _variant_key(_parameters(raw_parameters), splat_mode)
    except (TypeError, ValueError):
        fallback = hashlib.sha256(str(job_request.get("job", {})).encode()).hexdigest()[:10]
        variant = f"invalid-{fallback}"
    log_path = workspace / f"runner-{variant}.log"
    report_path = workspace / f"metrics-{variant}.json"
    stage = "validation"
    monitor: ResourceMonitor | None = None
    metrics: list[dict[str, Any]] = []
    report: dict[str, Any] = {"inputs": job_request.get("inputs", {})}

    with tee_job_output(log_path):
        try:
            job = job_request["job"]
            parameters = _parameters(job.get("parameters"))
            report["parameters"] = parameters
            prepared_image = workspace / "input-panorama.png"
            primary, source_image, resolution = _prepare_input(job_request, prepared_image)
            monitor = ResourceMonitor(
                sample_data={"data.image": str(source_image)},
                output_dir=workspace,
            )
            monitor.start()
            cache_root = _configure_model_cache()
            logger.info(
                event_message(
                    "worldgen_started",
                    job_id=job.get("job_id"),
                    primary_sample=primary,
                    resolution=list(resolution),
                    splat_mode=splat_mode,
                    model_cache=str(cache_root),
                    model_revision=DA2_REVISION,
                )
            )

            stage = "model_inference"
            splat = _generate_splat(prepared_image, splat_mode)
            stage = "export"
            output_name = f"3DGS-{variant}.ply"
            gaussian_count = _write_graphdeco_ply(splat, workspace / output_name)
            output_files = {primary: {"3dgs": output_name}}
            output_metadata = dict(OUTPUT_METADATA)
            model_metrics = [
                {
                    "namespace": "model",
                    "name": "gaussian_count",
                    "type": "integer",
                    "value": gaussian_count,
                    "unit": "gaussians",
                    "source": "model",
                },
                {
                    "namespace": "model",
                    "name": "checkpoint_revision",
                    "type": "string",
                    "value": DA2_REVISION,
                    "source": "model",
                },
                {
                    "namespace": "model",
                    "name": "splat_mode",
                    "type": "string",
                    "value": splat_mode,
                    "source": "model",
                },
            ]
            if splat_mode == "sharp":
                from runner_wrapper.sharp_adapter import (
                    SHARP_CHECKPOINT_ETAG,
                    SHARP_SOURCE_REVISION,
                )

                model_metrics.extend(
                    [
                        {
                            "namespace": "model",
                            "name": "sharp_checkpoint_etag",
                            "type": "string",
                            "value": SHARP_CHECKPOINT_ETAG,
                            "source": "model",
                        },
                        {
                            "namespace": "model",
                            "name": "sharp_source_revision",
                            "type": "string",
                            "value": SHARP_SOURCE_REVISION,
                            "source": "model",
                        },
                    ]
                )
            report.update(
                output_files=output_files,
                output_metadata=output_metadata,
                input_resolution=list(resolution),
                model_metrics=model_metrics,
            )
            result: dict[str, Any] = {
                "status": "completed",
                "output_files": output_files,
                "output_metadata": output_metadata,
                "failure": None,
            }
            logger.info(
                event_message(
                    "worldgen_completed",
                    job_id=job.get("job_id"),
                    gaussian_count=gaussian_count,
                    output=output_name,
                )
            )
        except Exception as exc:
            traceback.print_exc()
            retryable = stage == "model_inference" and isinstance(
                exc, (ConnectionError, OSError, TimeoutError)
            )
            failure = {
                "code": "WORLDGEN_FAILED",
                "message": f"{stage}: {exc}",
                "retryable": retryable,
                "stage": stage,
            }
            report["failure"] = failure
            result = {"status": "failed", "failure": failure}
        finally:
            if monitor is not None:
                metrics = monitor.stop()

        model_metrics = report.get("model_metrics", [])
        all_metrics = metrics + model_metrics
        report["resource_metrics"] = metrics
        _write_report(report_path, report)

    result.update(
        started_at=utc_time(started_at),
        completed_at=utc_time(time.time()),
        metrics=all_metrics,
        artifacts=[
            {"artifact_type": "job_log", "path": log_path.name},
            {"artifact_type": "metric_summary", "path": report_path.name},
        ],
    )
    return result


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    return _run_job(job_request, splat_mode="rgbd")
