from __future__ import annotations

"""WorldGen Sharp panorama-to-3DGS adapter."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from runner_wrapper.adapter import _run_job

SHARP_SOURCE_REVISION = "1eaa046834b81852261262b41b0919f5c1efdd2e"
SHARP_CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
SHARP_CHECKPOINT_NAME = "sharp_2572gikvuh.pt"
SHARP_CHECKPOINT_BYTES = 2_809_738_232
SHARP_CHECKPOINT_ETAG = "0d0d57485fdd957aa240fa996ed7fdaa-42"
SHARP_CHECKPOINT_PART_SIZE = 64 * 1024 * 1024


def _multipart_etag(path: Path, part_size: int = SHARP_CHECKPOINT_PART_SIZE) -> str:
    part_digests: list[bytes] = []
    with path.open("rb") as stream:
        while chunk := stream.read(part_size):
            part_digests.append(hashlib.md5(chunk, usedforsecurity=False).digest())
    if not part_digests:
        raise ValueError(f"Sharp checkpoint is empty: {path}")
    digest = hashlib.md5(b"".join(part_digests), usedforsecurity=False).hexdigest()
    return f"{digest}-{len(part_digests)}"


def _verify_checkpoint(
    path: Path,
    expected_bytes: int = SHARP_CHECKPOINT_BYTES,
    expected_etag: str = SHARP_CHECKPOINT_ETAG,
    part_size: int = SHARP_CHECKPOINT_PART_SIZE,
) -> None:
    size = path.stat().st_size
    if size != expected_bytes:
        raise RuntimeError(
            f"Sharp checkpoint has {size} bytes; expected {expected_bytes}: {path}"
        )
    actual_etag = _multipart_etag(path, part_size)
    if actual_etag != expected_etag:
        raise RuntimeError(
            f"Sharp checkpoint ETag is {actual_etag}; expected {expected_etag}: {path}"
        )


def _checkpoint_path() -> Path:
    import torch
    from filelock import FileLock

    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / SHARP_CHECKPOINT_NAME
    identity = checkpoint.with_suffix(checkpoint.suffix + ".identity.json")
    expected_identity = {
        "bytes": SHARP_CHECKPOINT_BYTES,
        "etag": SHARP_CHECKPOINT_ETAG,
        "part_size": SHARP_CHECKPOINT_PART_SIZE,
        "url": SHARP_CHECKPOINT_URL,
    }

    with FileLock(str(checkpoint) + ".lock"):
        if checkpoint.exists():
            if checkpoint.stat().st_size != SHARP_CHECKPOINT_BYTES:
                raise RuntimeError(f"cached Sharp checkpoint has the wrong size: {checkpoint}")
            try:
                recorded_identity = json.loads(identity.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                recorded_identity = None
            if recorded_identity != expected_identity:
                _verify_checkpoint(checkpoint)
        else:
            partial = checkpoint.with_name(f".{checkpoint.name}.partial-{os.getpid()}")
            try:
                torch.hub.download_url_to_file(
                    SHARP_CHECKPOINT_URL,
                    str(partial),
                    progress=True,
                )
                _verify_checkpoint(partial)
                os.replace(partial, checkpoint)
            finally:
                partial.unlink(missing_ok=True)

        temporary_identity = identity.with_name(f".{identity.name}.partial-{os.getpid()}")
        temporary_identity.write_text(
            json.dumps(expected_identity, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_identity, identity)
    return checkpoint


def _build_sharp_model(device: Any) -> Any:
    import torch
    from sharp.models import PredictorParams, create_predictor

    state_dict = torch.load(_checkpoint_path(), map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    return predictor.to(device)


def generate_sharp_splat(panorama: Any, depth_predictions: dict[str, Any], device: Any) -> Any:
    from worldgen.pano_sharp import predict_equirectangular

    predictor = _build_sharp_model(device)
    return predict_equirectangular(
        predictor,
        panorama,
        device,
        depth_predictions=depth_predictions,
    )


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    return _run_job(job_request, splat_mode="sharp")
