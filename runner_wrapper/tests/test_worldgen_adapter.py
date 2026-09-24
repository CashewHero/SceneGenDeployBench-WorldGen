from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from runner_wrapper.adapter import (
    DA2_MODEL_ID,
    DA2_REVISION,
    OUTPUT_METADATA,
    _parameters,
    _prepare_input,
    _write_graphdeco_ply,
    run_job,
)

try:
    import numpy as np
    from plyfile import PlyData
except ImportError:
    np = None
    PlyData = None


def request(root: Path, image_path: Path, **metadata: object) -> dict:
    return {
        "contract_version": 1,
        "job": {
            "job_id": "worldgen-test",
            "batch_id": "worldgen-batch",
            "job_type": "generation",
            "primary_sample": "frame-1",
            "primary_sample_metadata": {
                "projection": "equirectangular",
                "fov": [360, 180],
                **metadata,
            },
            "attempt": 1,
            "timeout_seconds": 3600,
            "parameters": {},
        },
        "inputs": {"data": {"frame-1": {"image": str(image_path)}}},
        "runtime": {"workspace_dir": str(root / "workspace")},
    }


class WorldGenAdapterTests(unittest.TestCase):
    def test_da2_checkpoint_is_revision_pinned(self) -> None:
        self.assertEqual(DA2_MODEL_ID, "haodongli/DA-2")
        self.assertRegex(DA2_REVISION, r"^[0-9a-f]{40}$")

    def test_output_metadata_describes_worldgen_coordinates(self) -> None:
        self.assertEqual(
            OUTPUT_METADATA,
            {
                "scene_scale": 1.0,
                "scene_coordinate_system": "RDF",
                "scene_units": "relative",
                "scene_origin": "primary_viewpoint",
            },
        )

    def test_parameters_reject_unknown_values(self) -> None:
        self.assertEqual(_parameters(None), {})
        self.assertEqual(_parameters({}), {})
        with self.assertRaisesRegex(ValueError, "unknown job parameters"):
            _parameters({"resolution": 1024})

    def test_prepare_input_accepts_full_equirectangular_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            destination = root / "prepared.png"
            Image.new("RGB", (32, 16), (20, 40, 60)).save(source)
            primary, original, resolution = _prepare_input(
                request(root, source), destination
            )
            self.assertEqual(primary, "frame-1")
            self.assertEqual(original, source)
            self.assertEqual(resolution, (32, 16))
            with Image.open(destination) as image:
                self.assertEqual(image.size, (32, 16))
                self.assertEqual(image.mode, "RGB")

    def test_prepare_input_rejects_wrong_projection_and_aspect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            Image.new("RGB", (32, 16)).save(source)
            with self.assertRaisesRegex(ValueError, "projection must be equirectangular"):
                _prepare_input(
                    request(root, source, projection="pinhole"), root / "prepared.png"
                )
            Image.new("RGB", (32, 15)).save(source)
            with self.assertRaisesRegex(ValueError, "2:1 equirectangular"):
                _prepare_input(request(root, source), root / "prepared.png")

    def test_run_job_returns_3dgs_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            Image.new("RGB", (32, 16), (20, 40, 60)).save(source)

            def fake_writer(_splat: object, destination: Path) -> int:
                destination.write_bytes(b"ply\n")
                return 123

            with (
                patch("runner_wrapper.adapter._generate_splat", return_value=object()),
                patch("runner_wrapper.adapter._write_graphdeco_ply", side_effect=fake_writer),
                patch.dict(os.environ, {"PATH_MODEL_CACHE": str(root / "model-cache")}),
            ):
                result = run_job(request(root, source))

            self.assertEqual(result["status"], "completed")
            output_name = result["output_files"]["frame-1"]["3dgs"]
            self.assertTrue((root / "workspace" / output_name).is_file())
            self.assertEqual(result["output_metadata"], OUTPUT_METADATA)
            gaussian_count = next(
                metric for metric in result["metrics"] if metric["name"] == "gaussian_count"
            )
            self.assertEqual(gaussian_count["value"], 123)
            checkpoint_revision = next(
                metric
                for metric in result["metrics"]
                if metric["name"] == "checkpoint_revision"
            )
            self.assertEqual(checkpoint_revision["value"], DA2_REVISION)

    @unittest.skipUnless(np is not None and PlyData is not None, "NumPy and plyfile are not installed")
    def test_graphdeco_export_converts_alpha_and_canonicalizes_scale(self) -> None:
        assert np is not None
        assert PlyData is not None
        splat = SimpleNamespace(
            centers=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            rgbs=np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32),
            opacities=np.asarray([[1.0]], dtype=np.float32),
            scales=np.asarray([[-1e-8, 0.5, 1.0]], dtype=np.float32),
            rotations=np.asarray([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "scene.ply"
            self.assertEqual(_write_graphdeco_ply(splat, output), 1)
            vertex = PlyData.read(output)["vertex"].data[0]
        alpha = 1.0 / (1.0 + np.exp(-float(vertex["opacity"])))
        self.assertGreater(alpha, 0.99999)
        self.assertTrue(np.isfinite(float(vertex["scale_0"])))
        self.assertAlmostEqual(float(vertex["rot_0"]), 1.0)


if __name__ == "__main__":
    unittest.main()
