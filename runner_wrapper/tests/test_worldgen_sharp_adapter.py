from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from runner_wrapper.sharp_adapter import (
    SHARP_CHECKPOINT_BYTES,
    SHARP_CHECKPOINT_ETAG,
    SHARP_SOURCE_REVISION,
    _multipart_etag,
    _verify_checkpoint,
    run_job,
)
from runner_wrapper.tests.test_worldgen_adapter import request


class WorldGenSharpAdapterTests(unittest.TestCase):
    def test_sharp_assets_are_pinned(self) -> None:
        self.assertEqual(SHARP_CHECKPOINT_BYTES, 2_809_738_232)
        self.assertEqual(SHARP_CHECKPOINT_ETAG, "0d0d57485fdd957aa240fa996ed7fdaa-42")
        self.assertRegex(SHARP_SOURCE_REVISION, r"^[0-9a-f]{40}$")

    def test_multipart_checkpoint_verification(self) -> None:
        payload = b"abcdefghijk"
        part_size = 4
        parts = [payload[index : index + part_size] for index in range(0, len(payload), part_size)]
        digests = [hashlib.md5(part, usedforsecurity=False).digest() for part in parts]
        expected = (
            hashlib.md5(b"".join(digests), usedforsecurity=False).hexdigest()
            + f"-{len(parts)}"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            checkpoint.write_bytes(payload)
            self.assertEqual(_multipart_etag(checkpoint, part_size), expected)
            _verify_checkpoint(checkpoint, len(payload), expected, part_size)
            with self.assertRaisesRegex(RuntimeError, "expected 12"):
                _verify_checkpoint(checkpoint, 12, expected, part_size)

    def test_run_job_uses_sharp_mode_and_reports_asset_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            Image.new("RGB", (32, 16), (20, 40, 60)).save(source)

            def fake_writer(_splat: object, destination: Path) -> int:
                destination.write_bytes(b"ply\n")
                return 456

            with (
                patch("runner_wrapper.adapter._generate_splat", return_value=object()) as generate,
                patch("runner_wrapper.adapter._write_graphdeco_ply", side_effect=fake_writer),
                patch.dict(os.environ, {"PATH_MODEL_CACHE": str(root / "model-cache")}),
            ):
                result = run_job(request(root, source))

            self.assertEqual(result["status"], "completed")
            generate.assert_called_once()
            self.assertEqual(generate.call_args.args[1], "sharp")
            output_name = result["output_files"]["frame-1"]["3dgs"]
            self.assertTrue(output_name.startswith("3DGS-sharp-"))
            metrics = {metric["name"]: metric["value"] for metric in result["metrics"]}
            self.assertEqual(metrics["gaussian_count"], 456)
            self.assertEqual(metrics["splat_mode"], "sharp")
            self.assertEqual(metrics["sharp_checkpoint_etag"], SHARP_CHECKPOINT_ETAG)
            self.assertEqual(metrics["sharp_source_revision"], SHARP_SOURCE_REVISION)


if __name__ == "__main__":
    unittest.main()
