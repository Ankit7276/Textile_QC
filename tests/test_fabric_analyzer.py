"""Unit and pipeline tests for the fabric analyzer."""
import os
import tempfile
import unittest

import cv2
import numpy as np

from fabric_analyzer import (
    _peak_freq,
    analyze,
    color_features,
    detect_defects,
    flatten_lighting,
    pattern_regularity,
    thread_count,
)


class FabricAnalyzerTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:256, :256]
        weave = 128 + 28 * np.sin(2 * np.pi * x / 8) + 28 * np.sin(2 * np.pi * y / 8)
        self.gray = np.clip(weave, 0, 255).astype(np.uint8)
        self.bgr = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

    def test_peak_frequency_finds_known_period(self):
        x = np.arange(512)
        profile = np.sin(2 * np.pi * x / 16)
        frequency, strength = _peak_freq(profile)
        self.assertAlmostEqual(frequency, 1 / 16, delta=1 / (512 * 8))
        self.assertGreater(strength, 1)

    def test_flatten_lighting_preserves_shape_and_texture(self):
        flattened = flatten_lighting(self.gray)
        self.assertEqual(flattened.shape, self.gray.shape)
        self.assertTrue(np.isfinite(flattened).all())
        self.assertGreater(float(flattened.std()), 1.0)

    def test_thread_count_detects_synthetic_weave_spacing(self):
        measured = thread_count(self.gray.astype(np.float32), ppi=800)
        self.assertAlmostEqual(measured["vertical_threads"]["tpi_fft"], 100, delta=4)
        self.assertAlmostEqual(measured["horizontal_threads"]["tpi_fft"], 100, delta=4)
        self.assertAlmostEqual(measured["thread_count_fft"], 200, delta=8)

    def test_blank_fabric_has_no_detected_defects(self):
        blank = np.full((128, 128), 120, dtype=np.uint8)
        bgr = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
        defects = detect_defects(bgr, flatten_lighting(blank), ppi=300, period_px=8)
        self.assertEqual(defects["total"], 0)
        self.assertEqual(defects["items"], [])

    def test_color_features_convert_bgr_to_rgb(self):
        bgr = np.full((20, 20, 3), (10, 20, 30), dtype=np.uint8)
        features = color_features(bgr)
        self.assertEqual([round(v) for v in features["mean_rgb"]], [30, 20, 10])
        self.assertEqual(set(features["hist_rgb_8bins"]), {"R", "G", "B"})

    def test_pattern_regularity_returns_finite_score(self):
        flattened = flatten_lighting(self.gray)
        regularity = pattern_regularity(flattened)
        self.assertTrue(np.isfinite(regularity["regularity_score"]))
        self.assertGreaterEqual(regularity["regularity_score"], 0)
        self.assertLessEqual(regularity["regularity_score"], 1)
        self.assertIsInstance(regularity["low_periodicity"], bool)

    def test_analyze_pipeline_writes_overlay_and_returns_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "woven.png")
            overlay = os.path.join(directory, "overlay.png")
            self.assertTrue(cv2.imwrite(source, self.bgr))
            result = analyze(source, width_in=2.0, overlay_path=overlay)
            self.assertTrue(os.path.isfile(overlay))
            self.assertGreater(os.path.getsize(overlay), 0)
            self.assertAlmostEqual(result["scale_ppi"], 128.0)
            for key in ("thread_count", "defects", "texture", "color", "edges", "regularity"):
                self.assertIn(key, result)

    def test_analyze_rejects_missing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "missing.png")
            with self.assertRaises(FileNotFoundError):
                analyze(missing, ppi=300)


if __name__ == "__main__":
    unittest.main()
