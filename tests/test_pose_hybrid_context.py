import unittest

from horse_behavior.infer_behavior import Detection
from horse_behavior.pose_hybrid_context import DetectionContextCache, filter_context_detections, should_run_detector


def det(name, xyxy=(0.0, 0.0, 10.0, 10.0), conf=0.9):
    return Detection(name=name, conf=conf, xyxy=xyxy)


class PoseHybridContextTests(unittest.TestCase):
    def test_should_run_detector_uses_interval(self):
        self.assertTrue(should_run_detector(0, 8))
        self.assertFalse(should_run_detector(7, 8))
        self.assertTrue(should_run_detector(8, 8))
        self.assertTrue(should_run_detector(3, 1))

    def test_filter_context_detections_keeps_grass_and_water_only(self):
        filtered = filter_context_detections([det("horse"), det("grass"), det("water"), det("head")])

        self.assertEqual([d.name for d in filtered], ["grass", "water"])

    def test_cache_reuses_detection_until_ttl_expires(self):
        cache = DetectionContextCache(ttl_frames=3)
        cache.update(frame_index=10, detections=[det("grass")])

        self.assertEqual([d.name for d in cache.current(frame_index=12)], ["grass"])
        self.assertEqual(cache.current(frame_index=14), [])

    def test_cache_empty_before_update(self):
        cache = DetectionContextCache(ttl_frames=3)

        self.assertEqual(cache.current(frame_index=0), [])


if __name__ == "__main__":
    unittest.main()
