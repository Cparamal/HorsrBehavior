from dataclasses import dataclass, field

from horse_behavior.infer_behavior import Detection


CONTEXT_CLASSES = {"grass", "water"}


def should_run_detector(frame_index: int, interval: int) -> bool:
    interval = max(1, int(interval))
    return int(frame_index) % interval == 0


def filter_context_detections(detections: list[Detection]) -> list[Detection]:
    return [d for d in detections if d.name in CONTEXT_CLASSES]


@dataclass
class DetectionContextCache:
    ttl_frames: int
    detections: list[Detection] = field(default_factory=list)
    updated_frame_index: int | None = None

    def update(self, frame_index: int, detections: list[Detection]) -> list[Detection]:
        self.detections = filter_context_detections(detections)
        self.updated_frame_index = int(frame_index)
        return list(self.detections)

    def current(self, frame_index: int) -> list[Detection]:
        if self.updated_frame_index is None:
            return []
        if int(frame_index) - self.updated_frame_index > max(0, int(self.ttl_frames)):
            return []
        return list(self.detections)
