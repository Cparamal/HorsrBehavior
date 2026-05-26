import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BehaviorRow:
    split: str
    image: str
    label: str


def read_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not classes:
        raise RuntimeError(f"No classes found in {path}")
    return classes


def read_behavior_rows(path: Path) -> list[BehaviorRow]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"split", "image", "label"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise RuntimeError(f"{path} missing required columns: {', '.join(missing)}")
        rows = []
        for line_number, row in enumerate(reader, 2):
            split = (row.get("split") or "").strip()
            image = (row.get("image") or "").strip()
            label = (row.get("label") or "").strip()
            if not split or not image or not label:
                raise RuntimeError(f"{path}:{line_number}: split, image and label are required")
            rows.append(BehaviorRow(split=split, image=image, label=label))
    return rows
