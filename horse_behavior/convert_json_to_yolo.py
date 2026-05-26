import json
import random
from pathlib import Path

CLASS_MAP = {
    "horse": 0,
    "head": 1,
    "grass": 2,
    "water": 3,
    "lying_horse": 4,
}

src_dirs = [
    Path("datasets/images1"),
    Path("datasets/images2"),
    Path("datasets/images3"),
    Path("datasets/images4"),
]

out_dir = Path("dataset")
img_train = out_dir / "images/train"
img_val = out_dir / "images/val"
lbl_train = out_dir / "labels/train"
lbl_val = out_dir / "labels/val"

for d in [img_train, img_val, lbl_train, lbl_val]:
    d.mkdir(parents=True, exist_ok=True)
    # Clean old files
    for f in d.iterdir():
        f.unlink()

# Collect all image-label pairs
pairs = []
for src_dir in src_dirs:
    for jf in sorted(src_dir.glob("*.json")):
        stem = jf.stem
        # Find matching image
        img_file = src_dir / f"{stem}.jpg"
        if not img_file.exists():
            img_file = src_dir / f"{stem}.png"
        if img_file.exists():
            pairs.append((jf, img_file, src_dir.name))

random.seed(42)
random.shuffle(pairs)

total = len(pairs)
split = int(total * 0.8)

print(f"Total: {total}, Train: {split}, Val: {total - split}")

class_counts = {k: 0 for k in CLASS_MAP}

for i, (jf, img_file, folder_name) in enumerate(pairs):
    data = json.loads(jf.read_text(encoding="utf-8"))
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            print(f"  Unknown label '{label}' in {folder_name}/{jf.name}, skipping")
            continue
        cls_id = CLASS_MAP[label]
        class_counts[label] += 1
        pts = shape["points"]
        x1, y1 = pts[0]
        x2, y2 = pts[2]
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Use folder prefix to avoid name collision
    new_stem = f"{folder_name}_{jf.stem}"

    if i < split:
        dst_img = img_train / f"{new_stem}.jpg"
        dst_lbl = lbl_train / f"{new_stem}.txt"
    else:
        dst_img = img_val / f"{new_stem}.jpg"
        dst_lbl = lbl_val / f"{new_stem}.txt"

    dst_img.write_bytes(img_file.read_bytes())
    dst_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

print(f"Train: {min(split, len(pairs))}, Val: {len(pairs) - min(split, len(pairs))}")

# Write data.yaml
yaml_content = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: horse
  1: head
  2: grass
  3: water
  4: lying_horse
"""
(out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
print("data.yaml written")
print(f"Class distribution: {class_counts}")
