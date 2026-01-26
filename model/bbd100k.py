# convert_bdd100k.py
import json
from pathlib import Path


def convert_labels(input_dir, output_dir, img_width=1280, img_height=720):
    """Convert BDD100K per-image JSON files to YOLO txt format."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_map = {
        "pedestrian": 0,
        "rider": 1,
        "car": 2,
        "truck": 3,
        "bus": 4,
        "train": 5,
        "motorcycle": 6,
        "bicycle": 7,
        "traffic light": 8,
        "traffic sign": 9,
    }

    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    for i, json_file in enumerate(json_files):
        if i % 10000 == 0:
            print(f"Processing {i}/{len(json_files)}...")

        with open(json_file) as f:
            data = json.load(f)

        lines = []

        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                category = obj.get("category", "")

                if category not in class_map:
                    continue

                if "box2d" not in obj:
                    continue

                box = obj["box2d"]
                x_center = (box["x1"] + box["x2"]) / 2 / img_width
                y_center = (box["y1"] + box["y2"]) / 2 / img_height
                width = (box["x2"] - box["x1"]) / img_width
                height = (box["y2"] - box["y1"]) / img_height

                # Clamp values to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                lines.append(
                    f"{class_map[category]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        txt_file = output_dir / f"{json_file.stem}.txt"
        txt_file.write_text("\n".join(lines))

    print(f"Done! Converted {len(json_files)} files to {output_dir}")


if __name__ == "__main__":
    base = Path("./")

    # Convert train labels
    convert_labels(base / "labels_json_backup/100k/train", base / "labels/100k/train")

    # Convert val labels
    convert_labels(base / "labels_json_backup/100k/val", base / "labels/100k/val")
