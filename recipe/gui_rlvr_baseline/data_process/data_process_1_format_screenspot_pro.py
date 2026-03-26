import argparse
import json
import random
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: normalize ScreenSpot-Pro into human-readable json + image folder."
    )
    parser.add_argument(
        "--input-root",
        default="/home/kaiyu/Data/raw_data/ScreenSpot-Pro",
        help="Raw ScreenSpot-Pro root containing annotations/ and images/.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/screenspot_pro_grounding/data_process_1",
        help="Output directory. Defaults to dataset/screenspot_pro_grounding/data_process_1.",
    )
    parser.add_argument(
        "--instruction-field",
        default="instruction",
        choices=("instruction", "instruction_cn"),
        help="Which raw instruction field becomes record['instruction'].",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for train/val split.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional cap for debugging.")
    return parser.parse_args()


def load_raw_records(annotations_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        with annotation_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"Annotation file must contain a list: {annotation_path}")
        for record in records:
            rows.append(
                {
                    "annotation_file": annotation_path.name,
                    "annotation_stem": annotation_path.stem,
                    "record": record,
                }
            )
    return rows


def load_image_size_and_bytes(image_path: Path) -> tuple[tuple[int, int], bytes]:
    from PIL import Image

    image_bytes = image_path.read_bytes()
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
    return image_size, image_bytes


def sanitize_filename_component(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "image"


def make_output_image_name(sample_id: int, annotation_stem: str, img_filename: str) -> str:
    basename = Path(img_filename).name
    safe_basename = sanitize_filename_component(basename)
    safe_annotation = sanitize_filename_component(annotation_stem)
    return f"{sample_id:06d}_{safe_annotation}_{safe_basename}"


def clamp_bbox_xyxy(bbox_xyxy: list[float], image_size: tuple[int, int]) -> tuple[list[float], bool]:
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]

    clamped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    was_clamped = clamped != [x1, y1, x2, y2]

    if not (clamped[0] < clamped[2] and clamped[1] < clamped[3]):
        raise ValueError(f"Invalid bbox after clamping: raw={bbox_xyxy}, clamped={clamped}, image_size={image_size}")
    return clamped, was_clamped


def build_normalized_record(
    row: dict,
    sample_id: int,
    split: str,
    images_root: Path,
    output_root: Path,
    instruction_field: str,
) -> tuple[dict, bool]:
    sample = row["record"]
    raw_image_rel_path = Path(sample["img_filename"])
    raw_image_abs_path = images_root / raw_image_rel_path
    if not raw_image_abs_path.exists():
        raise FileNotFoundError(f"Image not found: {raw_image_abs_path}")

    image_size, image_bytes = load_image_size_and_bytes(raw_image_abs_path)
    annotated_image_size = tuple(int(value) for value in sample["img_size"])
    if annotated_image_size != image_size:
        raise ValueError(
            f"Image size mismatch for {raw_image_rel_path}: annotation={annotated_image_size}, actual={image_size}"
        )

    bbox_xyxy, was_clamped = clamp_bbox_xyxy(sample["bbox"], image_size=image_size)

    image_name = make_output_image_name(
        sample_id=sample_id,
        annotation_stem=row["annotation_stem"],
        img_filename=sample["img_filename"],
    )
    image_rel_path = Path(split) / image_name
    image_abs_path = output_root / "images" / image_rel_path
    image_abs_path.parent.mkdir(parents=True, exist_ok=True)
    image_abs_path.write_bytes(image_bytes)

    normalized = {
        "id": sample_id,
        "split": split,
        "instruction": str(sample[instruction_field]).strip(),
        "image": image_rel_path.as_posix(),
        "bbox_xyxy": bbox_xyxy,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "meta": {
            "instruction_cn": sample.get("instruction_cn", "").strip(),
            "application": sample.get("application"),
            "platform": sample.get("platform"),
            "ui_type": sample.get("ui_type"),
            "group": sample.get("group"),
        },
        "source": {
            "annotation_file": row["annotation_file"],
            "raw_id": sample.get("id"),
            "raw_img_filename": raw_image_rel_path.as_posix(),
            "raw_bbox_xyxy": [float(value) for value in sample["bbox"]],
            "raw_instruction_field": instruction_field,
            "bbox_was_clamped": was_clamped,
        },
    }
    return normalized, was_clamped


def write_json(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")


def compute_val_indices(num_records: int, val_ratio: float, seed: int) -> set[int]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")

    indices = list(range(num_records))
    random.Random(seed).shuffle(indices)
    val_size = int(num_records * val_ratio)
    return set(indices[:val_size])


def main():
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    annotations_dir = input_root / "annotations"
    images_root = input_root / "images"
    output_root = Path(args.output_root).expanduser().resolve()

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    records = load_raw_records(annotations_dir=annotations_dir)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    num_records = len(records)
    if num_records < 1:
        raise ValueError(f"Need at least 1 sample, got {num_records}")

    val_indices = compute_val_indices(num_records=num_records, val_ratio=args.val_ratio, seed=args.seed)

    train_rows = []
    val_rows = []
    num_bbox_clamped = 0

    for sample_id, row in enumerate(records):
        split = "val" if sample_id in val_indices else "train"
        normalized, was_clamped = build_normalized_record(
            row=row,
            sample_id=sample_id,
            split=split,
            images_root=images_root,
            output_root=output_root,
            instruction_field=args.instruction_field,
        )
        num_bbox_clamped += int(was_clamped)
        if split == "train":
            train_rows.append(normalized)
        else:
            val_rows.append(normalized)

    write_json(output_root / "train.json", train_rows)
    write_json(output_root / "val.json", val_rows)

    summary = {
        "dataset_name": "screenspot_pro_grounding",
        "num_total": len(train_rows) + len(val_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_annotation_files": len(list(annotations_dir.glob("*.json"))),
        "num_bbox_clamped": num_bbox_clamped,
        "input_root": str(input_root),
        "instruction_field": args.instruction_field,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved normalized train json to {output_root / 'train.json'}")
    print(f"Saved normalized val json to {output_root / 'val.json'}")
    print(f"Saved normalized images under {output_root / 'images'}")


if __name__ == "__main__":
    main()
