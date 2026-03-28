import argparse
import json
import os
import random
from io import BytesIO
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: normalize UI-AGILE data into human-readable json + image folder.")
    parser.add_argument(
        "--input-json",
        default="/home/kaiyu/Data/raw_data/InfiX-ai/UI-AGILE-Data/train_filtered.json",
        help="Raw filtered UI-AGILE json file.",
    )
    parser.add_argument(
        "--images-zip",
        default="/home/kaiyu/Data/raw_data/InfiX-ai/UI-AGILE-Data/training_data/train_imgs.zip",
        help="Zip file containing the screenshots.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/ui_agile_grounding/data_process_1",
        help="Output directory. Defaults to dataset/ui_agile_grounding/data_process_1.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for train/val split.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional cap for debugging.")
    return parser.parse_args()


def resolve_zip_member(zip_members: set[str], img_filename: str) -> str:
    normalized = img_filename.strip().lstrip("/").replace("\\", "/")
    direct_candidates = [normalized, f"train_imgs/{normalized}"]
    for candidate in direct_candidates:
        if candidate in zip_members:
            return candidate

    basename = os.path.basename(normalized)
    suffix_matches = [member for member in zip_members if member.endswith(f"/{basename}") or member == basename]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if not suffix_matches:
        raise FileNotFoundError(f"Image '{img_filename}' not found in zip file")
    raise ValueError(f"Image '{img_filename}' is ambiguous inside zip: {suffix_matches[:5]}")


def load_image_size_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        return image.size


def make_output_image_name(sample_id: int, img_filename: str) -> str:
    basename = os.path.basename(img_filename.strip().replace("\\", "/"))
    return f"{sample_id:06d}_{basename}"


def build_normalized_record(
    sample: dict,
    sample_id: int,
    split: str,
    zip_file: zipfile.ZipFile,
    zip_members: set[str],
    output_root: Path,
) -> dict:
    original_img_filename = sample["img_filename"]
    zip_member = resolve_zip_member(zip_members, original_img_filename)
    image_bytes = zip_file.read(zip_member)
    image_width, image_height = load_image_size_from_bytes(image_bytes)

    image_name = make_output_image_name(sample_id=sample_id, img_filename=original_img_filename)
    image_rel_path = Path(split) / image_name
    image_abs_path = output_root / "images" / image_rel_path
    image_abs_path.parent.mkdir(parents=True, exist_ok=True)
    image_abs_path.write_bytes(image_bytes)

    return {
        "id": sample_id,
        "split": split,
        "instruction": sample["instruction"].strip(),
        "image": image_rel_path.as_posix(),
        "bbox_xyxy": [float(value) for value in sample["gt_bbox"]],
        "image_size": [int(image_width), int(image_height)],
        "meta": sample.get("metadata", {}),
        "source": {
            "raw_img_filename": original_img_filename,
            "zip_member": zip_member,
        },
    }


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

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as file:
        records = json.load(file)

    if args.max_samples > 0:
        records = records[: args.max_samples]

    num_records = len(records)
    if num_records < 1:
        raise ValueError(f"Need at least 1 sample, got {num_records}")

    val_indices = compute_val_indices(num_records=num_records, val_ratio=args.val_ratio, seed=args.seed)

    train_rows = []
    val_rows = []

    with zipfile.ZipFile(args.images_zip) as zip_file:
        zip_members = {name for name in zip_file.namelist() if not name.endswith("/")}
        for sample_id, sample in enumerate(records):
            split = "val" if sample_id in val_indices else "train"
            normalized = build_normalized_record(
                sample=sample,
                sample_id=sample_id,
                split=split,
                zip_file=zip_file,
                zip_members=zip_members,
                output_root=output_root,
            )
            if split == "train":
                train_rows.append(normalized)
            else:
                val_rows.append(normalized)

    write_json(output_root / "train.json", train_rows)
    write_json(output_root / "val.json", val_rows)

    summary = {
        "dataset_name": "ui_agile_grounding",
        "num_total": len(train_rows) + len(val_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "input_json": args.input_json,
        "images_zip": args.images_zip,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved normalized train json to {output_root / 'train.json'}")
    print(f"Saved normalized val json to {output_root / 'val.json'}")
    print(f"Saved normalized images under {output_root / 'images'}")


if __name__ == "__main__":
    main()

# python -m recipe.gui_project.data_process.data_process_1_format_ui_agile \
#     --output-root "dataset/ui_agile_grounding/data_process_1" \
#     --val-ratio 0.0
