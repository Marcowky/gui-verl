import argparse
import json
import re
from collections import Counter
from pathlib import Path

from recipe.gui_project.data_process.extract_gta1_grounding_image import (
    get_image_size_from_bytes,
    read_gta1_grounding_image_bytes,
)

SOURCE_COORDINATE_SIZE = (1000.0, 1000.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: normalize sampled GTA1 grounding json shards into data_process_2-compatible json + image folder."
    )
    parser.add_argument(
        "--input-root",
        default="dataset/gta1_grounding/data_process_0",
        help="Input directory containing sampled_shard_*.json from stage 0.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/gta1_grounding/data_process_1",
        help="Output directory. Defaults to dataset/gta1_grounding/data_process_1.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def find_input_shards(input_root: Path) -> list[Path]:
    shard_paths = sorted(input_root.glob("sampled_shard_*.json"))
    if len(shard_paths) != 4:
        raise ValueError(f"Expected exactly 4 sampled shards under {input_root}, got {len(shard_paths)}")
    return shard_paths


def extract_instruction(conversations: list[dict]) -> tuple[str, str]:
    for turn in conversations:
        speaker = str(turn.get("from", "")).strip().lower()
        if speaker not in {"human", "user"}:
            continue

        raw_instruction = str(turn.get("value", ""))
        instruction = re.sub(r"(?i)^(\s*<image>\s*)+", "", raw_instruction).strip()
        if not instruction:
            raise ValueError(f"Instruction becomes empty after removing <image> tag: {raw_instruction!r}")
        return instruction, raw_instruction

    raise ValueError(f"No human/user instruction found in conversations: {conversations!r}")


def scale_and_clamp_bbox_xyxy(bbox_xyxy: list[float], image_size: tuple[int, int]) -> tuple[list[float], bool]:
    source_width, source_height = SOURCE_COORDINATE_SIZE
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]

    scaled = [
        x1 * float(width) / source_width,
        y1 * float(height) / source_height,
        x2 * float(width) / source_width,
        y2 * float(height) / source_height,
    ]

    clamped = [
        min(max(scaled[0], 0.0), float(width)),
        min(max(scaled[1], 0.0), float(height)),
        min(max(scaled[2], 0.0), float(width)),
        min(max(scaled[3], 0.0), float(height)),
    ]
    was_clamped = clamped != scaled

    if not (clamped[0] < clamped[2] and clamped[1] < clamped[3]):
        raise ValueError(
            f"Invalid bbox after scaling/clamping: raw={bbox_xyxy}, scaled={scaled}, clamped={clamped}, image_size={image_size}"
        )

    return clamped, was_clamped


def classify_record(record: dict) -> str:
    image_path = str(record["image"])

    if image_path.startswith("dataset/Aria-UI_Data/web/"):
        return "web"
    if image_path.startswith("dataset/Aria-UI_Context-aware_Data/AMEX/"):
        return "web"
    if image_path.startswith("dataset/Aria-UI_Context-aware_Data/android_control/"):
        return "mobile"
    if image_path.startswith("desktop_domain/"):
        return "desktop"
    if image_path.startswith("dataset/omniact/"):
        return "desktop"

    raise ValueError(f"Unknown GTA1 grounding source for image path: {image_path}")


def materialize_image(
    raw_image_rel_path: str,
    output_images_root: Path,
    image_cache: dict[str, dict],
) -> tuple[str, tuple[int, int]]:
    image_rel_path = Path(raw_image_rel_path).as_posix()
    image_abs_path = output_images_root / image_rel_path

    cached = image_cache.get(image_rel_path)
    if cached is not None:
        return image_rel_path, tuple(cached["image_size"])

    image_abs_path.parent.mkdir(parents=True, exist_ok=True)
    if image_abs_path.exists():
        image_bytes = image_abs_path.read_bytes()
    else:
        image_bytes = read_gta1_grounding_image_bytes(image_rel_path)
        image_abs_path.write_bytes(image_bytes)

    try:
        image_size = get_image_size_from_bytes(image_bytes)
    except Exception as exc:
        raise ValueError(f"Failed to decode image bytes for {image_rel_path}: {exc}") from exc

    image_cache[image_rel_path] = {
        "image_size": [int(image_size[0]), int(image_size[1])],
    }
    return image_rel_path, image_size


def build_normalized_record(
    record: dict,
    new_id: int,
    source_shard: str,
    output_images_root: Path,
    image_cache: dict[str, dict],
) -> tuple[dict, bool]:
    raw_image_rel_path = str(record["image"]).strip()
    image_rel_path, image_size = materialize_image(
        raw_image_rel_path=raw_image_rel_path,
        output_images_root=output_images_root,
        image_cache=image_cache,
    )

    instruction, raw_instruction = extract_instruction(record["conversations"])
    bbox_xyxy, was_clamped = scale_and_clamp_bbox_xyxy(record["bbox"], image_size=image_size)

    normalized = {
        "id": new_id,
        "split": "train",
        "instruction": instruction,
        "image": image_rel_path,
        "bbox_xyxy": bbox_xyxy,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "meta": {
            "platform_group": classify_record(record),
        },
        "source": {
            "source_shard": source_shard,
            "raw_id": record.get("id"),
            "raw_image": raw_image_rel_path,
            "raw_bbox_xyxy": [float(value) for value in record["bbox"]],
            "raw_bbox_coordinate_size": [int(SOURCE_COORDINATE_SIZE[0]), int(SOURCE_COORDINATE_SIZE[1])],
            "raw_instruction": raw_instruction,
            "bbox_was_clamped": was_clamped,
        },
    }
    return normalized, was_clamped


def main():
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    output_images_root.mkdir(parents=True, exist_ok=True)

    input_shards = find_input_shards(input_root)

    image_cache: dict[str, dict] = {}
    shard_summaries = {}
    next_id = 0
    total_bbox_clamped = 0
    total_category_counts = Counter()
    output_json_paths = []
    skipped_records = []
    skipped_counts = Counter()

    for input_shard_path in input_shards:
        records = load_json(input_shard_path)
        normalized_rows = []
        shard_category_counts = Counter()
        shard_bbox_clamped = 0
        shard_skipped = 0

        for row_idx, record in enumerate(records):
            try:
                normalized, was_clamped = build_normalized_record(
                    record=record,
                    new_id=next_id,
                    source_shard=input_shard_path.name,
                    output_images_root=output_images_root,
                    image_cache=image_cache,
                )
            except Exception as exc:
                shard_skipped += 1
                skipped_counts[type(exc).__name__] += 1
                if len(skipped_records) < 100:
                    skipped_records.append(
                        {
                            "source_shard": input_shard_path.name,
                            "row_index": row_idx,
                            "raw_image": record.get("image"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                continue

            next_id += 1
            normalized_rows.append(normalized)

            category = normalized["meta"]["platform_group"]
            shard_category_counts[category] += 1
            total_category_counts[category] += 1
            shard_bbox_clamped += int(was_clamped)
            total_bbox_clamped += int(was_clamped)

        output_json_path = output_root / input_shard_path.name
        write_json(output_json_path, normalized_rows)
        output_json_paths.append(str(output_json_path))

        shard_summaries[input_shard_path.name] = {
            "num_rows": len(normalized_rows),
            "category_counts": dict(shard_category_counts),
            "num_bbox_clamped": shard_bbox_clamped,
            "num_skipped": shard_skipped,
        }

    summary = {
        "dataset_name": "gta1_grounding",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_rows": next_id,
        "num_unique_images": len(image_cache),
        "num_bbox_clamped": total_bbox_clamped,
        "category_counts": dict(total_category_counts),
        "num_skipped": sum(skipped_counts.values()),
        "skipped_error_counts": dict(skipped_counts),
        "skipped_examples": skipped_records,
        "output_json_paths": output_json_paths,
        "images_root": str(output_images_root),
        "shards": shard_summaries,
    }
    write_json(output_root / "summary.json", summary)

    print(f"Saved normalized GTA1 json shards under {output_root}")
    print(f"Saved extracted images under {output_images_root}")
    print(f"Total rows: {next_id}")
    print(f"Unique images: {len(image_cache)}")


if __name__ == "__main__":
    main()
