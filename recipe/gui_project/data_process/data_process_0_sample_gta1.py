import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from recipe.gui_project.data_process.extract_gta1_grounding_image import has_gta1_grounding_image


TARGET_RATIO = {
    "web": 5,
    "desktop": 4,
    "mobile": 1,
}
SOURCE_COORDINATE_SIZE = (1000.0, 1000.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 0: sample GTA1 grounding inp.json into 4 json shards without touching images."
    )
    parser.add_argument(
        "--input-json",
        default="/home/kaiyu/Data/raw_data/HelloKKMe/grounding_dataset/inp.json",
        help="Raw GTA1 grounding inp.json file.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/gta1_grounding/data_process_0",
        help="Output directory. Defaults to dataset/gta1_grounding/data_process_0.",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=100000,
        help="Total number of sampled rows. Defaults to 100000.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of output json shards. Defaults to 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reservoir sampling and final shuffling.",
    )
    return parser.parse_args()


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


def has_non_empty_instruction(record: dict) -> bool:
    for turn in record.get("conversations", []):
        speaker = str(turn.get("from", "")).strip().lower()
        if speaker not in {"human", "user"}:
            continue
        raw_instruction = str(turn.get("value", ""))
        instruction = re.sub(r"(?i)^(\s*<image>\s*)+", "", raw_instruction).strip()
        return bool(instruction)
    return False


def has_valid_bbox(record: dict) -> bool:
    source_width, source_height = SOURCE_COORDINATE_SIZE
    x1, y1, x2, y2 = [float(value) for value in record["bbox"]]
    clamped = [
        min(max(x1, 0.0), source_width),
        min(max(y1, 0.0), source_height),
        min(max(x2, 0.0), source_width),
        min(max(y2, 0.0), source_height),
    ]
    return clamped[0] < clamped[2] and clamped[1] < clamped[3]


def compute_target_counts(target_total: int, raw_counts: Counter) -> dict[str, int]:
    if target_total <= 0:
        raise ValueError(f"target_total must be positive, got {target_total}")

    total_ratio = sum(TARGET_RATIO.values())
    if target_total % total_ratio != 0:
        raise ValueError(
            f"target_total must be divisible by the ratio sum {total_ratio}, got {target_total}"
        )

    ratio_unit = target_total // total_ratio
    target_counts = {category: ratio * ratio_unit for category, ratio in TARGET_RATIO.items()}

    if sum(target_counts.values()) != target_total:
        raise ValueError(f"Target counts do not sum to {target_total}: {target_counts}")

    for category, target_size in target_counts.items():
        if raw_counts[category] < target_size:
            raise ValueError(
                f"Raw count for '{category}' is smaller than target count: raw={raw_counts[category]}, target={target_size}"
            )

    return target_counts


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024):
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    reached_end = False

    with path.open("r", encoding="utf-8") as file:
        while True:
            if position >= len(buffer) and reached_end:
                break

            if position >= len(buffer) - chunk_size // 4 and not reached_end:
                remainder = buffer[position:]
                chunk = file.read(chunk_size)
                if chunk:
                    buffer = remainder + chunk
                    position = 0
                else:
                    buffer = remainder
                    position = 0
                    reached_end = True

            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1

            if not started:
                if position >= len(buffer):
                    if reached_end:
                        break
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected '[' at start of json array, got: {buffer[position]!r}")
                started = True
                position += 1
                continue

            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1

            if position >= len(buffer):
                if reached_end:
                    break
                continue

            if buffer[position] == "]":
                break

            try:
                obj, next_position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if reached_end:
                    raise
                remainder = buffer[position:]
                chunk = file.read(chunk_size)
                if not chunk:
                    raise
                buffer = remainder + chunk
                position = 0
                continue

            yield obj
            position = next_position


def run_reservoir_sampling(input_json: Path, target_counts: dict[str, int], seed: int) -> tuple[dict[str, list[dict]], Counter]:
    rng = random.Random(seed)
    raw_counts = Counter()
    valid_counts = Counter()
    reservoirs = {category: [] for category in target_counts}

    for record in iter_json_array(input_json):
        category = classify_record(record)
        raw_counts[category] += 1

        if not has_gta1_grounding_image(record["image"]):
            continue

        if not has_non_empty_instruction(record):
            continue

        if not has_valid_bbox(record):
            continue

        valid_counts[category] += 1

        reservoir = reservoirs[category]
        target_size = target_counts[category]
        seen_count = valid_counts[category]

        if len(reservoir) < target_size:
            reservoir.append(record)
            continue

        replace_index = rng.randrange(seen_count)
        if replace_index < target_size:
            reservoir[replace_index] = record

    return reservoirs, raw_counts, valid_counts


def write_json(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    args = parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # These raw counts are from the full GTA1 grounding inp.json and let us compute the
    # desired sample plan without first loading the entire dataset.
    raw_counts_reference = Counter(
        {
            "web": 1040424,
            "desktop": 432455,
            "mobile": 86656,
        }
    )
    target_counts = compute_target_counts(args.target_total, raw_counts_reference)

    reservoirs, observed_raw_counts, observed_valid_counts = run_reservoir_sampling(
        input_json=input_json,
        target_counts=target_counts,
        seed=args.seed,
    )

    for category, target_size in target_counts.items():
        actual_size = len(reservoirs[category])
        if actual_size != target_size:
            raise ValueError(
                f"Insufficient sampled rows for category '{category}': expected {target_size}, got {actual_size}"
            )

    sampled_rows = []
    for category in ("web", "desktop", "mobile"):
        sampled_rows.extend(reservoirs[category])

    rng = random.Random(args.seed)
    rng.shuffle(sampled_rows)

    if len(sampled_rows) != args.target_total:
        raise ValueError(f"Expected {args.target_total} sampled rows, got {len(sampled_rows)}")
    if args.target_total % args.num_shards != 0:
        raise ValueError(
            f"target_total must be divisible by num_shards, got target_total={args.target_total}, num_shards={args.num_shards}"
        )

    shard_size = args.target_total // args.num_shards
    shard_paths = []
    shard_category_counts = {}

    for shard_idx in range(args.num_shards):
        start = shard_idx * shard_size
        end = start + shard_size
        shard_rows = sampled_rows[start:end]
        shard_path = output_root / f"sampled_shard_{shard_idx:02d}.json"
        write_json(shard_path, shard_rows)
        shard_paths.append(str(shard_path))
        shard_category_counts[f"shard_{shard_idx:02d}"] = dict(Counter(classify_record(row) for row in shard_rows))

    summary = {
        "dataset_name": "gta1_grounding",
        "input_json": str(input_json),
        "output_root": str(output_root),
        "seed": args.seed,
        "target_total": args.target_total,
        "num_shards": args.num_shards,
        "shard_size": shard_size,
        "sampling_rule": "sample by platform ratio web:desktop:mobile = 5:4:1",
        "target_ratio": TARGET_RATIO,
        "target_counts": target_counts,
        "observed_raw_counts": dict(observed_raw_counts),
        "observed_valid_counts": dict(observed_valid_counts),
        "output_shards": shard_paths,
        "output_shard_category_counts": shard_category_counts,
    }
    write_json(output_root / "summary.json", summary)

    print(f"Saved {args.num_shards} sampled json shards under {output_root}")
    print(f"Target counts: {target_counts}")
    print(f"Observed raw counts: {dict(observed_raw_counts)}")
    print(f"Observed valid counts: {dict(observed_valid_counts)}")


if __name__ == "__main__":
    main()
