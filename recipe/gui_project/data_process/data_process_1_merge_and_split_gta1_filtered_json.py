import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple filtered json files, then split merged records into train/val json files."
    )
    parser.add_argument(
        "--input-jsons",
        nargs="+",
        required=True,
        help="Input json files. Each file must contain a json array of records.",
    )
    parser.add_argument(
        "--all-output",
        required=True,
        help="Output path for the merged json array.",
    )
    parser.add_argument(
        "--train-output",
        required=True,
        help="Output path for the train split json array.",
    )
    parser.add_argument(
        "--val-output",
        required=True,
        help="Output path for the val split json array.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        required=True,
        help="Number of records to sample into the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/val split.",
    )
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise TypeError(f"Expected json array in {path}, got {type(records).__name__}")
    return records


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def merge_records(input_paths: list[Path]) -> list[dict]:
    merged_records = []
    for path in input_paths:
        merged_records.extend(load_json_array(path))
    return merged_records


def compute_val_indices(num_records: int, val_size: int, seed: int) -> set[int]:
    if val_size < 0:
        raise ValueError(f"val_size must be >= 0, got {val_size}")
    if val_size > num_records:
        raise ValueError(f"val_size={val_size} exceeds total records={num_records}")

    indices = list(range(num_records))
    random.Random(seed).shuffle(indices)
    return set(indices[:val_size])


def build_split_records(records: list[dict], val_indices: set[int]) -> tuple[list[dict], list[dict]]:
    train_rows = []
    val_rows = []

    for index, record in enumerate(records):
        updated = dict(record)
        if index in val_indices:
            updated["split"] = "val"
            val_rows.append(updated)
        else:
            updated["split"] = "train"
            train_rows.append(updated)

    return train_rows, val_rows


def main():
    args = parse_args()

    input_paths = [Path(path).expanduser().resolve() for path in args.input_jsons]
    all_output = Path(args.all_output).expanduser().resolve()
    train_output = Path(args.train_output).expanduser().resolve()
    val_output = Path(args.val_output).expanduser().resolve()

    merged_records = merge_records(input_paths)
    num_records = len(merged_records)
    if num_records < 1:
        raise ValueError("No records found in input json files")

    write_json(all_output, merged_records)

    val_indices = compute_val_indices(num_records=num_records, val_size=args.val_size, seed=args.seed)
    train_rows, val_rows = build_split_records(records=merged_records, val_indices=val_indices)

    write_json(train_output, train_rows)
    write_json(val_output, val_rows)

    summary = {
        "num_total": num_records,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "seed": args.seed,
        "val_size": args.val_size,
        "input_jsons": [str(path) for path in input_paths],
        "all_output": str(all_output),
        "train_output": str(train_output),
        "val_output": str(val_output),
    }
    summary_path = all_output.with_suffix(".summary.json")
    write_json(summary_path, summary)

    print(f"Saved merged json to {all_output}")
    print(f"Saved train split json to {train_output}")
    print(f"Saved val split json to {val_output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()


# python recipe/gui_project/data_process/data_process_1_merge_and_split_gta1_filtered_json.py \
#     --input-jsons \
#         dataset/gta1_grounding/data_process_1_dapo_filtered/sampled_shard_00_filtered.json \
#         dataset/gta1_grounding/data_process_1_dapo_filtered/sampled_shard_01_filtered.json \
#         dataset/gta1_grounding/data_process_1_dapo_filtered/sampled_shard_02_filtered.json \
#         dataset/gta1_grounding/data_process_1_dapo_filtered/sampled_shard_03_filtered.json \
#     --all-output dataset/gta1_grounding/data_process_1_dapo_filtered/all_filtered.json \
#     --train-output dataset/gta1_grounding/data_process_1_dapo_filtered/train_filtered.json \
#     --val-output dataset/gta1_grounding/data_process_1_dapo_filtered/val_filtered.json \
#     --val-size 1000 \
#     --seed 42
