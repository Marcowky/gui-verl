import argparse
import json
import os
import datasets
from pathlib import Path

from recipe.gui_rlvr_baseline.qwen3vl_utils import build_prompt_messages, normalize_bbox_xyxy


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: convert one normalized json + img folder pair to verl parquet.")
    parser.add_argument(
        "--img-folder",
        required=True,
        help="Image folder root. The json file should reference images relative to this folder.",
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help="Human-readable normalized json file from stage 1.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name written into the verl parquet as data_source.",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional cap for debugging.")
    return parser.parse_args()


def load_json(path: Path, max_samples: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if max_samples > 0:
        records = records[: max_samples]
    return records


def build_verl_row(record: dict, img_folder: Path, dataset_name: str) -> dict:
    image_rel_path = record["image"]
    image_abs_path = img_folder / image_rel_path
    image_bytes = image_abs_path.read_bytes()

    image_size = tuple(record["image_size"])
    bbox_normalized = normalize_bbox_xyxy(record["bbox_xyxy"], image_size=image_size)

    return {
        "data_source": dataset_name,
        "prompt": build_prompt_messages(record["instruction"]),
        "images": [{"bytes": image_bytes, "path": image_rel_path}],
        "ability": "gui_grounding",
        "reward_model": {
            "style": "rule",
            "ground_truth": bbox_normalized,
        },
        "extra_info": {
            "split": record["split"],
            "index": record["id"],
            "instruction": record["instruction"],
            "img_filename": image_rel_path,
            "gt_bbox_pixels": record["bbox_xyxy"],
            "image_size": {"width": image_size[0], "height": image_size[1]},
            # "meta": record.get("meta", {}),
            # "source": record.get("source", {}),
        },
    }


def generate_rows(records: tuple[dict, ...], img_folder: str, dataset_name: str):
    img_folder_path = Path(img_folder)
    for record in records:
        yield build_verl_row(record=record, img_folder=img_folder_path, dataset_name=dataset_name)


def build_dataset(records: list[dict], img_folder: Path, dataset_name: str) -> datasets.Dataset:
    return datasets.Dataset.from_generator(
        generate_rows,
        gen_kwargs={
            "records": tuple(records),
            "img_folder": str(img_folder),
            "dataset_name": dataset_name,
        },
    )


def main():
    args = parse_args()

    img_folder = Path(args.img_folder).expanduser().resolve()
    json_path = Path(args.json_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_DATASETS_CACHE", str(output_path.parent / ".hf_cache"))

    records = load_json(json_path, max_samples=args.max_samples)

    dataset = build_dataset(records=records, img_folder=img_folder, dataset_name=args.dataset_name)
    dataset.to_parquet(str(output_path))

    summary = {
        "dataset_name": args.dataset_name,
        "json_path": str(json_path),
        "img_folder": str(img_folder),
        "num_rows": len(dataset),
        "parquet_path": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved parquet to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
