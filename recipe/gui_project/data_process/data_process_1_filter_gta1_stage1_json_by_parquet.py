import argparse
import json
from collections.abc import Iterable
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "pyarrow is required to read filtered parquet files. "
        "Please run this script in an environment that has pyarrow installed."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter a Stage-1 GTA1 json shard using kept rows from a filtered verl parquet. "
            "Rows are matched by parquet extra_info.index == json id."
        )
    )
    parser.add_argument(
        "--input-json",
        default="dataset/gta1_grounding/data_process_1/sampled_shard_02.json",
        help="Stage-1 normalized GTA1 json shard.",
    )
    parser.add_argument(
        "--filtered-parquet",
        default=(
            "outputs/dapo_prefilter/gui_rlvr_dapo/"
            "20260328_073650-qwen3_vl_4b_gta1_prefilter_vllm_02/filtered_train.parquet"
        ),
        help="Filtered verl parquet produced by the DAPO prefilter step.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output filtered json path. Defaults to <input-json stem>.filtered.json next to the input json.",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Optional summary json path. Defaults to <output-json>.summary.json.",
    )
    parser.add_argument(
        "--disable-content-check",
        action="store_true",
        help="Only match by id and skip instruction/image consistency checks.",
    )
    return parser.parse_args()


def load_json_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise TypeError(f"Expected json array in {path}, got {type(rows).__name__}")
    return rows


def iter_parquet_keep_rows(path: Path) -> Iterable[dict]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=["extra_info"], batch_size=4096):
        for extra_info in batch.column(0).to_pylist():
            if extra_info is None:
                raise ValueError("Encountered parquet row with missing extra_info")
            yield {
                "index": int(extra_info["index"]),
                "instruction": str(extra_info.get("instruction", "")),
                "img_filename": str(extra_info.get("img_filename", "")),
            }


def ensure_unique_json_ids(rows: list[dict], json_path: Path) -> dict[int, dict]:
    id_to_row = {}
    duplicate_ids = []

    for row in rows:
        row_id = int(row["id"])
        if row_id in id_to_row:
            duplicate_ids.append(row_id)
            continue
        id_to_row[row_id] = row

    if duplicate_ids:
        preview = duplicate_ids[:10]
        raise ValueError(
            f"Found duplicate ids in {json_path}: first duplicates={preview}, total_duplicates={len(duplicate_ids)}"
        )

    return id_to_row


def build_filtered_rows(
    json_rows: list[dict],
    input_json: Path,
    parquet_keep_rows: list[dict],
    enable_content_check: bool,
) -> tuple[list[dict], dict]:
    id_to_row = ensure_unique_json_ids(json_rows, json_path=input_json)

    missing_ids = []
    mismatched_rows = []
    filtered_rows = []
    seen_keep_ids = set()

    for keep_row in parquet_keep_rows:
        keep_id = keep_row["index"]
        if keep_id in seen_keep_ids:
            raise ValueError(f"Duplicate keep id found in parquet: {keep_id}")
        seen_keep_ids.add(keep_id)

        json_row = id_to_row.get(keep_id)
        if json_row is None:
            missing_ids.append(keep_id)
            continue

        if enable_content_check:
            mismatches = []
            if str(json_row["instruction"]) != keep_row["instruction"]:
                mismatches.append(
                    {
                        "field": "instruction",
                        "json": str(json_row["instruction"]),
                        "parquet": keep_row["instruction"],
                    }
                )
            if str(json_row["image"]) != keep_row["img_filename"]:
                mismatches.append(
                    {
                        "field": "image",
                        "json": str(json_row["image"]),
                        "parquet": keep_row["img_filename"],
                    }
                )

            if mismatches:
                mismatched_rows.append({"id": keep_id, "mismatches": mismatches})
                continue

        filtered_rows.append(json_row)

    if missing_ids:
        preview = missing_ids[:20]
        raise ValueError(f"Parquet ids not found in input json: first_missing={preview}, total_missing={len(missing_ids)}")

    if mismatched_rows:
        preview = mismatched_rows[:5]
        raise ValueError(
            "Found parquet/json content mismatches for matched ids: "
            f"first_mismatches={json.dumps(preview, ensure_ascii=False)}, total_mismatches={len(mismatched_rows)}"
        )

    summary = {
        "input_json_count": len(json_rows),
        "parquet_keep_count": len(parquet_keep_rows),
        "filtered_json_count": len(filtered_rows),
        "dropped_json_count": len(json_rows) - len(filtered_rows),
    }
    return filtered_rows, summary


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    args = parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    filtered_parquet = Path(args.filtered_parquet).expanduser().resolve()

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
    else:
        output_json = input_json.with_name(f"{input_json.stem}.filtered.json")

    if args.output_summary:
        output_summary = Path(args.output_summary).expanduser().resolve()
    else:
        output_summary = output_json.with_name(f"{output_json.stem}.summary.json")

    json_rows = load_json_rows(input_json)
    parquet_keep_rows = list(iter_parquet_keep_rows(filtered_parquet))

    filtered_rows, summary = build_filtered_rows(
        json_rows=json_rows,
        input_json=input_json,
        parquet_keep_rows=parquet_keep_rows,
        enable_content_check=not args.disable_content_check,
    )

    summary.update(
        {
            "input_json": str(input_json),
            "filtered_parquet": str(filtered_parquet),
            "output_json": str(output_json),
            "content_check_enabled": not args.disable_content_check,
        }
    )

    write_json(output_json, filtered_rows)
    write_json(output_summary, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# python recipe/gui_project/data_process/filter_gta1_stage1_json_by_parquet.py \
#   --input-json dataset/gta1_grounding/data_process_1/sampled_shard_00.json \
#   --filtered-parquet outputs/dapo_prefilter/gui_rlvr_dapo/20260327_164507-qwen3_vl_4b_gta1_prefilter_vllm/filtered_train.parquet \
#   --output-json dataset/gta1_grounding/data_process_1_filtered/sampled_shard_00_filtered.json
