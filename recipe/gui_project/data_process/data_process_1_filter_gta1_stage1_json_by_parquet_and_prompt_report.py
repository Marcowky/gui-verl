import argparse
import json
from collections import Counter
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
            "Filter a Stage-1 GTA1 json shard using kept rows from a filtered verl parquet "
            "and prompt_report.jsonl metric_values. Rows are matched by "
            "parquet extra_info.index == prompt_report dataset_row_idx == json id. "
            "Only rows with mixed metric_values are kept; all-correct and all-wrong rows are removed."
        )
    )
    parser.add_argument(
        "--input-json",
        default="dataset/gta1_grounding/data_process_1/sampled_shard_00.json",
        help="Stage-1 normalized GTA1 json shard.",
    )
    parser.add_argument(
        "--filtered-parquet",
        default=(
            "outputs/dapo_prefilter/gui_rlvr_dapo/"
            "20260327_164507-qwen3_vl_4b_gta1_prefilter_vllm/filtered_train.parquet"
        ),
        help="Filtered verl parquet produced by the DAPO prefilter step.",
    )
    parser.add_argument(
        "--prompt-report-jsonl",
        default=(
            "outputs/dapo_prefilter/gui_rlvr_dapo/"
            "20260327_164507-qwen3_vl_4b_gta1_prefilter_vllm/prompt_report.jsonl"
        ),
        help="Prompt report jsonl produced by the DAPO prefilter step.",
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


def classify_metric_values(metric_values: list[float | int]) -> str:
    if not metric_values:
        raise ValueError("Encountered empty metric_values")

    normalized_values = {float(value) for value in metric_values}
    if normalized_values == {1.0}:
        return "all_correct"
    if normalized_values == {0.0}:
        return "all_wrong"
    return "mixed"


def load_prompt_report_rows(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            metric_values = row["metric_values"]
            rows.append(
                {
                    "dataset_row_idx": int(row["dataset_row_idx"]),
                    "img_filename": str(row.get("img_filename", "")),
                    "data_source": row.get("data_source"),
                    "ground_truth": row.get("ground_truth"),
                    "metric": row.get("metric"),
                    "metric_values": metric_values,
                    "metric_class": classify_metric_values(metric_values),
                    "kept": bool(row.get("kept", False)),
                    "line_number": line_number,
                }
            )

    return rows


def build_kept_prompt_report_rows(prompt_report_rows: list[dict], parquet_keep_count: int) -> list[dict]:
    kept_rows = [row for row in prompt_report_rows if row["kept"]]

    if len(kept_rows) != parquet_keep_count:
        raise ValueError(
            "Kept prompt report row count does not match filtered parquet row count: "
            f"prompt_report_kept={len(kept_rows)}, parquet_keep_count={parquet_keep_count}"
        )

    return kept_rows


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


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_filtered_rows(
    json_rows: list[dict],
    input_json: Path,
    parquet_keep_rows: list[dict],
    prompt_report_rows: list[dict],
    enable_content_check: bool,
) -> tuple[list[dict], dict]:
    id_to_row = ensure_unique_json_ids(json_rows, json_path=input_json)
    kept_prompt_report_rows = build_kept_prompt_report_rows(
        prompt_report_rows=prompt_report_rows,
        parquet_keep_count=len(parquet_keep_rows),
    )

    missing_json_ids = []
    mismatched_rows = []
    filtered_rows = []
    seen_keep_ids = set()
    parquet_metric_counter = Counter()
    kept_metric_counter = Counter()

    for keep_row, report_row in zip(parquet_keep_rows, kept_prompt_report_rows, strict=True):
        keep_id = keep_row["index"]
        if keep_id in seen_keep_ids:
            raise ValueError(f"Duplicate keep id found in parquet: {keep_id}")
        seen_keep_ids.add(keep_id)

        json_row = id_to_row.get(keep_id)
        if json_row is None:
            missing_json_ids.append(keep_id)
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
            if keep_row["img_filename"] != report_row["img_filename"]:
                mismatches.append(
                    {
                        "field": "img_filename",
                        "parquet": keep_row["img_filename"],
                        "prompt_report": report_row["img_filename"],
                    }
                )

            if mismatches:
                mismatched_rows.append({"id": keep_id, "mismatches": mismatches})
                continue

        metric_class = report_row["metric_class"]
        parquet_metric_counter[metric_class] += 1
        if metric_class == "mixed":
            filtered_rows.append(json_row)
            kept_metric_counter[metric_class] += 1

    if missing_json_ids:
        preview = missing_json_ids[:20]
        raise ValueError(
            f"Parquet ids not found in input json: first_missing={preview}, total_missing={len(missing_json_ids)}"
        )

    if mismatched_rows:
        preview = mismatched_rows[:5]
        raise ValueError(
            "Found parquet/json content mismatches for matched ids: "
            f"first_mismatches={json.dumps(preview, ensure_ascii=False)}, total_mismatches={len(mismatched_rows)}"
        )

    report_metric_counter = Counter(row["metric_class"] for row in prompt_report_rows)
    prompt_report_kept_metric_counter = Counter(row["metric_class"] for row in kept_prompt_report_rows)
    summary = {
        "input_json_count": len(json_rows),
        "prompt_report_count": len(prompt_report_rows),
        "prompt_report_kept_count": len(kept_prompt_report_rows),
        "parquet_keep_count": len(parquet_keep_rows),
        "filtered_json_count": len(filtered_rows),
        "dropped_json_count": len(json_rows) - len(filtered_rows),
        "report_metric_class_counts": dict(report_metric_counter),
        "prompt_report_kept_metric_class_counts": dict(prompt_report_kept_metric_counter),
        "parquet_metric_class_counts": dict(parquet_metric_counter),
        "kept_metric_class_counts": dict(kept_metric_counter),
    }
    return filtered_rows, summary


def main():
    args = parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    filtered_parquet = Path(args.filtered_parquet).expanduser().resolve()
    prompt_report_jsonl = Path(args.prompt_report_jsonl).expanduser().resolve()

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
    prompt_report_rows = load_prompt_report_rows(prompt_report_jsonl)

    filtered_rows, summary = build_filtered_rows(
        json_rows=json_rows,
        input_json=input_json,
        parquet_keep_rows=parquet_keep_rows,
        prompt_report_rows=prompt_report_rows,
        enable_content_check=not args.disable_content_check,
    )

    summary.update(
        {
            "input_json": str(input_json),
            "filtered_parquet": str(filtered_parquet),
            "prompt_report_jsonl": str(prompt_report_jsonl),
            "output_json": str(output_json),
            "content_check_enabled": not args.disable_content_check,
        }
    )

    write_json(output_json, filtered_rows)
    write_json(output_summary, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# python recipe/gui_project/data_process/data_process_1_filter_gta1_stage1_json_by_parquet_and_prompt_report.py \
#   --input-json dataset/gta1_grounding/data_process_1/sampled_shard_00.json \
#   --filtered-parquet outputs/dapo_prefilter/gui_rlvr_dapo/20260327_164507-qwen3_vl_4b_gta1_prefilter_vllm/filtered_train.parquet \
#   --prompt-report-jsonl outputs/dapo_prefilter/gui_rlvr_dapo/20260327_164507-qwen3_vl_4b_gta1_prefilter_vllm/prompt_report.jsonl \
#   --output-json dataset/gta1_grounding/data_process_1_dapo_filtered_without_all_wrong/sampled_shard_00_filtered.json
