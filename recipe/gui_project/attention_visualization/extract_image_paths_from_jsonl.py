#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


IMAGE_PATH_RE = re.compile(
    r"(?i)^.*?\.(png|jpg|jpeg|webp|bmp|gif|tiff|svg)(?:\?[^#\s]*)?(?:#[^\s]*)?$"
)


def looks_like_image_path(value: str) -> bool:
    return bool(IMAGE_PATH_RE.match(value.strip()))


def walk_values(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk_values(value)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from walk_values(item)
        return
    if isinstance(obj, str):
        yield obj


def extract_from_field(obj: Any, field: str):
    if isinstance(obj, dict):
        if field in obj and isinstance(obj[field], str):
            yield obj[field]
        for value in obj.values():
            yield from extract_from_field(value, field)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from extract_from_field(item, field)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract image paths from a JSONL file."
    )
    parser.add_argument("jsonl_path", type=Path, help="Path to the input JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write results to this file instead of stdout",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. 'json' writes a JSON array.",
    )
    parser.add_argument(
        "--field",
        help="Only extract values from this field name, for example: img_filename",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate paths instead of deduplicating them",
    )
    args = parser.parse_args()

    if not args.jsonl_path.is_file():
        print(f"File not found: {args.jsonl_path}", file=sys.stderr)
        return 1

    seen = set()
    extracted_count = 0
    results = []

    with args.jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Skip invalid JSON at line {line_no}: {exc}",
                    file=sys.stderr,
                )
                continue

            values = (
                extract_from_field(obj, args.field)
                if args.field
                else walk_values(obj)
            )

            for value in values:
                if not looks_like_image_path(value):
                    continue
                extracted_count += 1
                if args.keep_duplicates:
                    results.append(value)
                    continue
                if value not in seen:
                    seen.add(value)
                    results.append(value)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(results, f, ensure_ascii=False, indent=2)
                f.write("\n")
            else:
                for item in results:
                    f.write(f"{item}\n")
    else:
        if args.format == "json":
            json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            for item in results:
                print(item)

    if not args.keep_duplicates:
        print(
            f"Extracted {len(seen)} unique image paths "
            f"({extracted_count} matches before dedup).",
            file=sys.stderr,
        )
    else:
        print(f"Extracted {extracted_count} image paths.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
