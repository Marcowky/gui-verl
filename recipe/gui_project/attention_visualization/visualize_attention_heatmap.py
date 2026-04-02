#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from matplotlib import colormaps  # noqa: E402


DEFAULT_JSONL = Path(
    "/home/kaiyu/Project/gui-project/gui-verl/checkpoints/gui_rlvr_baseline/"
    "20260402_232803-qwen3_vl_4b_gta1_filtered_grpo/validation_data/0.jsonl"
)
DEFAULT_IMAGE_ROOT = Path(
    "/home/kaiyu/Project/gui-project/gui-verl/dataset/gta1_grounding/temp_img"
)
DEFAULT_ATTENTION_ROOT = Path(
    "/home/kaiyu/Project/gui-project/gui-verl/dataset/temp/saved_attention"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/kaiyu/Project/gui-project/gui-verl/dataset/temp/attention_visualization"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize layer-averaged image attention as a heatmap overlay and "
            "draw the GT bbox on the same image."
        )
    )
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--attention-root", type=Path, default=DEFAULT_ATTENTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Heatmap overlay strength in [0, 1].",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples.",
    )
    parser.add_argument(
        "--bbox-width",
        type=int,
        default=5,
        help="GT bbox line width in pixels.",
    )
    parser.add_argument(
        "--use-percentile",
        action="store_true",
        help="Normalize attention with [1, 99] percentiles instead of min/max.",
    )
    return parser.parse_args()


def load_records(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON at line {line_no}: {exc}")


def normalize_attention(attention_2d: np.ndarray, use_percentile: bool) -> np.ndarray:
    attention_2d = attention_2d.astype(np.float32)
    if use_percentile:
        lo, hi = np.percentile(attention_2d, [1, 99])
    else:
        lo = float(attention_2d.min())
        hi = float(attention_2d.max())

    if hi <= lo:
        return np.zeros_like(attention_2d, dtype=np.float32)

    clipped = np.clip(attention_2d, lo, hi)
    return (clipped - lo) / (hi - lo)


def create_heatmap_overlay(
    attention_2d: np.ndarray,
    image_size: tuple[int, int],
    alpha: float,
    use_percentile: bool,
) -> Image.Image:
    normalized = normalize_attention(attention_2d, use_percentile=use_percentile)
    heatmap_small = Image.fromarray(np.uint8(normalized * 255), mode="L")
    heatmap_resized = heatmap_small.resize(image_size, resample=Image.Resampling.BILINEAR)
    heatmap_array = np.asarray(heatmap_resized, dtype=np.float32) / 255.0
    heatmap_rgb = colormaps["jet"](heatmap_array)[..., :3]
    heatmap_rgb = np.uint8(np.clip(heatmap_rgb * 255.0, 0, 255))
    overlay = np.dstack(
        [heatmap_rgb, np.uint8(np.clip(heatmap_array * alpha * 255.0, 0, 255))]
    )
    return Image.fromarray(overlay, mode="RGBA")


def draw_bbox(image: Image.Image, bbox: dict, line_width: int) -> None:
    width, height = image.size
    x1 = int(round(float(bbox["x1"]) * width))
    x2 = int(round(float(bbox["x2"]) * width))
    y1 = int(round(float(bbox["y1"]) * height))
    y2 = int(round(float(bbox["y2"]) * height))
    draw = ImageDraw.Draw(image)
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=line_width)


def draw_predicted_click(image: Image.Image, pred_x: float, pred_y: float) -> None:
    width, height = image.size
    x = int(round(float(pred_x) * width))
    y = int(round(float(pred_y) * height))
    radius = max(8, min(width, height) // 80)
    cross = max(10, radius + 4)

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 255, 255),
        width=3,
    )
    draw.ellipse(
        (x - radius + 2, y - radius + 2, x + radius - 2, y + radius - 2),
        fill=(255, 0, 0),
        outline=(255, 0, 0),
    )
    draw.line((x - cross, y, x + cross, y), fill=(255, 255, 255), width=3)
    draw.line((x, y - cross, x, y + cross), fill=(255, 255, 255), width=3)


def visualize_record(
    record: dict,
    image_root: Path,
    attention_root: Path,
    output_root: Path,
    alpha: float,
    bbox_width: int,
    use_percentile: bool,
) -> Path:
    rel_image_path = Path(record["img_filename"])
    image_path = image_root / rel_image_path
    attention_path = attention_root / f"{record['index']}.npy"
    output_path = output_root / f"{record['index']}.png"

    with Image.open(image_path) as image:
        image = image.convert("RGBA")
        attention = np.load(attention_path)

        if attention.ndim == 3:
            attention_2d = attention.mean(axis=0)
        elif attention.ndim == 2:
            attention_2d = attention
        else:
            raise ValueError(
                f"Unsupported attention shape for {attention_path}: {attention.shape}"
            )

        overlay = create_heatmap_overlay(
            attention_2d,
            image.size,
            alpha=alpha,
            use_percentile=use_percentile,
        )
        composed = Image.alpha_composite(image, overlay).convert("RGB")
        draw_bbox(composed, record["gts"], line_width=bbox_width)
        draw_predicted_click(
            composed,
            pred_x=record["pred_x"],
            pred_y=record["pred_y"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)
    return output_path


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    failures = 0

    for line_no, record in load_records(args.jsonl):
        if args.limit is not None and processed >= args.limit:
            break

        try:
            output_path = visualize_record(
                record=record,
                image_root=args.image_root,
                attention_root=args.attention_root,
                output_root=args.output_root,
                alpha=args.alpha,
                bbox_width=args.bbox_width,
                use_percentile=args.use_percentile,
            )
            processed += 1
            if processed <= 3 or processed % 20 == 0:
                print(f"[{processed}] saved {output_path}")
        except Exception as exc:
            failures += 1
            print(
                f"Failed at jsonl line {line_no}, index {record.get('index')}: {exc}"
            )

    print(
        f"Done. Processed {processed} samples. Failed {failures}. "
        f"Output root: {args.output_root}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
