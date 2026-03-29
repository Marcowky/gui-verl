from typing import Literal, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
PointMarkerStyle = Literal["solid", "ring_dot"]
MarkerStyle = Literal["box", "point", "ring_dot", "cursor_in_box"]
RGBAColor = Tuple[int, int, int, int]

DEFAULT_MARKER_FILL = (255, 59, 48, 255)
DEFAULT_MARKER_OUTLINE = (255, 255, 255, 255)
DEFAULT_BOX_OUTLINE = (255, 59, 48, 235)


def _bbox_width_height(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return x2 - x1, y2 - y1


def _bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _draw_point_marker(
    draw: ImageDraw.ImageDraw,
    point: Point,
    radius: int,
    marker_style: PointMarkerStyle = "solid",
    fill: RGBAColor = DEFAULT_MARKER_FILL,
    outline: RGBAColor = DEFAULT_MARKER_OUTLINE,
) -> None:
    x, y = point
    if marker_style == "ring_dot":
        outer_radius = max(radius + 4, round(radius * 1.5))
        ring_width = max(2, outer_radius // 8)
        draw.ellipse(
            (x - outer_radius, y - outer_radius, x + outer_radius, y + outer_radius),
            outline=fill,
            width=ring_width,
        )
        dot_radius = max(2, round(radius / 2.8))
        draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=fill,
        )
        return

    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=outline,
        width=max(2, radius // 4),
    )


def _draw_cursor(draw: ImageDraw.ImageDraw, point: Point, scale: int) -> None:
    x, y = point
    arrow = [
        (x, y),
        (x, y + scale * 1.9),
        (x + scale * 0.45, y + scale * 1.45),
        (x + scale * 0.85, y + scale * 2.45),
        (x + scale * 1.3, y + scale * 2.25),
        (x + scale * 0.92, y + scale * 1.25),
        (x + scale * 1.9, y + scale * 1.25),
    ]
    draw.polygon(arrow, fill=(32, 32, 32, 255), outline=(255, 255, 255, 255))


def _load_label_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _draw_marker_label(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    anchor_bbox: BBox,
    label_text: str,
    *,
    label_fill: RGBAColor,
) -> None:
    image_width, image_height = image_size
    min_dim = min(image_width, image_height)
    x1, y1, x2, y2 = anchor_bbox
    font_size = max(12, round(min_dim / 200))
    font = _load_label_font(font_size)
    stroke_width = max(1, round(font_size / 10))
    pad_x = max(4, round(font_size / 4))
    pad_y = max(3, round(font_size / 6))
    margin = max(4, round(min_dim / 400))
    text_bbox = draw.textbbox((0, 0), label_text, font=font, stroke_width=stroke_width)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_width = text_width + pad_x * 2
    label_height = text_height + pad_y * 2

    label_x = min(max(x1, 0.0), max(0.0, image_width - label_width))
    label_y = y1 - label_height - margin
    if label_y < 0:
        label_y = min(max(y2 + margin, 0.0), max(0.0, image_height - label_height))

    draw.text(
        (label_x + pad_x, label_y + pad_y),
        label_text,
        font=font,
        fill=label_fill,
        stroke_width=stroke_width,
        stroke_fill=DEFAULT_MARKER_OUTLINE,
    )


def render_point_marker(
    image: Image.Image,
    marker_point: Point,
    radius: Optional[int] = None,
    label_text: str | None = None,
    marker_style: PointMarkerStyle = "solid",
    fill: RGBAColor = DEFAULT_MARKER_FILL,
    outline: RGBAColor = DEFAULT_MARKER_OUTLINE,
) -> Image.Image:
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    resolved_radius = radius if radius is not None else max(5, round(min(rgba.width, rgba.height) / 300))
    _draw_point_marker(draw, marker_point, resolved_radius, marker_style=marker_style, fill=fill, outline=outline)
    if label_text:
        x, y = marker_point
        label_radius = resolved_radius if marker_style == "solid" else max(resolved_radius + 4, round(resolved_radius * 1.5))
        _draw_marker_label(
            draw,
            rgba.size,
            (x - label_radius, y - label_radius, x + label_radius, y + label_radius),
            label_text,
            label_fill=fill,
        )
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def render_bbox_marker(
    image: Image.Image,
    marker_bbox: BBox,
    marker_style: MarkerStyle = "box",
    label_text: str | None = None,
    box_outline: RGBAColor = DEFAULT_BOX_OUTLINE,
    point_fill: RGBAColor = DEFAULT_MARKER_FILL,
    point_outline: RGBAColor = DEFAULT_MARKER_OUTLINE,
) -> Image.Image:
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    box_w, box_h = _bbox_width_height(marker_bbox)
    line_width = max(2, min(round(min(rgba.width, rgba.height) / 300), round(min(box_w, box_h) / 10)))

    if marker_style in {"box", "cursor_in_box"}:
        draw.rectangle(
            marker_bbox,
            outline=box_outline,
            width=line_width,
        )

    center = _bbox_center(marker_bbox)
    point_radius = max(4, round(min(box_w, box_h) / 6))

    if marker_style == "point":
        _draw_point_marker(draw, center, point_radius, marker_style="solid", fill=point_fill, outline=point_outline)
    elif marker_style == "ring_dot":
        _draw_point_marker(draw, center, point_radius, marker_style="ring_dot", fill=point_fill, outline=point_outline)
    elif marker_style == "cursor_in_box":
        cursor_scale = max(10, round(min(box_w, box_h) / 2.8))
        cursor_anchor = (center[0] - cursor_scale * 0.55, center[1] - cursor_scale * 0.95)
        _draw_cursor(draw, cursor_anchor, cursor_scale)

    if label_text:
        _draw_marker_label(
            draw,
            rgba.size,
            marker_bbox,
            label_text,
            label_fill=box_outline,
        )

    return Image.alpha_composite(rgba, overlay).convert("RGB")
