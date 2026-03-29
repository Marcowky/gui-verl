from __future__ import annotations

from PIL import Image, ImageDraw

DEFAULT_BOX_OUTLINE = (255, 59, 48, 235)


def _clamp_bbox_xyxy(bbox_xyxy: list[float] | tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]

    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def draw_bbox_on_image(
    image: Image.Image,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    *,
    transparency: float,
    outline_color: tuple[int, int, int, int] = DEFAULT_BOX_OUTLINE,
) -> Image.Image:
    """Draw a red bbox outline on an image.

    `transparency=0.0` means fully opaque and `transparency=1.0` means fully transparent.
    """
    if not 0.0 <= transparency <= 1.0:
        raise ValueError(f"transparency must be in [0.0, 1.0], got {transparency}")

    base_image = image.convert("RGBA")
    left, top, right, bottom = _clamp_bbox_xyxy(bbox_xyxy=bbox_xyxy, image_size=base_image.size)

    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(overlay)

    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    outline_alpha = int(round(outline_color[3] * (1.0 - transparency)))
    outline_width = max(2, min(round(min(base_image.width, base_image.height) / 300), round(min(box_width, box_height) / 10)))

    if outline_alpha <= 0:
        return base_image.convert("RGB")

    drawer.rectangle(
        (left, top, right, bottom),
        outline=(*outline_color[:3], outline_alpha),
        width=outline_width,
    )

    return Image.alpha_composite(base_image, overlay).convert("RGB")
