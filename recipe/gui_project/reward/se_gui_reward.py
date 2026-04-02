import math

from recipe.gui_project.model.qwen3vl_utils import extract_point_from_response


def is_point_in_bbox(point: list[float] | None, bbox_xyxy_normalized: dict[str, float]) -> bool:
    if point is None:
        return False
    return (
        bbox_xyxy_normalized["x1"] <= point[0] <= bbox_xyxy_normalized["x2"]
        and bbox_xyxy_normalized["y1"] <= point[1] <= bbox_xyxy_normalized["y2"]
    )


def compute_bbox_center(bbox_xyxy_normalized: dict[str, float]) -> tuple[float, float]:
    center_x = (bbox_xyxy_normalized["x1"] + bbox_xyxy_normalized["x2"]) / 2.0
    center_y = (bbox_xyxy_normalized["y1"] + bbox_xyxy_normalized["y2"]) / 2.0
    return center_x, center_y


def compute_normalized_distance(point: list[float], bbox_xyxy_normalized: dict[str, float]) -> float:
    center_x, center_y = compute_bbox_center(bbox_xyxy_normalized)
    return math.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)


def compute_d_max(bbox_xyxy_normalized: dict[str, float]) -> float:
    center_x, center_y = compute_bbox_center(bbox_xyxy_normalized)
    corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    return max(math.sqrt((corner_x - center_x) ** 2 + (corner_y - center_y) ** 2) for corner_x, corner_y in corners)


def compute_point_reward(point: list[float], bbox_xyxy_normalized: dict[str, float]) -> tuple[float, float, float]:
    d = compute_normalized_distance(point, bbox_xyxy_normalized)
    d_max = compute_d_max(bbox_xyxy_normalized)
    distance_term = max(0.0, 1.0 - d / d_max) ** 2 if d_max > 0 else 1.0

    return distance_term, d, d_max


def compute_attention_metrics(
    image_attention,
    bbox_xyxy_normalized: dict[str, float],
) -> tuple[float, float]:
    if image_attention is None:
        return -1.0, -1.0

    grid_h = len(image_attention)
    if grid_h == 0:
        return -1.0, -1.0
    grid_w = len(image_attention[0])
    if grid_w == 0:
        return -1.0, -1.0

    attention_sum = 0.0
    for row in image_attention:
        for value in row:
            attention_sum += float(value)

    image_token_avg_attention = attention_sum / float(grid_h * grid_w)
    if image_token_avg_attention <= 0:
        return float(attention_sum), -1.0

    x1 = min(max(float(bbox_xyxy_normalized["x1"]), 0.0), 1.0)
    y1 = min(max(float(bbox_xyxy_normalized["y1"]), 0.0), 1.0)
    x2 = min(max(float(bbox_xyxy_normalized["x2"]), 0.0), 1.0)
    y2 = min(max(float(bbox_xyxy_normalized["y2"]), 0.0), 1.0)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x_start = min(int(math.floor(x1 * grid_w)), grid_w - 1)
    x_end = max(x_start + 1, int(math.ceil(x2 * grid_w)))
    x_end = min(x_end, grid_w)

    y_start = min(int(math.floor(y1 * grid_h)), grid_h - 1)
    y_end = max(y_start + 1, int(math.ceil(y2 * grid_h)))
    y_end = min(y_end, grid_h)

    bbox_attention_sum = 0.0
    bbox_token_count = 0
    for y_idx in range(y_start, y_end):
        for x_idx in range(x_start, x_end):
            bbox_attention_sum += float(image_attention[y_idx][x_idx])
            bbox_token_count += 1

    if bbox_token_count == 0:
        return float(attention_sum), -1.0

    bbox_token_avg_attention = bbox_attention_sum / float(bbox_token_count)
    bbox_attention_ratio = bbox_token_avg_attention / image_token_avg_attention
    return float(attention_sum), float(bbox_attention_ratio)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    correct_reward=1.0,
    wrong_reward=0.0,
    invalid_format_reward=0.0,
):
    extra_info = extra_info or {}
    image_attention = extra_info.get("image_attention")
    image_attention_sum, bbox_image_attention_ratio = compute_attention_metrics(image_attention, ground_truth)
    parsed = extract_point_from_response(solution_str)
    point = parsed["point"]

    if point is None:
        return {
            "score": float(invalid_format_reward),
            "format_ok": 0.0,
            "in_box": 0.0,
            "pred_x": -1.0,
            "pred_y": -1.0,
            "distance_to_center": -1.0,
            "distance_max": -1.0,
            "parse_error": parsed["parse_error"] or "unknown",
            "data_source": str(data_source),
            "img_filename": extra_info.get("img_filename", ""),
            "index": str(extra_info.get("index", -1)),
            "has_image_attention": float(image_attention is not None),
            "image_attention_sum": float(image_attention_sum),
            "bbox_image_attention_ratio": float(bbox_image_attention_ratio),
        }

    in_box = is_point_in_bbox(point, ground_truth)
    distance_term, distance_to_center, distance_max = compute_point_reward(point, ground_truth)

    if in_box:
        score = correct_reward + distance_term
    else:
        score = wrong_reward + distance_term

    return {
        "score": float(score),
        "format_ok": 1.0,
        "in_box": float(in_box),
        "pred_x": float(point[0]),
        "pred_y": float(point[1]),
        "distance_to_center": float(distance_to_center),
        "distance_max": float(distance_max),
        "parse_error": "",
        "data_source": str(data_source),
        "img_filename": extra_info.get("img_filename", ""),
        "index": str(extra_info.get("index", -1)),
        "has_image_attention": float(image_attention is not None),
        "image_attention_sum": float(image_attention_sum),
        "bbox_image_attention_ratio": float(bbox_image_attention_ratio),
    }
