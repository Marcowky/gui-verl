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
    }
