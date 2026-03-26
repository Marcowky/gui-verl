from recipe.gui_project.model.qwen3vl_utils import extract_point_from_response


def is_point_in_bbox(point: list[float] | None, bbox_xyxy_normalized: dict[str, float]) -> bool:
    if point is None:
        return False
    return (
        bbox_xyxy_normalized["x1"] <= point[0] <= bbox_xyxy_normalized["x2"]
        and bbox_xyxy_normalized["y1"] <= point[1] <= bbox_xyxy_normalized["y2"]
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    correct_reward=1.0,
    wrong_reward=0.0,
    invalid_format_reward=0.0,
):
    del data_source

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
            "parse_error": parsed["parse_error"] or "unknown",
            "img_filename": extra_info.get("img_filename", ""),
            "index": str(extra_info.get("index", -1)),
        }

    in_box = float(is_point_in_bbox(point, ground_truth))
    score = float(correct_reward if in_box else wrong_reward)

    return {
        "score": score,
        "format_ok": 1.0,
        "in_box": in_box,
        "pred_x": float(point[0]),
        "pred_y": float(point[1]),
        "parse_error": "",
        "img_filename": extra_info.get("img_filename", ""),
        "index": str(extra_info.get("index", -1)),
    }
