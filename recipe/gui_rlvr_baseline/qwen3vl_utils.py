import json
import re


SYSTEM_PROMPT = """You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. 
* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def build_prompt_messages(instruction: str) -> list[dict]:
    instruction = instruction.strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>{instruction}"},
    ]


def normalize_bbox_xyxy(bbox_xyxy: list[float], image_size: tuple[int, int]) -> dict[str, float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return {
        "x1": x1 / width,
        "y1": y1 / height,
        "x2": x2 / width,
        "y2": y2 / height,
    }


def extract_point_from_response(response: str) -> dict:
    response = response or ""
    tool_call_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL)

    action_payload = None
    parse_error = None

    for block in reversed(tool_call_blocks):
        try:
            action_payload = json.loads(block)
            break
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)

    if action_payload is None:
        return {
            "point": None,
            "coordinates": None,
            "parse_error": parse_error or "missing_tool_call",
        }

    try:
        coordinates = action_payload["arguments"]["coordinate"]
        if len(coordinates) == 2:
            point_x, point_y = float(coordinates[0]), float(coordinates[1])
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = [float(value) for value in coordinates]
            point_x = (x1 + x2) / 2.0
            point_y = (y1 + y2) / 2.0
        else:
            raise ValueError(f"unexpected coordinate length: {len(coordinates)}")
    except Exception as exc:  # noqa: BLE001
        return {
            "point": None,
            "coordinates": None,
            "parse_error": str(exc),
        }

    return {
        "point": [point_x / 1000.0, point_y / 1000.0],
        "coordinates": coordinates,
        "parse_error": None,
    }