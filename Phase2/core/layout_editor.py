from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


ROOM_PATTERN = re.compile(r"(\d+)\s*x\s*(\d+)")


def _normalize_name(name: str) -> str:
    return name.strip().lstrip(".").lower()


def _parse_room_dims(layout_row: dict[str, Any]) -> tuple[int, int] | None:
    room = layout_row.get("room")
    if isinstance(room, dict):
        w = room.get("width")
        h = room.get("height")
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            return w, h
    prompt = str(layout_row.get("prompt", ""))
    match = ROOM_PATTERN.search(prompt)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _set_room_dims(layout_row: dict[str, Any], width: int, height: int) -> None:
    layout_row["room"] = {"width": width, "height": height}
    prompt = str(layout_row.get("prompt", ""))
    if prompt:
        if ROOM_PATTERN.search(prompt):
            layout_row["prompt"] = ROOM_PATTERN.sub(f"{width}x{height}", prompt, count=1)
        else:
            layout_row["prompt"] = f"{prompt} Room size: {width}x{height}."
    query_id = str(layout_row.get("query_id", "layout"))
    layout_row["query_id"] = ROOM_PATTERN.sub(f"{width}x{height}", query_id) if ROOM_PATTERN.search(query_id) else query_id


def _shift_box(box: list[float], direction: str, delta: float = 0.03) -> list[float]:
    if len(box) != 4:
        return box
    x1, y1, x2, y2 = box
    d = direction.lower()
    if "left" in d:
        x1 -= delta
        x2 -= delta
    elif "right" in d:
        x1 += delta
        x2 += delta
    elif "up" in d or "north" in d:
        y1 -= delta
        y2 -= delta
    elif "down" in d or "south" in d:
        y1 += delta
        y2 += delta
    elif "away" in d:
        x1 += delta
        x2 += delta
        y1 += delta
        y2 += delta
    return [x1, y1, x2, y2]


def _find_object_index(layout_row: dict[str, Any], target: str) -> int | None:
    objects = layout_row.get("object_list", [])
    for index, obj in enumerate(objects):
        if not isinstance(obj, list) or len(obj) != 2:
            continue
        if _normalize_name(str(obj[0])) == _normalize_name(target):
            return index
    return None


def _apply_single_edit(layout_row: dict[str, Any], edit: dict[str, Any]) -> None:
    edit_type = str(edit.get("type", "")).lower()
    objects = layout_row.get("object_list", [])

    if edit_type == "resize_room":
        width = int(edit.get("width", 0))
        height = int(edit.get("height", 0))
        if width > 0 and height > 0:
            old_dims = _parse_room_dims(layout_row)
            _set_room_dims(layout_row, width=width, height=height)
            # Scale absolute-pixel boxes if dimensions are known and coordinates look absolute.
            if old_dims and old_dims[0] > 0 and old_dims[1] > 0:
                old_w, old_h = old_dims
                x_scale = width / old_w
                y_scale = height / old_h
                for i, obj in enumerate(objects):
                    if isinstance(obj, list) and len(obj) == 2 and isinstance(obj[1], list) and len(obj[1]) == 4:
                        box = obj[1]
                        if any(abs(v) > 1.5 for v in box):
                            objects[i][1] = [box[0] * x_scale, box[1] * y_scale, box[2] * x_scale, box[3] * y_scale]
        layout_row["object_list"] = objects
        return

    if edit_type == "move":
        target = str(edit.get("target", ""))
        idx = _find_object_index(layout_row, target) if target else None
        if idx is None:
            return
        box = objects[idx][1]
        if not isinstance(box, list):
            return
        if "dx" in edit or "dy" in edit:
            dx = float(edit.get("dx", 0))
            dy = float(edit.get("dy", 0))
            if len(box) == 4:
                objects[idx][1] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
        else:
            direction = str(edit.get("direction", "right"))
            delta = float(edit.get("delta", 0.03))
            objects[idx][1] = _shift_box(box, direction=direction, delta=delta)
        layout_row["object_list"] = objects
        return

    if edit_type == "set_box":
        target = str(edit.get("target", ""))
        idx = _find_object_index(layout_row, target) if target else None
        new_box = edit.get("box")
        if idx is not None and isinstance(new_box, list) and len(new_box) == 4:
            objects[idx][1] = [float(v) for v in new_box]
            layout_row["object_list"] = objects


def edit_layout_json(
    *,
    layout_row: dict[str, Any],
    instruction: dict[str, Any] | None = None,
    target_room_width: int | None = None,
    target_room_height: int | None = None,
) -> dict[str, Any]:
    updated = deepcopy(layout_row)

    edits: list[dict[str, Any]] = []
    if target_room_width and target_room_height:
        edits.append({"type": "resize_room", "width": target_room_width, "height": target_room_height})
    if instruction:
        if isinstance(instruction.get("edits"), list):
            edits.extend([e for e in instruction["edits"] if isinstance(e, dict)])
        else:
            # Backward-compatible single-instruction format.
            edits.append(
                {
                    "type": str(instruction.get("action", "move")),
                    "target": instruction.get("target", "furniture"),
                    "direction": instruction.get("direction") or instruction.get("directive") or "right",
                }
            )

    for edit in edits:
        _apply_single_edit(updated, edit)
    return updated
