from __future__ import annotations

from typing import Any

from .layout_editor import edit_layout_json


def apply_instruction_to_layout(
    *,
    layout_row: dict[str, Any],
    instruction: dict[str, Any],
    target_room_width: int | None = None,
    target_room_height: int | None = None,
) -> dict[str, Any]:
    return edit_layout_json(
        layout_row=layout_row,
        instruction=instruction,
        target_room_width=target_room_width,
        target_room_height=target_room_height,
    )
