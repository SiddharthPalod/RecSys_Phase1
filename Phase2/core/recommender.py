from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from ..config import get_ollama_model


class LLMSelection(BaseModel):
    recommended_id: str
    pitch: str


@dataclass
class LayoutCandidate:
    layout_id: str
    description: str
    raw_layout: dict[str, Any]


def _normalize_item_name(item_name: str) -> str:
    return item_name.strip().lstrip(".").replace("-", " ")


def _layout_description(row: dict[str, Any], room: dict[str, Any]) -> str:
    room_width = room.get("width", "unknown")
    room_height = room.get("height", "unknown")
    iter_id = row.get("iter", "unknown")
    objects = row.get("object_list", [])
    normalized = []
    for obj in objects:
        if not isinstance(obj, list) or len(obj) != 2:
            continue
        label = _normalize_item_name(str(obj[0]))
        box = obj[1]
        normalized.append(f"{label} at {box}")
    object_text = ", ".join(normalized) if normalized else "no objects"
    return f"Layout {iter_id}: room {room_width}x{room_height}, objects -> {object_text}."


def _build_candidates(filtered_data: dict[str, Any], max_candidates: int) -> list[LayoutCandidate]:
    room = filtered_data.get("meta", {}).get("room", {})
    rows = filtered_data.get("rows", [])
    candidates: list[LayoutCandidate] = []
    for row in rows[:max_candidates]:
        iter_value = str(row.get("iter", "unknown"))
        query_id = row.get("query_id", "layout")
        layout_id = f"{query_id}:{iter_value}"
        candidates.append(
            LayoutCandidate(
                layout_id=layout_id,
                description=_layout_description(row=row, room=room),
                raw_layout=row,
            )
        )
    return candidates


def _fallback_rank(candidates: list[LayoutCandidate], user_request: str) -> LLMSelection:
    request = user_request.lower()
    best = candidates[0]
    best_score = -1
    for candidate in candidates:
        score = 0
        desc = candidate.description.lower()
        for token in request.split():
            if token in desc:
                score += 1
        if score > best_score:
            best = candidate
            best_score = score
    return LLMSelection(
        recommended_id=best.layout_id,
        pitch="Selected this layout using fallback lexical ranking because LLM output was unavailable.",
    )


def _recommend_with_langchain(candidates: list[LayoutCandidate], user_request: str) -> LLMSelection | None:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama
    except ImportError:
        return None

    model = ChatOllama(model=get_ollama_model(), temperature=0, format="json")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an interior layout recommender. "
                    "Choose one layout id from the candidates and explain briefly. "
                    "Return only strict JSON with keys: recommended_id, pitch."
                ),
            ),
            (
                "human",
                (
                    "User preference:\n{user_request}\n\n"
                    "Candidate layouts:\n{candidate_block}\n\n"
                    "Respond as JSON only."
                ),
            ),
        ]
    )
    candidate_block = "\n".join(
        f"- id={candidate.layout_id} | {candidate.description}" for candidate in candidates
    )
    chain = prompt | model
    try:
        response = chain.invoke({"user_request": user_request, "candidate_block": candidate_block})
    except Exception:
        return None
    content = str(response.content).strip()
    if not content.startswith("{"):
        # Some local models wrap JSON in text/code-fences; salvage first object.
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            content = match.group(0)
    try:
        parsed = LLMSelection.model_validate_json(content)
    except (ValidationError, ValueError):
        # Last attempt: decode then validate to tolerate minor quoting quirks.
        try:
            parsed = LLMSelection.model_validate(json.loads(content))
        except Exception:
            return None
    valid_ids = {candidate.layout_id for candidate in candidates}
    if parsed.recommended_id not in valid_ids:
        return None
    return parsed


def recommend_layout(
    filtered_data: dict[str, Any],
    user_request: str,
    max_candidates: int = 10,
) -> dict[str, Any] | None:
    candidates = _build_candidates(filtered_data=filtered_data, max_candidates=max_candidates)
    if not candidates:
        return None

    selection = _recommend_with_langchain(candidates=candidates, user_request=user_request)
    if selection is None:
        selection = _fallback_rank(candidates=candidates, user_request=user_request)

    chosen = next((c for c in candidates if c.layout_id == selection.recommended_id), None)
    if chosen is None:
        chosen = candidates[0]
        selection = LLMSelection(
            recommended_id=chosen.layout_id,
            pitch="Fallback default selected because recommended id was not found.",
        )

    return {
        "recommended_id": selection.recommended_id,
        "pitch": selection.pitch,
        "recommended_layout": chosen.raw_layout,
    }
