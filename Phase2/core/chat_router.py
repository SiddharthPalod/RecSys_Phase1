from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from ..config import get_ollama_model
from .recommender import recommend_layout


class RouterDecision(BaseModel):
    action: str
    reason: str
    instruction: dict[str, Any] | None = None


@dataclass
class SessionState:
    current_layout_id: str | None = None
    current_layout: dict[str, Any] | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    known_preferences: dict[str, str] = field(default_factory=dict)
    temporary_layouts: list[dict[str, Any]] = field(default_factory=list)


SESSION_STORE: dict[str, SessionState] = {}


def start_session(session_id: str | None = None) -> str:
    resolved = session_id or str(uuid4())
    SESSION_STORE.setdefault(resolved, SessionState())
    return resolved


def reset_session(session_id: str) -> bool:
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        return True
    return False


def set_current_layout(session_id: str, recommended_id: str, layout: dict[str, Any]) -> None:
    state = SESSION_STORE.setdefault(session_id, SessionState())
    state.current_layout_id = recommended_id
    state.current_layout = layout


def _extract_preferences(user_message: str) -> dict[str, str]:
    lowered = user_message.lower()
    extracted: dict[str, str] = {}
    if "spacious" in lowered or "open space" in lowered:
        extracted["space"] = "more_open"
    if "minimal" in lowered:
        extracted["style"] = "minimal"
    if "cozy" in lowered:
        extracted["style"] = "cozy"
    if "door" in lowered:
        extracted["door_constraint"] = "mentioned"
    if "privacy" in lowered:
        extracted["privacy"] = "high"
    if "bright" in lowered or "light" in lowered:
        extracted["lighting"] = "bright"
    if "work" in lowered or "study" in lowered:
        extracted["primary_activity"] = "work"
    if "sleep" in lowered or "rest" in lowered:
        extracted["primary_activity"] = "rest"
    if "wall" in lowered:
        extracted["placement_bias"] = "walls"
    return extracted


def _confidence_score(state: SessionState) -> float:
    history_depth = min(len(state.history), 6)
    pref_depth = min(len(state.known_preferences), 7)
    has_current_layout = 1 if state.current_layout_id else 0
    score = 0.25 * has_current_layout + 0.08 * history_depth + 0.12 * pref_depth
    return max(0.0, min(1.0, score))


def _fallback_clarifying_question(state: SessionState) -> str:
    # Ask progressively deeper questions across style, function, constraints.
    if "primary_activity" not in state.known_preferences:
        return "What is the main use of this room right now: sleep, work, relaxation, or mixed?"
    if "style" not in state.known_preferences:
        return "Which aesthetic should dominate: minimal, cozy, modern, or balanced?"
    if "space" not in state.known_preferences:
        return "For circulation, should we prioritize central walk space or wall-aligned furniture?"
    if "privacy" not in state.known_preferences:
        return "Do you want stronger privacy zoning (bed away from entrance sightline), or is openness fine?"
    if "lighting" not in state.known_preferences:
        return "Should task furniture (desk/chair) be closer to brighter areas, or is lighting not a priority?"
    return "Which exact object and direction should be optimized next (for example: move bed north, chair near wall)?"


def _llm_clarifying_question(state: SessionState) -> str | None:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama
    except ImportError:
        return None

    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in state.history[-10:])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a CRS for interior layout design. "
                    "Ask exactly ONE concise, high-information follow-up question that best reduces uncertainty "
                    "for recommending or modifying a room layout. Return plain text only."
                ),
            ),
            (
                "human",
                (
                    "Known preferences: {prefs}\n"
                    "Current layout id: {layout_id}\n"
                    "Recent dialogue:\n{history}\n\n"
                    "Generate the next best question."
                ),
            ),
        ]
    )
    model = ChatOllama(model=get_ollama_model(), temperature=0.2)
    chain = prompt | model
    try:
        response = chain.invoke(
            {
                "prefs": str(state.known_preferences),
                "layout_id": state.current_layout_id or "none",
                "history": history_text or "none",
            }
        )
    except Exception:
        return None
    question = str(response.content).strip()
    if not question:
        return None
    return question


def _clarifying_question(state: SessionState) -> str:
    return _llm_clarifying_question(state) or _fallback_clarifying_question(state)


def _fallback_route(user_message: str) -> RouterDecision:
    text = user_message.lower()
    modify_keywords = ["move", "shift", "rotate", "swap", "change bed", "change chair", "change tv", "away", "closer"]
    if any(k in text for k in modify_keywords):
        target = "furniture"
        if "bed" in text:
            target = "bed"
        elif "chair" in text:
            target = "chair"
        elif "tv" in text:
            target = "tv-stand"
        return RouterDecision(
            action="modify",
            reason="User requested a direct furniture adjustment on current layout.",
            instruction={"target": target, "action": "move", "directive": user_message},
        )
    if "not sure" in text or "help me decide" in text:
        return RouterDecision(action="ask_clarification", reason="Low-confidence preference signal; ask a targeted follow-up.")
    return RouterDecision(action="switch", reason="User feedback suggests selecting a different candidate layout.")


def _route_with_langchain(user_message: str, state: SessionState) -> RouterDecision | None:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama
    except ImportError:
        return None

    model = ChatOllama(model=get_ollama_model(), temperature=0, format="json")
    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in state.history[-8:])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a routing agent for conversational interior layout refinement.\n"
                    "Return strict JSON only with keys: action, reason, instruction.\n"
                    "action must be either 'switch', 'modify', or 'ask_clarification'.\n"
                    "Use 'modify' if user asks explicit furniture changes in current layout.\n"
                    "Use 'switch' if user asks for a different overall layout recommendation.\n"
                    "Use 'ask_clarification' if details are insufficient for either action."
                ),
            ),
            ("human", "Current layout id: {layout_id}\nRecent history:\n{history}\n\nLatest user message:\n{message}"),
        ]
    )
    chain = prompt | model
    try:
        response = chain.invoke(
            {"layout_id": state.current_layout_id or "none", "history": history_text or "no previous history", "message": user_message}
        )
    except Exception:
        return None
    content = str(response.content).strip()
    if not content.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            content = match.group(0)
    try:
        parsed = RouterDecision.model_validate_json(content)
    except (ValidationError, ValueError):
        try:
            parsed = RouterDecision.model_validate(json.loads(content))
        except Exception:
            return None
    if parsed.action not in {"switch", "modify", "ask_clarification"}:
        return None
    return parsed


def route_chat_turn(
    *,
    filtered_data: dict[str, Any],
    session_id: str,
    user_message: str,
    max_candidates: int,
) -> dict[str, Any]:
    state = SESSION_STORE.setdefault(session_id, SessionState())
    state.history.append({"role": "user", "content": user_message})
    state.known_preferences.update(_extract_preferences(user_message))

    confidence = _confidence_score(state)
    if confidence < 0.45:
        question = _clarifying_question(state)
        state.history.append({"role": "assistant", "content": question})
        return {
            "session_id": session_id,
            "action": "ask_clarification",
            "message": question,
            "recommended_id": state.current_layout_id,
            "recommended_layout": state.current_layout,
            "instruction": None,
            "confidence": confidence,
        }

    decision = _route_with_langchain(user_message=user_message, state=state) or _fallback_route(user_message=user_message)

    if decision.action == "ask_clarification":
        question = _clarifying_question(state)
        state.history.append({"role": "assistant", "content": question})
        return {
            "session_id": session_id,
            "action": "ask_clarification",
            "message": question,
            "recommended_id": state.current_layout_id,
            "recommended_layout": state.current_layout,
            "instruction": None,
            "confidence": confidence,
        }

    if decision.action == "switch":
        recommendation = recommend_layout(filtered_data=filtered_data, user_request=user_message, max_candidates=max_candidates)
        if recommendation is None:
            return {
                "session_id": session_id,
                "action": "switch",
                "message": "Could not find a replacement layout right now.",
                "recommended_id": None,
                "recommended_layout": None,
                "instruction": None,
                "confidence": confidence,
            }
        state.current_layout_id = recommendation["recommended_id"]
        state.current_layout = recommendation["recommended_layout"]
        state.history.append({"role": "assistant", "content": recommendation["pitch"]})
        return {
            "session_id": session_id,
            "action": "switch",
            "message": recommendation["pitch"],
            "recommended_id": recommendation["recommended_id"],
            "recommended_layout": recommendation["recommended_layout"],
            "instruction": None,
            "confidence": confidence,
        }

    instruction = decision.instruction or {"target": "furniture", "action": "move", "directive": user_message}
    assistant_message = "I understood this as a modify request on the current layout. Apply these instructions in your regeneration step."
    state.history.append({"role": "assistant", "content": assistant_message})
    return {
        "session_id": session_id,
        "action": "modify",
        "message": assistant_message,
        "recommended_id": state.current_layout_id,
        "recommended_layout": state.current_layout,
        "instruction": instruction,
        "confidence": confidence,
    }


def store_regenerated_layout(
    *,
    session_id: str,
    generated_layout: dict[str, Any],
    source_layout_id: str | None,
    instruction: dict[str, Any],
) -> dict[str, Any]:
    state = SESSION_STORE.setdefault(session_id, SessionState())
    suffix = len(state.temporary_layouts) + 1
    new_id = f"{source_layout_id or 'session_layout'}:regen_{suffix}"
    wrapped = {
        **generated_layout,
        "generated_from": source_layout_id,
        "regeneration_instruction": instruction,
        "session_temp_id": new_id,
    }
    state.temporary_layouts.append(wrapped)
    state.current_layout_id = new_id
    state.current_layout = wrapped
    state.history.append({"role": "assistant", "content": "How does this look? I applied your requested modification."})
    return {
        "recommended_id": new_id,
        "recommended_layout": wrapped,
        "message": "How does this look? I applied your requested modification.",
    }
