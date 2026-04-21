from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from uuid import uuid4

from ..config import get_ollama_model
from ..core import (
    apply_instruction_to_layout,
    recommend_layout,
    route_chat_turn,
    set_current_layout,
    store_regenerated_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 conversational CRS CLI")
    parser.add_argument("--filter-path", type=Path, default=Path("Phase1/filter.json"))
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument("--max-questions", type=int, default=8)
    return parser.parse_args()


def ask(prompt: str) -> str:
    return input(prompt).strip()


def load_filtered(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_layout_summary(layout_row: dict | None) -> None:
    if not layout_row:
        print("No layout selected yet.")
        return
    print("Current layout:")
    print(f"- iter: {layout_row.get('iter')}")
    print(f"- query_id: {layout_row.get('query_id')}")
    print(f"- objects: {layout_row.get('object_list')}")


def _normalize_item_name(name: str) -> str:
    return str(name).strip().lstrip(".").lower()


def _candidate_furniture_options(filtered_data: dict) -> list[str]:
    options: list[str] = []
    for item in filtered_data.get("meta", {}).get("required_items", []):
        n = _normalize_item_name(item)
        if n and n not in options:
            options.append(n)
    for item in filtered_data.get("items_list", []):
        n = _normalize_item_name(item)
        if n and n not in options:
            options.append(n)
    for row in filtered_data.get("rows", [])[:3]:
        for obj in row.get("object_list", []):
            if isinstance(obj, list) and obj:
                n = _normalize_item_name(obj[0])
                if n and n not in options:
                    options.append(n)
    return options[:6] if options else ["bed", "chair", "table", "tv-stand"]


def _layout_context_snippet(filtered_data: dict, n_rows: int = 2) -> str:
    snippets: list[str] = []
    for row in filtered_data.get("rows", [])[:n_rows]:
        qid = row.get("query_id", "unknown")
        iter_id = row.get("iter", "unknown")
        items = []
        for obj in row.get("object_list", []):
            if isinstance(obj, list) and obj:
                items.append(_normalize_item_name(obj[0]))
        snippets.append(f"{qid}:{iter_id} -> items={items}")
    return " | ".join(snippets) if snippets else "no layout samples"


def _parse_json_tolerant(raw: str) -> dict:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_questions(filtered_data: dict) -> list[dict[str, object]]:
    return [
        {
            "question": "What is the main use of this room?",
            "options": ["sleep", "work", "relax", "mixed"],
        },
        {
            "question": "Which style should dominate?",
            "options": ["minimal", "cozy", "modern", "balanced"],
        },
        {
            "question": "Which spatial priority matters most?",
            "options": ["open center", "privacy", "wall-aligned", "work triangle"],
        },
        {
            "question": "Which furniture item should be prioritized first?",
            "options": _candidate_furniture_options(filtered_data)[:4],
        },
    ]


def _extract_preference_slots(text: str) -> dict[str, str]:
    lowered = text.lower()
    slots: dict[str, str] = {}
    if any(k in lowered for k in ["sleep", "rest"]):
        slots["activity"] = "sleep"
    elif any(k in lowered for k in ["work", "study"]):
        slots["activity"] = "work"
    elif any(k in lowered for k in ["relax", "chill"]):
        slots["activity"] = "relax"
    elif "mixed" in lowered:
        slots["activity"] = "mixed"

    if any(k in lowered for k in ["minimal", "modern", "cozy", "balanced"]):
        for k in ["minimal", "modern", "cozy", "balanced"]:
            if k in lowered:
                slots["style"] = k
                break

    if "open center" in lowered or "open" in lowered or "walk" in lowered:
        slots["spatial_priority"] = "open center"
    elif "privacy" in lowered:
        slots["spatial_priority"] = "privacy"
    elif "wall" in lowered:
        slots["spatial_priority"] = "wall-aligned"
    elif "triangle" in lowered:
        slots["spatial_priority"] = "work triangle"

    if any(k in lowered for k in ["bed", "chair", "desk", "table", "tv-stand", "cupboard"]):
        m = re.search(r"\b(bed|chair|desk|table|tv-stand|cupboard)\b", lowered)
        if m:
            slots["priority_item"] = m.group(1)
    return slots


def _preference_confidence(slots: dict[str, str]) -> float:
    required = ["activity", "style", "spatial_priority"]
    optional = ["priority_item"]
    score = 0.0
    score += 0.22 * sum(1 for k in required if k in slots)
    score += 0.12 * sum(1 for k in optional if k in slots)
    return min(1.0, score)


def _next_fallback_question(slots: dict[str, str], filtered_data: dict) -> dict[str, object]:
    if "activity" not in slots:
        return {"question": "What is the main use of this room?", "options": ["sleep", "work", "relax", "mixed"]}
    if "style" not in slots:
        return {"question": "Which style should dominate?", "options": ["minimal", "cozy", "modern", "balanced"]}
    if "spatial_priority" not in slots:
        return {
            "question": "Which spatial priority matters most?",
            "options": ["open center", "privacy", "wall-aligned", "work triangle"],
        }
    return {
        "question": "Which furniture item should be prioritized first?",
        "options": _candidate_furniture_options(filtered_data)[:4],
    }


def _load_dynamic_questions(filtered_data: dict) -> list[dict[str, object]]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Generate exactly 3 CRS questions for interior layout preference elicitation. "
                    "For each question include exactly 4 short options. "
                    "Use the provided furniture options only when asking item-priority questions. "
                    "Return strict JSON only in this format: "
                    "{{\"questions\":[{{\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"]}}]}}"
                ),
            ),
            (
                "human",
                (
                    "Create questions covering activity, style, and spatial priorities.\n"
                    "Furniture options: {furniture_options}\n"
                    "Sample layouts: {layout_samples}"
                ),
            ),
        ]
    )
    model = ChatOllama(
        model=get_ollama_model(),
        temperature=0.2,
        format="json",
        client_kwargs={"timeout": 35},
    )
    raw = str(
        (prompt | model).invoke(
            {
                "furniture_options": ", ".join(_candidate_furniture_options(filtered_data)),
                "layout_samples": _layout_context_snippet(filtered_data),
            }
        ).content
    ).strip()
    data = _parse_json_tolerant(raw)
    questions = data.get("questions", [])
    valid: list[dict[str, object]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        options = item.get("options", [])
        if not q or not isinstance(options, list):
            continue
        opt_text = [str(o).strip() for o in options if str(o).strip()]
        if len(opt_text) < 2:
            continue
        valid.append({"question": q, "options": opt_text[:4]})
    if len(valid) != 3:
        raise ValueError("Invalid dynamic question payload")
    return valid


def _load_next_dynamic_question(slots: dict[str, str], history: list[str], filtered_data: dict) -> dict[str, object]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a conversational recommender for interior layouts. "
                    "Ask exactly one next-best question to reduce uncertainty. "
                    "Return strict JSON only: {{\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"]}}"
                ),
            ),
            (
                "human",
                (
                    "Known slots: {slots}\n"
                    "Question/answer history: {history}\n"
                    "Furniture options: {furniture_options}\n"
                    "Sample layouts: {layout_samples}\n"
                    "Ask one best next question now."
                ),
            ),
        ]
    )
    model = ChatOllama(
        model=get_ollama_model(),
        temperature=0.2,
        format="json",
        client_kwargs={"timeout": 35},
    )
    raw = str(
        (prompt | model).invoke(
            {
                "slots": str(slots),
                "history": " | ".join(history),
                "furniture_options": ", ".join(_candidate_furniture_options(filtered_data)),
                "layout_samples": _layout_context_snippet(filtered_data),
            }
        ).content
    ).strip()
    data = _parse_json_tolerant(raw)
    q = str(data.get("question", "")).strip()
    options = [str(o).strip() for o in data.get("options", []) if str(o).strip()]
    if not q or len(options) < 2:
        raise ValueError("Invalid dynamic question payload")
    return {"question": q, "options": options[:4]}


def initial_questionnaire(filtered_data: dict, min_confidence: float, max_questions: int) -> str:
    print("\nCRS preference elicitation:")
    slots: dict[str, str] = {}
    history: list[str] = []
    q_count = 0

    while q_count < max_questions:
        confidence = _preference_confidence(slots)
        if confidence >= min_confidence:
            break
        try:
            if q_count < 3:
                warmup = _load_dynamic_questions(filtered_data)
                item = warmup[q_count]
            else:
                item = _load_next_dynamic_question(slots=slots, history=history, filtered_data=filtered_data)
            source = "LLM"
        except Exception as exc:
            if q_count == 0:
                print(f"Using fallback CRS questions (LLM question generation failed: {exc}).")
            item = _next_fallback_question(slots, filtered_data=filtered_data)
            source = "fallback"

        q_count += 1
        question = str(item["question"])
        options = [str(o) for o in item.get("options", [])]
        print(f"\n[{source}] Q{q_count} (confidence={confidence:.2f}). {question}")
        for idx, opt in enumerate(options, start=1):
            print(f"  {idx}. {opt}")
        raw = ask("Your choice (number or custom text): ").strip()
        if not raw:
            print("Please enter a choice. Re-asking same question.")
            q_count -= 1
            continue
        if raw.isdigit():
            pick = int(raw)
            ans = options[pick - 1] if 1 <= pick <= len(options) else "not specified"
        else:
            ans = raw or "not specified"
        pair = f"{question} -> {ans}"
        history.append(pair)
        slots.update(_extract_preference_slots(ans))
        slots.update(_extract_preference_slots(question + " " + ans))

    print(f"\nCRS readiness confidence: {_preference_confidence(slots):.2f} after {q_count} questions.")
    return " | ".join(history)


def main() -> None:
    args = parse_args()
    filter_path = args.filter_path
    if not filter_path.is_absolute():
        root = Path(__file__).resolve().parent.parent.parent
        filter_path = root / filter_path
    if not filter_path.exists():
        raise SystemExit(f"Missing filter file: {filter_path}")

    filtered_data = load_filtered(filter_path)
    session_id = str(uuid4())
    print("=== Phase2 Conversational CRS CLI ===")
    print(f"Session: {session_id}")
    print(f"Model: {get_ollama_model()}")
    print(f"Filter: {filter_path}")

    recommendation = recommend_layout(
        filtered_data=filtered_data,
        user_request=initial_questionnaire(
            filtered_data=filtered_data,
            min_confidence=args.min_confidence,
            max_questions=args.max_questions,
        ),
        max_candidates=args.max_candidates,
    )
    if recommendation is None:
        raise SystemExit("Could not generate initial recommendation.")

    set_current_layout(session_id=session_id, recommended_id=recommendation["recommended_id"], layout=recommendation["recommended_layout"])
    print("\nCRS Recommendation:")
    print(f"- recommended_id: {recommendation['recommended_id']}")
    print(f"- pitch: {recommendation['pitch']}")
    print_layout_summary(recommendation["recommended_layout"])

    print("\nType feedback to continue. Commands: /show, /exit (or plain 'exit')")
    while True:
        user_message = ask("\nYou: ")
        if not user_message:
            continue
        if user_message in {"/exit", "exit", "quit"}:
            print("Session ended.")
            break
        if user_message == "/show":
            print_layout_summary(recommendation.get("recommended_layout"))
            continue

        routed = route_chat_turn(
            filtered_data=filtered_data,
            session_id=session_id,
            user_message=user_message,
            max_candidates=args.max_candidates,
        )
        print(f"CRS action: {routed['action']} (confidence={routed.get('confidence')})")
        print(f"CRS: {routed['message']}")

        if routed["action"] == "switch":
            recommendation = {"recommended_id": routed["recommended_id"], "recommended_layout": routed["recommended_layout"]}
            print_layout_summary(routed["recommended_layout"])
            continue

        if routed["action"] == "modify" and routed.get("recommended_layout") and routed.get("instruction"):
            regenerated = apply_instruction_to_layout(
                layout_row=routed["recommended_layout"],
                instruction=routed["instruction"],
            )
            stored = store_regenerated_layout(
                session_id=session_id,
                generated_layout=regenerated,
                source_layout_id=routed.get("recommended_id"),
                instruction=routed["instruction"],
            )
            recommendation = {"recommended_id": stored["recommended_id"], "recommended_layout": stored["recommended_layout"]}
            print("CRS regenerated layout based on your request.")
            print_layout_summary(stored["recommended_layout"])


if __name__ == "__main__":
    main()
