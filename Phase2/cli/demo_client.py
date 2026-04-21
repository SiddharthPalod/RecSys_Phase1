from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def _pretty(title: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


def run_demo(base_url: str) -> None:
    session_start = requests.post(f"{base_url}/api/session/start", json={}, timeout=30)
    session_start.raise_for_status()
    session_data = session_start.json()
    session_id = session_data["session_id"]
    _pretty("Session Start", session_data)

    rec = requests.post(
        f"{base_url}/api/recommend_initial",
        json={
            "session_id": session_id,
            "user_request": "I want a spacious layout with clear walking space in center.",
            "max_candidates": 10,
        },
        timeout=60,
    )
    rec.raise_for_status()
    rec_data = rec.json()
    _pretty("Initial Recommendation", rec_data)

    chat_modify = requests.post(
        f"{base_url}/api/chat",
        json={
            "session_id": session_id,
            "user_message": "Move the bed away from door and keep chair near wall.",
            "max_candidates": 10,
        },
        timeout=60,
    )
    chat_modify.raise_for_status()
    chat_modify_data = chat_modify.json()
    _pretty("Chat Turn (Modify/Switch/Ask)", chat_modify_data)

    if (
        chat_modify_data.get("action") == "modify"
        and chat_modify_data.get("instruction")
        and chat_modify_data.get("recommended_layout")
    ):
        regen = requests.post(
            f"{base_url}/api/regenerate_layout",
            json={
                "session_id": session_id,
                "source_layout_id": chat_modify_data.get("recommended_id"),
                "layout": chat_modify_data["recommended_layout"],
                "instruction": chat_modify_data["instruction"],
            },
            timeout=60,
        )
        regen.raise_for_status()
        _pretty("Regeneration Hook", regen.json())
    else:
        print("\n=== Regeneration Hook ===")
        print("Skipped (router did not return modify with instruction + layout).")

    chat_switch = requests.post(
        f"{base_url}/api/chat",
        json={
            "session_id": session_id,
            "user_message": "Show me another alternative layout option.",
            "max_candidates": 10,
        },
        timeout=60,
    )
    chat_switch.raise_for_status()
    _pretty("Chat Turn (Alternative Request)", chat_switch.json())
    print("\nDemo flow complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase2 API demo client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    run_demo(base_url=args.base_url.rstrip("/"))


if __name__ == "__main__":
    main()
