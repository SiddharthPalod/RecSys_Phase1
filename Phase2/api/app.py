from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core import (
    apply_instruction_to_layout,
    edit_layout_json,
    recommend_layout,
    reset_session,
    route_chat_turn,
    set_current_layout,
    start_session,
    store_regenerated_layout,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FILTER_PATH = BASE_DIR.parent / "Phase1" / "filter.json"


class RecommendInitialRequest(BaseModel):
    user_request: str = Field(...)
    filter_path: str | None = Field(default=None)
    max_candidates: int = Field(default=10, ge=1, le=25)
    session_id: str | None = Field(default=None)


class RecommendInitialResponse(BaseModel):
    recommended_id: str
    pitch: str
    recommended_layout: dict[str, Any]
    session_id: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    filter_path: str | None = Field(default=None)
    max_candidates: int = Field(default=10, ge=1, le=25)


class ChatResponse(BaseModel):
    session_id: str
    action: str
    message: str
    recommended_id: str | None
    recommended_layout: dict[str, Any] | None
    instruction: dict[str, Any] | None
    confidence: float | None = None


class SessionStartRequest(BaseModel):
    session_id: str | None = None


class SessionStartResponse(BaseModel):
    session_id: str
    message: str


class SessionResetRequest(BaseModel):
    session_id: str


class SessionResetResponse(BaseModel):
    session_id: str
    cleared: bool


class RegenerateLayoutRequest(BaseModel):
    session_id: str
    source_layout_id: str | None = None
    layout: dict[str, Any]
    instruction: dict[str, Any]
    target_room_width: int | None = None
    target_room_height: int | None = None


class RegenerateLayoutResponse(BaseModel):
    session_id: str
    recommended_id: str
    message: str
    recommended_layout: dict[str, Any]


class EditLayoutRequest(BaseModel):
    layout: dict[str, Any]
    instruction: dict[str, Any] | None = None
    target_room_width: int | None = None
    target_room_height: int | None = None


class EditLayoutResponse(BaseModel):
    edited_layout: dict[str, Any]


app = FastAPI(title="Phase 2 Conversational Layout Recommender")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session/start", response_model=SessionStartResponse)
def session_start(payload: SessionStartRequest) -> SessionStartResponse:
    session_id = start_session(payload.session_id)
    return SessionStartResponse(session_id=session_id, message="Session ready for conversational recommendation.")


@app.post("/api/session/reset", response_model=SessionResetResponse)
def session_reset(payload: SessionResetRequest) -> SessionResetResponse:
    return SessionResetResponse(session_id=payload.session_id, cleared=reset_session(payload.session_id))


@app.post("/api/recommend_initial", response_model=RecommendInitialResponse)
def recommend_initial(payload: RecommendInitialRequest) -> RecommendInitialResponse:
    filter_path = Path(payload.filter_path) if payload.filter_path else DEFAULT_FILTER_PATH
    if not filter_path.exists():
        raise HTTPException(status_code=404, detail=f"Filter file not found: {filter_path}")
    try:
        with filter_path.open("r", encoding="utf-8") as f:
            filtered_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid filter JSON: {exc}") from exc

    result = recommend_layout(filtered_data=filtered_data, user_request=payload.user_request, max_candidates=payload.max_candidates)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate recommendation")
    if payload.session_id:
        set_current_layout(session_id=payload.session_id, recommended_id=result["recommended_id"], layout=result["recommended_layout"])
        result["session_id"] = payload.session_id
    return RecommendInitialResponse(**result)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    filter_path = Path(payload.filter_path) if payload.filter_path else DEFAULT_FILTER_PATH
    if not filter_path.exists():
        raise HTTPException(status_code=404, detail=f"Filter file not found: {filter_path}")
    try:
        with filter_path.open("r", encoding="utf-8") as f:
            filtered_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid filter JSON: {exc}") from exc

    response = route_chat_turn(
        filtered_data=filtered_data,
        session_id=payload.session_id,
        user_message=payload.user_message,
        max_candidates=payload.max_candidates,
    )
    return ChatResponse(**response)


@app.post("/api/regenerate_layout", response_model=RegenerateLayoutResponse)
def regenerate_layout(payload: RegenerateLayoutRequest) -> RegenerateLayoutResponse:
    regenerated = apply_instruction_to_layout(
        layout_row=payload.layout,
        instruction=payload.instruction,
        target_room_width=payload.target_room_width,
        target_room_height=payload.target_room_height,
    )
    stored = store_regenerated_layout(
        session_id=payload.session_id,
        generated_layout=regenerated,
        source_layout_id=payload.source_layout_id,
        instruction=payload.instruction,
    )
    return RegenerateLayoutResponse(session_id=payload.session_id, **stored)


@app.post("/api/layout/edit", response_model=EditLayoutResponse)
def edit_layout(payload: EditLayoutRequest) -> EditLayoutResponse:
    edited = edit_layout_json(
        layout_row=payload.layout,
        instruction=payload.instruction,
        target_room_width=payload.target_room_width,
        target_room_height=payload.target_room_height,
    )
    return EditLayoutResponse(edited_layout=edited)
