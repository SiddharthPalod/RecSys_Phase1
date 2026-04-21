# Phase 2 - Conversational Recommendation API

Based on paper: 
@inproceedings{he23large,
  title = Large language models as zero-shot conversational recommenders",
  author = "Zhankui He and Zhouhang Xie and Rahul Jha and Harald Steck and Dawen Liang and Yesu Feng and Bodhisattwa Majumder and Nathan Kallus and Julian McAuley",
  year = "2023",
  booktitle = "CIKM"
}

## Folder Structure

- `Phase2/api/` FastAPI app and endpoint schemas
- `Phase2/core/` recommender, router, regeneration, session logic
- `Phase2/cli/` interactive terminal CRS and API demo client
- `Phase2/config/` runtime settings (Ollama model)
- `Phase2/MATHEMATICAL_FORMULATION.md` formal equations and pipeline derivation

This folder implements Phase 2 from the project plan:

- Load Phase 1 filtered layouts (`Phase1/filter.json`)
- Build text descriptions for each layout candidate
- Ask Ollama (via LangChain) to select a layout + pitch
- Return a UI-ready payload for rendering
- Keep in-memory session state for multi-turn feedback
- Route follow-up feedback into `switch` vs `modify`

## API

### `POST /api/recommend_initial`

Request body:

```json
{
  "user_request": "I want more open space near center",
  "max_candidates": 10
}
```

Response body:

```json
{
  "recommended_id": "room_100x1000_97fc2adf:4",
  "pitch": "I selected this because ...",
  "recommended_layout": { "...": "full row from filter.json" }
}
```

### `POST /api/chat`

Request body:

```json
{
  "session_id": "demo-user-1",
  "user_message": "Move the bed away from the door",
  "max_candidates": 10
}
```

Response body (action: `modify`):

```json
{
  "session_id": "demo-user-1",
  "action": "modify",
  "message": "I understood this as a modify request on the current layout...",
  "recommended_id": "room_100x1000_97fc2adf:4",
  "recommended_layout": { "...": "current layout row" },
  "instruction": {
    "target": "bed",
    "action": "move",
    "directive": "Move the bed away from the door"
  }
}
```

Response body (action: `switch`) returns a newly selected `recommended_id` and `recommended_layout`.

## Run

```bash
pip install -r Phase2/requirements.txt
uvicorn Phase2.api.app:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Notes

- If LangChain/Ollama is unavailable, the endpoint falls back to a lexical ranker so the API still returns a valid recommendation.
- `recommended_layout` is returned in full so your frontend can directly render the coordinate JSON in the iteration explorer.
- Session memory is in-process Python memory (`session_id` keyed dictionary). Restarting the API clears sessions.

## Full Phase Coverage

This implementation now covers all planned phases:

- **Phase 2**: Initial recommendation endpoint (`/api/recommend_initial`)
- **Phase 3**: Session management and memory (`/api/session/start`, `/api/session/reset`, `session_id` tracking)
- **Phase 4**: Critique loop routing (`/api/chat` with actions `switch`, `modify`, `ask_clarification`)
- **Phase 5**: Regeneration hook (`/api/regenerate_layout`) that applies instruction-driven coordinate updates and stores temporary regenerated layouts in session state

### `POST /api/session/start`

Optional request body:

```json
{
  "session_id": "demo-user-1"
}
```

If omitted, a UUID session id is generated.

### `POST /api/session/reset`

```json
{
  "session_id": "demo-user-1"
}
```

### `POST /api/regenerate_layout`

```json
{
  "session_id": "demo-user-1",
  "source_layout_id": "room_100x1000_97fc2adf:4",
  "layout": {
    "iter": 4,
    "object_list": [["bed", [0.0, 0.0, 0.1, 0.1]]]
  },
  "instruction": {
    "target": "bed",
    "action": "move",
    "direction": "away from door"
  },
  "target_room_width": 399,
  "target_room_height": 399
}
```

### `POST /api/layout/edit`

Direct JSON editor endpoint for dimension mismatch and fine edits.

```json
{
  "layout": { "...": "layout row json" },
  "target_room_width": 399,
  "target_room_height": 399,
  "instruction": {
    "edits": [
      { "type": "move", "target": "bed", "direction": "right", "delta": 0.02 },
      { "type": "set_box", "target": "chair", "box": [0.1, 0.2, 0.3, 0.4] }
    ]
  }
}
```

Supported edit operations:

- `resize_room` (`width`, `height`)
- `move` (`target`, plus either `direction`+`delta` or `dx`/`dy`)
- `set_box` (`target`, `box`)

## Demo Client

Run the API server first:

```bash
uvicorn Phase2.api.app:app --reload
```

Then execute the end-to-end demo flow:

```bash
python -m Phase2.cli.demo_client
```

Optional custom host:

```bash
python -m Phase2.cli.demo_client --base-url http://127.0.0.1:8000
```

## Interactive CLI CRS

If you want conversational flow directly in terminal (without frontend/API client), use:

```bash
python -m Phase2.cli.interactive
```

Optional:

```bash
python -m Phase2.cli.interactive --filter-path Phase1/filter.json --max-candidates 10
```

The CLI:

- asks initial preference questions,
- recommends a layout with pitch,
- accepts iterative feedback,
- routes feedback to `switch` / `modify` / `ask_clarification`,
- auto-applies regeneration on `modify` turns.

### Model alignment with Phase1

Phase2 now defaults to the same Ollama model as Phase1:

- default model: `llama3.2:1b`

You can override with env var:

```bash
set PHASE2_OLLAMA_MODEL=llama3.2:1b
```

If model invocation fails or model is missing, Phase2 gracefully falls back to lexical ranking/routing instead of returning server 500.
