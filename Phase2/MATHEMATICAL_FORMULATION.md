# Phase2 Mathematical Formulation

This document formalizes the current Phase2 implementation (API + CLI CRS loop) in mathematical terms.

## 1) Problem Setup

Let:

- `D = {l_i}_{i=1}^N` be candidate layouts from `Phase1/filter.json`.
- Each layout `l_i` has:
  - identifier `id_i`,
  - prompt/context `p_i`,
  - object set `O_i = {(c_{ij}, b_{ij})}_{j=1}^{m_i}` where:
    - `c_{ij}` = object class (e.g., bed, chair),
    - `b_{ij} = (x1, y1, x2, y2)` = box coordinates.
- Conversation session `s` has state:
  - `h_t` = history up to turn `t`,
  - `k_t` = extracted preference slots,
  - `l_t^*` = currently selected layout (or null initially).

Goal: learn enough preference information from dialogue, then select and iteratively refine layout:

`(D, h_t, k_t) -> l_t^* -> feedback -> switch/modify -> l_{t+1}^*`

---

## 2) Candidate Text Wrapping (for LLM ranking)

For each layout `l_i`, a text wrapper `w_i` is constructed:

`w_i = f_wrap(l_i) = "Layout iter: room WxH, objects -> class at box, ..."`

where object labels are normalized (`.bed -> bed`, etc.).

This gives candidate text set:

`W = {w_i}_{i=1}^N`

used by the LLM recommender.

---

## 3) Initial Recommendation Function

Implemented in `core/recommender.py`.

### 3.1 LLM structured selection

Given user intent text `u_t` and candidates `W`, LLM is asked to output JSON:

`y_t = {"recommended_id": id_k, "pitch": r_t}`

If valid and `id_k in {id_i}`, return layout `l_k`.

### 3.2 Fallback lexical ranker

If LLM call/parsing fails, score each candidate by token overlap:

`score_i = sum_{q in Tok(u_t)} 1[q in Tok(w_i)]`

Pick:

`k = argmax_i score_i`

and return fallback pitch.

So recommendation operator is:

`R(u_t, D) = l_k`

where `k` comes from LLM JSON or lexical fallback.

---

## 4) Conversational Preference State

Implemented in `core/chat_router.py` and CLI.

Preference extractor:

`k_t = k_{t-1} U Extract(h_t.user_message)`

Example slots:

- activity, style, spatial priority, privacy, lighting, priority item.

---

## 5) Confidence Heuristic

### API/router confidence (session loop)

Current formula:

- `hist = min(|h_t|, 6)`
- `pref = min(|k_t|, 7)`
- `cur = 1 if l_t^* exists else 0`

`C_api(t) = clip(0.25*cur + 0.08*hist + 0.12*pref, 0, 1)`

If `C_api(t) < 0.45`, system asks clarification.

### CLI pre-recommendation confidence

Required slots: activity, style, spatial_priority  
Optional slot: priority_item

`C_cli(t) = min(1, 0.22*#required_filled + 0.12*#optional_filled)`

CLI asks questions until:

`C_cli(t) >= tau` (default `tau = 0.72`)  
or `q_count >= q_max` (default `q_max = 8`).

---

## 6) CRS Action Routing

Given `(h_t, k_t, l_t^*)`, route:

`a_t in {ask_clarification, switch, modify}`

### 6.1 LLM router

LLM outputs strict JSON:

`{"action": a_t, "reason": ..., "instruction": ...}`

### 6.2 Rule fallback router

- If feedback contains move-like keywords -> `modify`
- If uncertainty phrase -> `ask_clarification`
- Else -> `switch`

So routing operator:

`a_t = Pi(h_t, k_t, l_t^*)`

---

## 7) Switch Transition

When `a_t = switch`:

`l_{t+1}^* = R(user_feedback_t, D)`

and append assistant pitch to history.

---

## 8) Modify Transition and Regeneration

When `a_t = modify`, instruction payload `m_t` is generated and applied.

## 8.1 Edit operator

Implemented in `core/layout_editor.py`:

`l_{t+1}^* = E(l_t^*, m_t, target_dims)`

where `E` supports:

1. `resize_room(width, height)`  
2. `move(target, direction, delta)` or `move(target, dx, dy)`  
3. `set_box(target, box)`

### Move transform

If box `b = (x1,y1,x2,y2)` and offset `(dx,dy)`:

`b' = (x1+dx, y1+dy, x2+dx, y2+dy)`

Direction form maps to signed `dx,dy` by rule.

### Resize transform

If old dims `(W,H)` and new dims `(W',H')`, for absolute-coordinate boxes:

`x' = x * (W'/W)`, `y' = y * (H'/H)`

Prompt/query metadata is also updated to reflect new room dimensions.

---

## 9) Session Memory Dynamics

Each session stores:

- `current_layout_id`
- `current_layout`
- `history`
- `known_preferences`
- `temporary_layouts` (regenerated variants)

Update map:

`S_{t+1} = T(S_t, user_t, assistant_t, action_t)`

where `T` applies route outcome and layout transitions.

---

## 10) End-to-End Pipeline Equation

For turn `t`:

1. Update preferences:
   - `k_t = k_{t-1} U Extract(user_t)`
2. Compute confidence:
   - `C_t = C(S_t)`
3. If `C_t < tau_ask`: ask clarification question `q_t`
4. Else route:
   - `a_t = Pi(S_t, user_t)`
5. Transition:
   - if `a_t = switch`: `l_{t+1}^* = R(user_t, D)`
   - if `a_t = modify`: `l_{t+1}^* = E(l_t^*, m_t, dims_t)`
6. Return `(a_t, message_t, l_{t+1}^*)`

---

## 11) Mapping to Your `test_run.txt`

From `Phase2/test_run.txt`:

- Q1..Q6 progressively fill slots (`activity`, `style`, `spatial_priority`, `priority_item`, etc.)
- Confidence reached `0.78` after 6 questions, crossing default threshold `0.72`
- Recommendation returned:
  - `recommended_id = roomgen_000027:2`
  - pitch explaining selected preference alignment

This is exactly the implemented adaptive CRS loop:

`ask -> update slots -> raise confidence -> recommend -> critique loop`

---

## 12) Practical Notes on Current Implementation

- LLM calls are strict-JSON first, with tolerant parsing fallback.
- If LLM fails, deterministic fallback logic keeps pipeline live.
- Confidence is heuristic (not learned); can be upgraded to data-driven scoring later.
- Edit/regeneration currently uses deterministic JSON transforms, which is efficient for small modifications.

