"""
Clean llm_output/counting/ollama.counting.room_dataset.json:
- Dedupe by (query_id, iter), keep first
- Keep only object labels that match the prompt's furniture list (per row)
- Normalize labels (NFKC, lowercase, canonical spelling when unambiguous)
- Drop rows that should have furniture but end up with no boxes after filtering
- Keep empty-room prompts when object_list is correctly empty after filtering
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path

BOXES_RE = re.compile(
    r"Place exactly \d+ movable objects with non-overlapping boxes:\s*(.+?)\.\s*Keep",
    re.DOTALL,
)


def alnum_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def parse_expected_objects(prompt: str) -> list[str] | None:
    if "intentionally empty" in prompt.lower():
        return []
    m = BOXES_RE.search(prompt)
    if not m:
        return None
    body = m.group(1)
    # Oxford comma: "..., table, and bookshelf" -> split on ", and " first
    body = re.sub(r",\s+and\s+", ",", body)
    parts = [p.strip() for p in body.split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p.lower().startswith("and "):
            p = p[4:].strip()
        if p:
            out.append(p)
    return out


def label_allowed(raw: str, expected: list[str]) -> bool:
    lab = raw.strip()
    if not lab or lab.startswith(("#", "(", ".")):
        return False
    if not alnum_key(lab):
        return False
    exp_alnums = {alnum_key(e) for e in expected}
    ln = alnum_key(lab)
    if ln in exp_alnums:
        return True
    low = unicodedata.normalize("NFKC", lab).strip().lower()
    if low == "chair" and any(alnum_key(e).startswith("chair") for e in expected):
        return True
    if low in ("tv", "television") and any(
        alnum_key(e) in ("tvstand", "tv") or "tvstand" in alnum_key(e) for e in expected
    ):
        return True
    if ln == "soaf" and "sofa" in exp_alnums:
        return True
    if ln == "smaltable" and "table" in exp_alnums:
        return True
    return False


def canonicalize_label(raw: str, expected: list[str]) -> str:
    lab_lower = unicodedata.normalize("NFKC", raw).strip().lower()
    ln = alnum_key(raw)
    matches = [e for e in expected if alnum_key(e) == ln]
    if len(matches) == 1:
        return matches[0].lower()
    if lab_lower == "chair":
        return "chair"
    if lab_lower in ("tv", "television"):
        for e in expected:
            ek = alnum_key(e)
            if ek in ("tvstand", "tv") or "tvstand" in ek:
                return e.lower()
    if ln == "soaf":
        for e in expected:
            if alnum_key(e) == "sofa":
                return e.lower()
    if ln == "smaltable":
        for e in expected:
            if alnum_key(e) == "table":
                return e.lower()
    if matches:
        return matches[0].lower()
    return lab_lower


def clean_record(rec: dict) -> dict | None:
    prompt = rec.get("prompt") or ""
    expected = parse_expected_objects(prompt)
    if expected is None:
        return rec

    new_list = []
    for item in rec.get("object_list") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        label, bbox = item[0], item[1]
        if not isinstance(label, str):
            continue
        if not label_allowed(label, expected):
            continue
        canon = canonicalize_label(label, expected)
        new_list.append([canon, bbox])

    out = {**rec, "object_list": new_list}

    is_empty_prompt = "intentionally empty" in prompt.lower()
    if is_empty_prompt:
        return out
    if not new_list:
        return None
    return out


def dedupe(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    dropped = 0
    for r in records:
        key = (r.get("query_id", ""), r.get("iter", -1))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("llm_output/counting/ollama.counting.room_dataset.json"),
    )
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    path = args.input
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    before = len(data)
    data, dup_dropped = dedupe(data)

    cleaned: list[dict] = []
    removed_no_furniture = 0
    for r in data:
        c = clean_record(r)
        if c is None:
            removed_no_furniture += 1
            continue
        cleaned.append(c)

    if not args.no_backup:
        bak = path.with_suffix(path.suffix + ".bak_pre_strict")
        shutil.copy2(path, bak)
        print("backup:", bak)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print("records before:", before)
    print("dedupe dropped:", dup_dropped)
    print("removed (prompt wanted objects but none left):", removed_no_furniture)
    print("records after:", len(cleaned))


if __name__ == "__main__":
    main()
