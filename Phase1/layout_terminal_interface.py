import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_FURNITURE = [
    "bed",
    "chair",
    "table",
    "sofa",
    "cupboard",
    "wardrobe",
    "desk",
    "bookshelf",
    "nightstand",
    "coffee-table",
    "tv-stand",
    "dining-table",
    "sideboard",
    "rug",
    "lamp",
    "side-table",
    "plant",
    "single-bed",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 1 terminal interface for room-layout dataset filtering and background generation."
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("Phase1/room_dataset_cleaned.json"),
        help="Path to cleaned dataset JSON list. Will be created if missing.",
    )
    ap.add_argument(
        "--layoutgpt_dir",
        type=Path,
        default=Path("LayoutGPT"),
        help="Directory containing run_layoutgpt_2d.py",
    )
    ap.add_argument(
        "--min_results",
        type=int,
        default=10,
        help="Trigger generation when filtered rows are below this count.",
    )
    ap.add_argument(
        "--show_limit",
        type=int,
        default=20,
        help="Maximum filtered rows to print in terminal.",
    )
    ap.add_argument(
        "--show_results",
        action="store_true",
        help="Print detailed filtered rows (debugging only).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not launch background generation; print what would be launched.",
    )
    ap.add_argument(
        "--job_status",
        action="store_true",
        help="Show background job status (running/finished/failed) and exit.",
    )
    ap.add_argument(
        "--filtered_items_out",
        type=Path,
        default=Path("Phase1/filtered_items.json"),
        help="Output JSON artifact for Phase 2 containing filtered rows + aggregated items list.",
    )
    return ap.parse_args()


def ensure_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]\n", encoding="utf-8")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Dataset JSON is invalid: {path} ({exc})") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Dataset must be a JSON list: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_label(label: str) -> str:
    s = (label or "").strip().lower()
    s = re.sub(r"[^a-z0-9\- ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def object_labels_from_row(row: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for item in row.get("object_list", []):
        if isinstance(item, list) and len(item) >= 1 and isinstance(item[0], str):
            labels.add(normalize_label(item[0]))
    return labels


FURNITURE_ALIASES: dict[str, set[str]] = {
    "bed": {"bed", "single-bed"},
    "single-bed": {"single-bed", "bed"},
    "tv": {"tv", "tv-stand", "television"},
    "television": {"tv", "tv-stand", "television"},
    "tv-stand": {"tv-stand", "tv", "television"},
}


def expand_required_items(required_items: list[str]) -> list[set[str]]:
    expanded: list[set[str]] = []
    for item in required_items:
        normalized = normalize_label(item)
        expanded.append(FURNITURE_ALIASES.get(normalized, {normalized}))
    return expanded


def matches_required_items(labels: set[str], required_items: list[str]) -> bool:
    required_groups = expand_required_items(required_items)
    return all(any(candidate in labels for candidate in group) for group in required_groups)


def read_int(prompt: str, min_value: int = 1) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
            if val < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue
            return val
        except ValueError:
            print("Please enter an integer.")


def choose_furniture() -> list[str]:
    print("\nSelect furniture items (comma-separated numbers).")
    for idx, item in enumerate(DEFAULT_FURNITURE, start=1):
        print(f"  {idx:2d}. {item}")
    print(f"  {len(DEFAULT_FURNITURE) + 1:2d}. other (custom)")

    selected: list[str] = []
    while True:
        raw = input("Your choices (example: 1,2,5): ").strip()
        if not raw:
            print("Pick at least one item.")
            continue
        try:
            picks = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("Please enter numbers separated by commas.")
            continue

        max_choice = len(DEFAULT_FURNITURE) + 1
        if any(p < 1 or p > max_choice for p in picks):
            print(f"Choices must be between 1 and {max_choice}.")
            continue

        seen: set[str] = set()
        for p in picks:
            if p == max_choice:
                custom = input("Enter custom furniture name: ").strip()
                if custom:
                    n = normalize_label(custom)
                    if n and n not in seen:
                        selected.append(n)
                        seen.add(n)
            else:
                n = normalize_label(DEFAULT_FURNITURE[p - 1])
                if n not in seen:
                    selected.append(n)
                    seen.add(n)

        if not selected:
            print("Pick at least one valid item.")
            continue
        return selected


def build_prompt(room_w: int, room_h: int, furniture: list[str]) -> str:
    if len(furniture) == 1:
        furniture_phrase = furniture[0]
    else:
        furniture_phrase = ", ".join(furniture[:-1]) + f", and {furniture[-1]}"

    return (
        f"A rectangular room interior seen from above (bird's-eye layout) on a {room_w}x{room_h} pixel canvas. "
        f"Place exactly {len(furniture)} movable objects with non-overlapping boxes: {furniture_phrase}. "
        "Keep all boxes fully inside the room. Leave walkspace through the center. "
        "Output only CSS boxes, one per line, with the given labels."
    )


def filter_dataset(
    dataset: list[dict[str, Any]],
    room_w: int,
    room_h: int,
    required_items: list[str],
) -> list[dict[str, Any]]:
    out = []
    room_dim_token = f"{room_w}x{room_h}"
    for row in dataset:
        labels = object_labels_from_row(row)
        if not matches_required_items(labels, required_items):
            continue
        prompt = str(row.get("prompt", ""))
        # Prefer exact room-size match when encoded in prompt; allow fallback otherwise.
        if room_dim_token in prompt:
            out.insert(0, row)
        else:
            out.append(row)
    return out


def pending_jobs_path(phase1_dir: Path) -> Path:
    return phase1_dir / "pending_jobs.json"


def load_pending_jobs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_pending_jobs(path: Path, jobs: list[dict[str, str]]) -> None:
    write_json(path, jobs)


def is_pid_running(pid: str) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def tail_log(path: Path, max_chars: int = 800) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def get_job_status(job: dict[str, str]) -> tuple[str, str]:
    output_file_raw = job.get("output_file", "").strip()
    log_file_raw = job.get("log_file", "").strip()
    output_file = Path(output_file_raw) if output_file_raw else None
    log_file = Path(log_file_raw) if log_file_raw else None
    pid = job.get("pid", "")

    if output_file is not None and output_file.exists():
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                return "finished", f"output rows={len(data)}"
            return "finished", "output file exists"
        except json.JSONDecodeError:
            return "failed", "output file is invalid JSON"

    if is_pid_running(pid):
        return "running", f"pid={pid}"

    # Process is not running and output does not exist -> likely failed.
    log_tail = tail_log(log_file).strip() if log_file is not None else ""
    if log_tail:
        last_line = log_tail.splitlines()[-1]
        return "failed", f"last_log='{last_line[:160]}'"
    return "failed", "no output and process not running"


def print_job_statuses(jobs_file: Path) -> None:
    jobs = load_pending_jobs(jobs_file)
    if not jobs:
        print("No pending jobs found.")
        return
    print(f"Pending jobs: {len(jobs)}")
    for job in jobs:
        job_id = job.get("job_id", "unknown")
        status, detail = get_job_status(job)
        print(f"- {job_id}: {status} ({detail})")


def append_completed_jobs(
    dataset_path: Path,
    dataset: list[dict[str, Any]],
    jobs_file: Path,
    root_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    jobs = load_pending_jobs(jobs_file)
    if not jobs:
        return dataset, 0

    existing_keys = {
        (str(r.get("query_id", "")), int(r.get("iter", -1))) for r in dataset if isinstance(r, dict)
    }

    remaining_jobs: list[dict[str, str]] = []
    appended = 0
    refresh_targets: set[tuple[int, int, tuple[str, ...], Path]] = set()
    for job in jobs:
        out_path = Path(job.get("output_file", ""))
        if not out_path.exists():
            remaining_jobs.append(job)
            continue

        try:
            generated = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(generated, list):
                generated = []
        except json.JSONDecodeError:
            remaining_jobs.append(job)
            continue

        for row in generated:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("query_id", "")), int(row.get("iter", -1)))
            if key in existing_keys:
                continue
            dataset.append(row)
            existing_keys.add(key)
            appended += 1

        room_w = int(job.get("room_w", "0") or "0")
        room_h = int(job.get("room_h", "0") or "0")
        furniture_raw = str(job.get("furniture", "")).strip()
        filtered_items_out_raw = str(job.get("filtered_items_out", "")).strip()
        if room_w > 0 and room_h > 0 and furniture_raw and filtered_items_out_raw:
            furniture = tuple(
                normalize_label(item)
                for item in furniture_raw.split(",")
                if normalize_label(item)
            )
            if furniture:
                out_path = Path(filtered_items_out_raw)
                if not out_path.is_absolute():
                    out_path = root_dir / out_path
                refresh_targets.add((room_w, room_h, furniture, out_path))

    if appended > 0:
        write_json(dataset_path, dataset)
        for room_w, room_h, furniture_tuple, out_path in sorted(
            refresh_targets, key=lambda t: str(t[3])
        ):
            filtered_rows = filter_dataset(dataset, room_w, room_h, list(furniture_tuple))
            payload = build_filtered_items_payload(filtered_rows, room_w, room_h, list(furniture_tuple))
            write_json(out_path, payload)
            print(f"Refreshed filtered items artifact from completed jobs: {out_path}")
    save_pending_jobs(jobs_file, remaining_jobs)
    return dataset, appended


def launch_background_generation(
    layoutgpt_dir: Path,
    phase1_dir: Path,
    prompt_text: str,
    room_w: int,
    room_h: int,
    furniture: list[str],
    filtered_items_out: Path,
    dry_run: bool,
) -> None:
    run_script = layoutgpt_dir / "run_layoutgpt_2d.py"
    if not run_script.exists():
        print(f"Cannot launch generation: missing script {run_script}")
        return

    jobs_dir = phase1_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    val_json = jobs_dir / f"{job_id}.val.json"
    output_json = jobs_dir / f"{job_id}.output.json"
    log_file = jobs_dir / f"{job_id}.log"

    query_id = f"room_{room_w}x{room_h}_{uuid.uuid4().hex[:8]}"
    write_json(val_json, [{"id": query_id, "prompt": prompt_text}])

    cmd = [
        sys.executable,
        str(run_script),
        "--llm_type",
        "ollama",
        "--ollama_model",
        "llama3.2:1b",
        "--ollama_temperature",
        "0.95",
        "--ollama_top_p",
        "0.95",
        "--ollama_num_predict",
        "512",
        "--icl_type",
        "fixed-random",
        "--setting",
        "counting",
        "--val_json",
        str(val_json),
        "--n_iter",
        "5",
        "--output_file",
        str(output_json),
        "--incremental",
    ]

    print("\nFiltered count is low. Launching background generation job...")
    print("Command:", " ".join(cmd))

    if dry_run:
        print("Dry-run mode: command not executed.")
        return

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"[{datetime.now().isoformat()}] Launching: {' '.join(cmd)}\n")
    with open(log_file, "a", encoding="utf-8") as lf:
        process = subprocess.Popen(
            cmd,
            cwd=str(layoutgpt_dir),
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    jobs_file = pending_jobs_path(phase1_dir)
    jobs = load_pending_jobs(jobs_file)
    jobs.append(
        {
            "job_id": job_id,
            "output_file": str(output_json),
            "val_file": str(val_json),
            "log_file": str(log_file),
            "pid": str(process.pid),
            "room_w": str(room_w),
            "room_h": str(room_h),
            "furniture": ",".join(furniture),
            "filtered_items_out": str(filtered_items_out),
        }
    )
    save_pending_jobs(jobs_file, jobs)

    print(f"Background job submitted: {job_id}")
    print(f"PID: {process.pid}")
    print(f"Log: {log_file}")
    print("It will be appended into dataset automatically on next run.")


def print_filtered_rows(rows: list[dict[str, Any]], show_limit: int) -> None:
    if not rows:
        print("\nNo matching rows found.")
        return
    print(f"\nFiltered rows: {len(rows)}")
    for i, row in enumerate(rows[:show_limit], start=1):
        qid = row.get("query_id", "N/A")
        labels = sorted(list(object_labels_from_row(row)))
        print(f"{i:2d}. query_id={qid} | items={labels}")
    if len(rows) > show_limit:
        print(f"... showing first {show_limit} rows.")


def build_filtered_items_payload(
    filtered_rows: list[dict[str, Any]],
    room_w: int,
    room_h: int,
    required_items: list[str],
) -> dict[str, Any]:
    item_counts: dict[str, int] = {}
    for row in filtered_rows:
        labels = object_labels_from_row(row)
        for label in labels:
            item_counts[label] = item_counts.get(label, 0) + 1

    sorted_items = sorted(item_counts.keys())
    sorted_item_counts = {k: item_counts[k] for k in sorted(item_counts.keys())}
    return {
        "meta": {
            "room": {"width": room_w, "height": room_h},
            "required_items": required_items,
            "total_filtered_rows": len(filtered_rows),
            "generated_at": datetime.now().isoformat(),
        },
        "items_list": sorted_items,
        "item_counts": sorted_item_counts,
        "rows": filtered_rows,
    }


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    phase1_dir = Path(__file__).resolve().parent
    dataset_path = args.dataset if args.dataset.is_absolute() else (root_dir / args.dataset)
    layoutgpt_dir = args.layoutgpt_dir if args.layoutgpt_dir.is_absolute() else (root_dir / args.layoutgpt_dir)
    filtered_items_out = (
        args.filtered_items_out
        if args.filtered_items_out.is_absolute()
        else (root_dir / args.filtered_items_out)
    )

    dataset = ensure_json_list(dataset_path)
    print(f"Dataset loaded: {dataset_path} | rows={len(dataset)}")
    jobs_file = pending_jobs_path(phase1_dir)
    if args.job_status:
        print_job_statuses(jobs_file)
        return
    dataset, appended_count = append_completed_jobs(dataset_path, dataset, jobs_file, root_dir)
    if appended_count > 0:
        print(f"Appended {appended_count} generated records from completed background jobs.")

    print("\n=== Phase 1 Layout Terminal Interface ===")
    room_w = read_int("Enter room width (pixels): ", min_value=32)
    room_h = read_int("Enter room height (pixels): ", min_value=32)
    furniture = choose_furniture()
    print(f"Filtering for furniture: {furniture}")

    filtered = filter_dataset(dataset, room_w, room_h, furniture)
    print(f"\nFiltered rows count: {len(filtered)}")
    filtered_payload = build_filtered_items_payload(filtered, room_w, room_h, furniture)
    write_json(filtered_items_out, filtered_payload)
    print(f"Filtered items artifact written: {filtered_items_out}")
    if args.show_results:
        print_filtered_rows(filtered, args.show_limit)
    else:
        print("Detailed row listing is hidden. Use --show_results for debugging.")

    if len(filtered) < args.min_results:
        prompt = build_prompt(room_w, room_h, furniture)
        launch_background_generation(
            layoutgpt_dir=layoutgpt_dir,
            phase1_dir=phase1_dir,
            prompt_text=prompt,
            room_w=room_w,
            room_h=room_h,
            furniture=furniture,
            filtered_items_out=filtered_items_out,
            dry_run=args.dry_run,
        )
    else:
        print(f"\nEnough results found ({len(filtered)} >= {args.min_results}). No generation needed.")


if __name__ == "__main__":
    main()
