import argparse
import json
import random


def _prompt(room_w: int, room_h: int, objects: list[str], extras: str) -> str:
    if len(objects) == 0:
        object_instruction = (
            "The room is intentionally empty (0 objects). "
            "Return no object boxes for this prompt."
        )
    else:
        obj_str = ", ".join(objects[:-1]) + f", and {objects[-1]}" if len(objects) > 1 else objects[0]
        object_instruction = (
            f"Place exactly {len(objects)} movable objects with non-overlapping boxes: {obj_str}. "
        )

    return (
        f"A rectangular room interior seen from above (bird's-eye layout) on a {room_w}x{room_h} pixel canvas. "
        + object_instruction
        + "Keep all boxes fully inside the room. Leave walkspace through the center. "
        + extras
        + " Output only CSS boxes, one per line, with the given labels."
    )


def main():
    ap = argparse.ArgumentParser(description="Generate a prompt list JSON for LayoutGPT 2D room layouts.")
    ap.add_argument("--out", type=str, default="prompt_lists/room_prompts_v1.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min_size", type=int, default=192)
    ap.add_argument("--max_size", type=int, default=512)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    base_sets = [
        [],
        ["bed"],
        ["chair", "table"],
        ["bed", "cupboard", "chair", "table"],
        ["bed", "wardrobe", "desk", "chair"],
        ["sofa", "coffee-table", "tv-stand", "rug"],
        ["dining-table", "chair1", "chair2", "sideboard"],
        ["single-bed", "nightstand", "desk", "chair"],
        ["bed", "wardrobe", "desk", "chair", "table", "bookshelf"],
        ["sofa", "coffee-table", "tv-stand", "rug", "lamp", "side-table", "plant"],
    ]
    extras = [
        "Anchor the largest object along a wall with a small margin (4–16px).",
        "Keep two related objects close (gap 2–10px) and aligned.",
        "Balance the layout so no single corner is overly cluttered.",
        "Prefer axis-aligned rectangles and clean margins.",
        "Ensure at least one clear walkway from one side to the opposite side.",
    ]

    items = []
    for i in range(args.n):
        w = rnd.randrange(args.min_size, args.max_size + 1, 32)
        h = rnd.randrange(args.min_size, args.max_size + 1, 32)
        obj = rnd.choice(base_sets)
        extra = " ".join(rnd.sample(extras, k=rnd.randint(2, 3)))
        prompt = _prompt(w, h, obj, extra)
        items.append({"id": f"roomgen_{i:06d}", "prompt": prompt})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(items)} prompts to {args.out}")


if __name__ == "__main__":
    main()

