#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch reformat Android trajectory annotations.

- Reads 'all_annot.json' to discover episode_ids.
- For each ID, loads '/annotations/{episode_id}.json'.
- Reformats to the target schema and writes to '/reformatted_annotations/{episode_id}.json'.

You can optionally override paths via CLI:

    python batch_reformat.py \
        --all /path/to/all_annot.json \
        --in_dir /annotations \
        --out_dir /reformatted_annotations

Defaults:
  --all     ./all_annot.json
  --in_dir  /annotations
  --out_dir /reformatted_annotations
"""

import os
import json
import math
import argparse
from typing import Any, Dict, List, Tuple

# ----------------------------- config -----------------------------

AGENT_METADATA = {
    "producer": "OpenAI",
    "model_name": "GPT4o",
    "prompt_version": ""
}

TASK_SOURCE = "GUIOdyssey"
IN_DOMAIN = "1"   # benchmark-derived
PLATFORM = "Mobile"

# ----------------------------- helpers -----------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def to_resolution(w, h) -> str:
    if w is None or h is None:
        return ""
    return f"{w}x{h}"

def to_subdomain(device_info: Dict[str, Any]) -> str:
    name = device_info.get("device_name") or "Unknown-Device"
    product = device_info.get("product") or "unknown_product"
    release = device_info.get("release_version")
    sdk = device_info.get("sdk_version")
    extras = []
    if release is not None:
        extras.append(f"Android {release}")
    if sdk is not None:
        extras.append(f"SDK {sdk}")
    extras_str = f" ({', '.join(extras)})" if extras else ""
    return f"{name} [{product}]{extras_str}"

def env_details(device_info: Dict[str, Any]) -> Dict[str, Any]:
    w = device_info.get("w")
    h = device_info.get("h")
    release = device_info.get("release_version")
    sdk = device_info.get("sdk_version")
    return {
        "screen_resolution": to_resolution(w, h),
        "os_version": f"Android {release} (SDK {sdk})" if release or sdk else "",
        "browser_name": "",
        "browser_version": ""
    }

# ----------------------- action normalization ----------------------

_SINGLE_POINT_TYPES = {
    "click", "double_click", "right_click", "tap", "hover",
    "longpress", "long_press", "long-press", "long press"
}

def normalize_action_type(a: str) -> str:
    if not a:
        return ""
    low = a.strip().lower()

    # unify longpress variants to "longpress"
    if low.replace("_", "").replace("-", "").replace(" ", "") == "longpress":
        return "longpress"

    if low in {"click"}:
        return "click"
    if low in {"double_click", "double click"}:
        return "double_click"
    if low in {"right_click", "right click"}:
        return "right_click"
    if low in {"tap"}:
        return "tap"
    if low in {"hover"}:
        return "hover"
    if low in {"drag", "pan"}:
        return "drag"
    if low in {"scroll"}:
        return "scroll"
    if low in {"type", "input", "text"}:   # treat TEXT as typing
        return "type"
    if low in {"hotkey", "shortcut"}:
        return "hotkey"
    if low in {"answer"}:
        return "answer"
    return low.replace(" ", "_")

def _rel(x, y, w, h):
    rx = round(float(x) / float(w), 6) if w else None
    ry = round(float(y) / float(h), 6) if h else None
    return [rx, ry]

def coord_obj(x, y, w, h) -> Dict[str, Any]:
    return {
        "absolute": [int(round(float(x))), int(round(float(y)))],
        "relative": _rel(x, y, w, h)
    }

def infer_scroll_direction(dx, dy) -> str:
    # Interpret direction as content movement given the swipe vector.
    # Dominant axis wins.
    if abs(dy) >= abs(dx):
        return "down" if dy < 0 else "up"
    return "left" if dx < 0 else "right"

_KEY_MAP = {
    "KEY_HOME": ["home"],
    "KEY_BACK": ["back"],
    "KEY_ENTER": ["enter"],
    "KEY_RETURN": ["enter"],
    "KEY_TAB": ["tab"],
    "KEY_ESC": ["esc"],
    "KEY_ESCAPE": ["esc"],
    "KEY_MENU": ["menu"],
    "KEY_SEARCH": ["search"],
    "KEY_VOLUMEUP": ["volume_up"],
    "KEY_VOLUMEDOWN": ["volume_down"],
    "KEY_POWER": ["power"],
}

def looks_like_key_token(s: str) -> bool:
    t = s.strip().upper()
    return t in _KEY_MAP or t.startswith("KEY_")

def map_key_token(s: str) -> List[str]:
    t = s.strip().upper()
    return _KEY_MAP.get(t, [t.replace("KEY_", "").lower()])

def build_action(atype: str, step: Dict[str, Any], w: int, h: int) -> Dict[str, Any]:
    """Build action dict per the specified rules."""
    info = step.get("info")

    # TYPE / TEXT
    if atype == "type":
        # prefer info if it's a string; fallbacks if some other field holds text
        content = (
            (info if isinstance(info, str) else None)
            or step.get("content")
            or step.get("text")
            or step.get("answer")
            or ""
        )
        return {"type": "type", "content": content}

    # HOTKEY (explicit)
    if atype == "hotkey":
        keys = step.get("keys") or step.get("hotkey") or []
        return {"type": "hotkey", "keys": keys if isinstance(keys, list) else [str(keys)]}

    # Collect coordinate pairs (if any)
    points: List[Tuple[float, float]] = []
    if isinstance(info, list):
        for p in info:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    x, y = float(p[0]), float(p[1])
                    points.append((x, y))
                except Exception:
                    pass

    # CLICK without coordinates but with key-like info → hotkey
    if atype == "click" and not points and isinstance(info, str) and looks_like_key_token(info):
        return {"type": "hotkey", "keys": map_key_token(info)}

    # Single-point (keep only the first)
    if atype in _SINGLE_POINT_TYPES or atype in {"click", "double_click", "right_click", "tap", "hover", "longpress"}:
        coords = []
        if points:
            x, y = points[0]
            coords = [coord_obj(x, y, w, h)]
        return {"type": normalize_action_type(atype), "coordinates": coords}

    # DRAG: two endpoints (first & last)
    if atype == "drag":
        coords = []
        if len(points) >= 1:
            coords.append(coord_obj(*points[0], w, h))
        if len(points) >= 2:
            coords.append(coord_obj(*points[-1], w, h))
        return {"type": "drag", "coordinates": coords}

    # SCROLL: two endpoints, plus direction & distance if two points present
    if atype == "scroll":
        coords = []
        action = {"type": "scroll"}
        if len(points) >= 1:
            coords.append(coord_obj(*points[0], w, h))
        if len(points) >= 2:
            coords.append(coord_obj(*points[-1], w, h))
            dx, dy = points[-1][0] - points[0][0], points[-1][1] - points[0][1]
            action["direction"] = infer_scroll_direction(dx, dy)
            action["distance_pixels"] = int(round(math.hypot(dx, dy)))
        action["coordinates"] = coords
        return action

    # ANSWER (non-coordinate)
    if atype == "answer":
        return {"type": "answer", "content": step.get("answer") or ""}

    # Fallback: if any points exist, keep first; else empty coordinates
    coords = [coord_obj(*points[0], w, h)] if points else []
    return {"type": atype, "coordinates": coords}

def build_raw_response(step: Dict[str, Any]) -> str:
    intention = step.get("intention") or ""
    lli = step.get("low_level_instruction") or ""
    action = step.get("action") or ""
    info = step.get("info")
    coord_str = ""
    if isinstance(info, list) and info and isinstance(info[0], (list, tuple)) and len(info[0]) == 2:
        coord_str = f" {tuple(info[0])}"
    parts = []
    if intention:
        parts.append(intention.strip())
    if lli:
        parts.append(lli.strip())
    if action:
        parts.append(f"{str(action).strip()}{coord_str}")
    return " | ".join(parts) if parts else ""

# ------------------------- transformation -------------------------

def transform_episode(original: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = original.get("episode_id") or "unknown_episode"
    device_info = original.get("device_info", {})
    task_info = original.get("task_info", {})

    w = device_info.get("w")
    h = device_info.get("h")

    transformed: Dict[str, Any] = {
        "trace_id": episode_id,
        "task_id": task_info.get("task") or task_info.get("meta_task") or "unknown_task_id",
        "task_source": TASK_SOURCE,
        "in_domain": IN_DOMAIN,
        "platform": PLATFORM,
        "subdomain": to_subdomain(device_info),
        "environment_details": env_details(device_info),
        "instruction": task_info.get("instruction") or "",
        "agent_metadata": dict(AGENT_METADATA),
        "trajectory": [],
        "trajectory_length": len(original.get("steps", [])),
        "orm_label": {
            "score": None,
            "binary_reward": None,
            "rationale": ""
        },
        "annotation_metadata": {
            "annotator_id": "",
            "annotation_tool_version": "",
            "timestamp": ""
        },
        "original": original  # Preserve full original JSON
    }

    steps = original.get("steps", [])
    for s in steps:
        idx = s.get("step")
        # Preserve the screenshot exactly as given
        screenshot_path = s.get("screenshot") or ""

        atype = normalize_action_type(s.get("action"))
        action = build_action(atype, s, w, h)

        step_obj = {
            "step_index": idx if isinstance(idx, int) else len(transformed["trajectory"]),
            "state": {
                "screenshot_path": screenshot_path
            },
            "raw_response": build_raw_response(s),
            "thought": s.get("low_level_instruction") or "",
            "action": action,
            "prm_label": {
                "is_error": False,
                "correction": None
            },
            "original_step": s  # keep the entire original step
        }
        transformed["trajectory"].append(step_obj)

    return transformed

# ----------------------------- main -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", default="all_annot.json", help="Path to all_annot.json (list with episode_id).")
    parser.add_argument("--in_dir", default="annotations", help="Directory containing per-episode JSONs named {episode_id}.json")
    parser.add_argument("--out_dir", default="reformatted_annotations", help="Output directory for reformatted JSONs.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load list of episodes
    all_list = load_json(args.all)
    if not isinstance(all_list, list):
        raise ValueError(f"{args.all} does not contain a list")

    os.makedirs(args.out_dir, exist_ok=True)

    # Process each episode id found in all_annot.json
    processed = 0
    errors = []
    for entry in all_list:
        eid = entry.get("episode_id")
        if not eid:
            continue
        in_path = os.path.join(args.in_dir, f"{eid}.json")
        out_path = os.path.join(args.out_dir, f"{eid}.json")

        try:
            original = load_json(in_path)
            transformed = transform_episode(original)
            save_json(out_path, transformed)
            processed += 1
        except Exception as e:
            errors.append((eid, str(e)))

    # Simple summary
    print(f"Processed: {processed}")
    if errors:
        print("Errors:")
        for eid, msg in errors:
            print(f"  {eid}: {msg}")

if __name__ == "__main__":
    main()
