#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Screen reformatted trajectories to compute COMPLETE/INCOMPLETE percentages,
and randomly mark N trajectories as held out.

Default paths:
  --dir /reformatted_annotations
Outputs:
  /reformatted_annotations/heldout_ids.txt
  /reformatted_annotations/summary.json

Usage:
  python screen_and_holdout.py \
      --dir /reformatted_annotations \
      --count 200 \
      --seed 42
"""

import os
import json
import argparse
import random
from typing import Tuple, Optional, Dict, Any, List

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def detect_status_from_last_step(traj: List[Dict[str, Any]]) -> Optional[str]:
    """
    Return 'COMPLETE', 'INCOMPLETE', or None (unknown) based on the last step's action.
    Rules:
      1) If last.action.type == 'answer' and last.action.content contains 'complete'/'incomplete'
      2) If last.action.type itself is 'complete' or 'incomplete'
      3) Fallback: if last.original_step.action equals 'COMPLETE'/'INCOMPLETE'
    Matching is case-insensitive and ignores surrounding whitespace.
    """
    if not traj:
        return None
    last = traj[-1]
    action = last.get("action", {}) or {}
    a_type = (action.get("type") or "").strip().lower()
    content = action.get("content")
    content_s = (content.strip().lower() if isinstance(content, str) else "")

    # 1) answer content
    if a_type == "answer":
        if "complete" in content_s:
            # Disambiguate in case content contains both words (rare)
            if "incomplete" in content_s and "complete" in content_s:
                # prefer exact token match if possible
                if content_s == "incomplete":
                    return "INCOMPLETE"
                if content_s == "complete":
                    return "COMPLETE"
                # fallback: treat as unknown
                return None
            return "INCOMPLETE" if "incomplete" in content_s else "COMPLETE"

    # 2) action type itself
    if a_type in {"complete", "completed"}:
        return "COMPLETE"
    if a_type in {"incomplete", "fail", "failed"}:
        return "INCOMPLETE"

    # 3) fallback to original step's raw action label
    orig = last.get("original_step", {}) or {}
    orig_action = (orig.get("action") or "").strip().lower()
    if orig_action in {"complete", "completed"}:
        return "COMPLETE"
    if orig_action in {"incomplete", "fail", "failed"}:
        return "INCOMPLETE"

    # 4) Sometimes authors put the token directly in content even if type != answer
    if isinstance(content, str):
        if content_s == "complete":
            return "COMPLETE"
        if content_s == "incomplete":
            return "INCOMPLETE"

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="reformatted_annotations", help="Directory containing transformed trajectory JSON files.")
    parser.add_argument("--count", type=int, default=200, help="Number of trajectories to mark as heldout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for heldout selection.")
    args = parser.parse_args()

    in_dir = args.dir
    out_summary_path = os.path.join(in_dir, "summary.json")
    out_heldout_list = os.path.join(in_dir, "heldout_ids.txt")

    files = [f for f in os.listdir(in_dir) if f.endswith(".json")]
    total = 0
    n_complete = 0
    n_incomplete = 0
    n_unknown = 0

    # Collect (trace_id, filepath) so we can sample for heldout after scanning
    items: List[Tuple[str, str]] = []

    for fname in files:
        fpath = os.path.join(in_dir, fname)
        try:
            data = read_json(fpath)
        except Exception:
            # Skip malformed file
            continue

        # Must be a transformed file with 'trajectory'
        if not isinstance(data, dict) or "trajectory" not in data:
            continue

        trace_id = data.get("trace_id") or os.path.splitext(fname)[0]
        traj = data.get("trajectory") or []
        status = detect_status_from_last_step(traj)

        total += 1
        if status == "COMPLETE":
            n_complete += 1
        elif status == "INCOMPLETE":
            n_incomplete += 1
        else:
            n_unknown += 1

        items.append((trace_id, fpath))

    # Compute percentages over determinable traces
    determinable = n_complete + n_incomplete
    pct_complete = (n_complete / determinable * 100.0) if determinable > 0 else 0.0
    pct_incomplete = (n_incomplete / determinable * 100.0) if determinable > 0 else 0.0

    # Pick heldout
    random.seed(args.seed)
    available = [trace_id for trace_id, _ in items]
    k = min(args.count, len(available))
    heldout_ids = set(random.sample(available, k)) if k > 0 else set()

    # Mark heldout in files (held_out: 1)
    updated = 0
    for trace_id, fpath in items:
        try:
            data = read_json(fpath)
            # Ensure field exists and set value
            data["held_out"] = 1 if trace_id in heldout_ids else data.get("held_out", 0)
            write_json(fpath, data)
            updated += 1
        except Exception:
            pass

    # Save heldout list
    with open(out_heldout_list, "w", encoding="utf-8") as f:
        for tid in heldout_ids:
            f.write(f"{tid}\n")

    # Save summary
    summary = {
        "total_files": total,
        "determinable": determinable,
        "unknown": n_unknown,
        "complete": n_complete,
        "incomplete": n_incomplete,
        "percent_complete": round(pct_complete, 2),
        "percent_incomplete": round(pct_incomplete, 2),
        "heldout_requested": args.count,
        "heldout_selected": len(heldout_ids),
        "seed": args.seed
    }
    write_json(out_summary_path, summary)

    # Print concise report
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Heldout IDs written to: {out_heldout_list}")

if __name__ == "__main__":
    main()
