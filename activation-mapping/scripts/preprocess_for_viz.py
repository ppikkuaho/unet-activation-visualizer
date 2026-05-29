#!/usr/bin/env python3
"""
Preprocess captured activation profiles into slim per-profile viz JSON + an index for the browser.

Reads the full profiles written by activation_engine.py and produces:
  1. profiles/viz/<key>.viz.json — scalar stats per block per timestep (norm/mean/std/max/log_norm),
     per-attention-head L2-norm groups (head_norms), the cross-attention output-magnitude summaries
     (attn_summaries), and the CFG cond/uncond split (cfg_split).
  2. profiles/viz/index.json — the profile registry the front-end loads first.

Profiles are listed in profiles/baseline_map.json as { "<key>": { "info": {...} } }; the
full-profile path is derived from the key, so the map carries no machine-specific paths.
"""

import json
import math
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROFILES_DIR = BASE_DIR / "profiles"
IMAGES_DIR = BASE_DIR / "images"
VIZ_DIR = PROFILES_DIR / "viz"
REGISTRY_PATH = BASE_DIR / "block_registry.json"
BASELINE_MAP_PATH = PROFILES_DIR / "baseline_map.json"

# Image/viz URLs in index.json are relative to the front-end (index.html at the repo root),
# which fetches all data under this prefix.
BASE_URL = "activation-mapping/"

HEAD_DIM = 64  # SDXL attention head dimension; channel_norms are grouped into 64-wide heads.


def load_json(path):
    with open(path) as f:
        return json.load(f)


def aggregate_head_norms(channel_norms, head_dim=HEAD_DIM):
    """Group per-channel L2 norms into per-attention-head means (320ch->5, 640ch->10, 1280ch->20)."""
    n = len(channel_norms)
    heads = n // head_dim
    if heads == 0:
        return [sum(channel_norms) / max(n, 1)]
    out = [sum(channel_norms[h * head_dim:(h + 1) * head_dim]) / head_dim for h in range(heads)]
    rem = n % head_dim
    if rem:
        out.append(sum(channel_norms[heads * head_dim:]) / rem)
    return out


def slim_profile(full_profile, registry_blocks):
    """Strip per-channel arrays to scalar stats + head-grouped norms; keep the CFG split."""
    slim = {}
    for ts, blocks in full_profile.items():
        slim[ts] = {}
        for name, s in blocks.items():
            entry = {
                "norm": s["norm"], "mean": s["mean"], "std": s["std"], "max": s["max"],
                "log_norm": math.log1p(s["norm"]), "shape": s["shape"],
                "in_registry": name in registry_blocks,
            }
            for k in ("cond_norm", "uncond_norm", "cfg_divergence"):
                if k in s:
                    entry[k] = s[k]
            if "channel_norms" in s:
                entry["head_norms"] = [round(v, 4) for v in aggregate_head_norms(s["channel_norms"])]
            slim[ts][name] = entry
    return slim


def compute_normalization(slim_data):
    """Per-timestep + global p5/p95 bounds on log_norm (percentile clip avoids outlier flattening)."""
    import numpy as np
    all_vals, per_step = [], {}
    for ts, blocks in slim_data.items():
        vals = [b["log_norm"] for b in blocks.values()]
        all_vals.extend(vals)
        a = np.array(vals)
        per_step[ts] = {"p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
                        "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}
    a = np.array(all_vals)
    glob = {"p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}
    return {"global": glob, "per_step": per_step}


def make_image_paths(key):
    """Relative image URLs the front-end loads (final image + per-step intermediates)."""
    return {
        "final": f"{BASE_URL}images/baselines/baseline_{key}.png",
        "intermediates": {str(s): f"{BASE_URL}images/intermediates/{key}/step_{s:02d}.jpg"
                          for s in range(1, 30)},
    }


def intermediates_exist(key):
    d = IMAGES_DIR / "intermediates" / key
    return d.exists() and len(list(d.glob("step_*.jpg"))) >= 29


def process_profile(key, profile_path, metadata, attn_path, registry_blocks):
    print(f"  Processing {key}...")
    full = load_json(profile_path)
    slim = slim_profile(full, registry_blocks)

    # Top-level CFG split: cond/uncond/divergence per block per step (drives the CFG view mode).
    cfg_split = {}
    for ts, blocks in full.items():
        step = {n: {"cond_norm": s["cond_norm"], "uncond_norm": s["uncond_norm"],
                    "cfg_divergence": s.get("cfg_divergence", 0)}
                for n, s in blocks.items() if "cond_norm" in s}
        if step:
            cfg_split[ts] = step

    viz = {
        "profile_key": key,
        "schema_version": 2,
        "metadata": metadata,
        "timesteps": sorted(int(k) for k in full.keys()),
        "normalization": compute_normalization(slim),
        "blocks": slim,
    }
    if attn_path and Path(attn_path).exists():
        viz["attn_summaries"] = load_json(attn_path)
    if cfg_split:
        viz["cfg_split"] = cfg_split
    return viz


def build_index(processed):
    profiles = []
    for key, viz in processed.items():
        m = viz["metadata"]
        profiles.append({
            "key": key,
            "character": m.get("char", "unknown"),
            "pose": m.get("pose", "unknown"),
            "seed": m.get("seed", 0),
            "prompt": m.get("prompt", ""),
            "negative": m.get("negative", ""),
            "lora": m.get("char_lora", ""),
            "timesteps": viz["timesteps"],
            "has_attn": "attn_summaries" in viz,
            "has_intermediates": intermediates_exist(key),
            "viz_json": f"{BASE_URL}profiles/viz/{key}.viz.json",
            "images": make_image_paths(key),
        })
    profiles.sort(key=lambda p: (p["character"], p["pose"]))
    return {"description": "UNet Activation Visualizer - profile index", "profiles": profiles}


def main():
    print("=" * 60)
    print("PREPROCESS FOR VIZ")
    print("=" * 60)
    registry = load_json(REGISTRY_PATH)
    registry_blocks = {b["id"] for b in registry["blocks"]}
    print(f"  Registry: {len(registry_blocks)} blocks")

    baseline_map = load_json(BASELINE_MAP_PATH) if BASELINE_MAP_PATH.exists() else {}
    print(f"  Baseline map: {len(baseline_map)} profiles")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    processed = {}
    for key, entry in baseline_map.items():
        profile_path = PROFILES_DIR / f"baseline_{key}.json"  # derived from key (no stored paths)
        if not profile_path.exists():
            print(f"  SKIP {key}: {profile_path.name} not found")
            continue
        attn_path = BASE_DIR / "attention" / key / "attn_summary.json"
        viz = process_profile(key, profile_path, entry.get("info", {}),
                              attn_path if attn_path.exists() else None, registry_blocks)
        out = VIZ_DIR / f"{key}.viz.json"
        with open(out, "w") as f:
            json.dump(viz, f)
        print(f"    Saved {out.name} ({out.stat().st_size / 1024:.1f} KB)")
        processed[key] = viz

    index = build_index(processed)
    with open(VIZ_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n  Saved index.json ({len(index['profiles'])} profiles)")
    print("=" * 60)
    print(f"Done. {len(processed)} profiles -> {VIZ_DIR}")
    print("=" * 60)
    return len(processed) > 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
