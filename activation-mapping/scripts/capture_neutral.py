#!/usr/bin/env python3
"""
Capture example activation profiles for the standalone UNet Activation Visualizer.

No LoRA. Per profile produces:
- Full profile (29 timesteps, all blocks)
- 29 intermediate decoded images
- Cross-attention summaries
- Final image

Then writes baseline_map.json and runs preprocess_for_viz.py to build the slim
viz JSONs + index.json the browser loads.
"""

import json
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from activation_engine import (
    load_model, generate_with_capture,
    PROFILES_DIR, IMAGES_DIR,
)

BASE_DIR = Path(__file__).parent.parent
ATTN_DIR = BASE_DIR / "attention"
INTERMEDIATES_DIR = IMAGES_DIR / "intermediates"
BASELINE_MAP_PATH = PROFILES_DIR / "baseline_map.json"

NEG = ("blurry, low quality, lowres, distorted, deformed, bad anatomy, "
       "watermark, signature, text, jpeg artifacts, cropped")

PROFILES = [
    {
        "key": "alpaca_standing_s42",
        "char": "alpaca", "pose": "standing", "seed": 42,
        "prompt": ("best quality, highly detailed photograph of an alpaca "
                   "standing in a grassy mountain meadow, soft natural daylight, "
                   "sharp focus, depth of field"),
        "negative": NEG,
    },
    {
        "key": "alpaca_portrait_s42",
        "char": "alpaca", "pose": "portrait", "seed": 42,
        "prompt": ("best quality, highly detailed close-up portrait photograph of "
                   "a fluffy alpaca looking at the camera, shallow depth of field, "
                   "bokeh background, soft light"),
        "negative": NEG,
    },
]


def main():
    print("=" * 60)
    print("NEUTRAL EXAMPLE CAPTURE")
    print(f"  {len(PROFILES)} profiles, no LoRA")
    print("=" * 60)

    # No char_lora_file -> clean base checkpoint, no character LoRA applied
    model_patcher, clip, vae = load_model(None)

    results = {}
    total_start = time.time()

    for p in PROFILES:
        key = p["key"]
        profile_path = PROFILES_DIR / f"baseline_{key}.json"
        image_path = IMAGES_DIR / "baselines" / f"baseline_{key}.png"
        intermediates_dir = INTERMEDIATES_DIR / key
        attn_path = ATTN_DIR / key / "attn_summary.json"

        print(f"\n  Capturing: {key}")
        print(f"    Prompt: {p['prompt'][:80]}...")

        result = generate_with_capture(
            model_patcher, clip, vae,
            prompt=p["prompt"], neg_prompt=p["negative"], seed=p["seed"],
            capture_steps=None,  # default = all 29
            image_path=image_path, profile_path=profile_path,
            save_intermediates=True, intermediates_dir=intermediates_dir,
            capture_attn_summary=True, attn_path=attn_path,
        )

        n_inter = len(list(intermediates_dir.glob("step_*.jpg")))
        print(f"    Done in {result['elapsed']:.1f}s | "
              f"{len(result['profile'])} steps | "
              f"{len(result['attn_summaries'])} attn steps | "
              f"{n_inter} intermediates")

        # baseline_map carries only metadata; the preprocessor derives paths from the key,
        # so no machine-specific absolute paths are persisted.
        results[key] = {
            "info": {
                "char": p["char"], "pose": p["pose"], "seed": p["seed"],
                "prompt": p["prompt"], "negative": p["negative"],
                "char_lora": "",
            },
        }

    BASELINE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_MAP_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote baseline_map.json ({len(results)} profiles)")

    print("\nRunning preprocess_for_viz.py...")
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "preprocess_for_viz.py")],
        check=True,
    )

    print(f"\n{'=' * 60}")
    print(f"CAPTURE COMPLETE — {len(results)} profiles in {time.time() - total_start:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
