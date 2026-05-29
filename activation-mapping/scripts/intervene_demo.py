#!/usr/bin/env python3
"""
Causal-intervention demo: regenerate a profile's image while editing one block's activations,
to show what a part of the U-Net controls.

Each Intervention(block_name, channels, operation, magnitude, timesteps) is applied in place
during the live forward pass (see activation_engine.py), so the generated image reflects the edit.
Run this to produce before/after pairs; baseline images already live under images/baselines/.

  ACTIVATION_CKPT=/path/to/sdxl.safetensors python intervene_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from activation_engine import load_model, generate_with_capture, Intervention, IMAGES_DIR

PROMPT = ("best quality, highly detailed photograph of an alpaca standing in a grassy mountain "
          "meadow, soft natural daylight, sharp focus, depth of field")
NEG = ("blurry, low quality, lowres, distorted, deformed, bad anatomy, "
       "watermark, signature, text, jpeg artifacts, cropped")
SEED = 42

OUT = IMAGES_DIR / "interventions"

# (label, [interventions]) — each runs as a separate generation at the same seed/prompt.
TRIALS = [
    ("zero_middle_block_1",   [Intervention("middle_block.1",   "all", "zero")]),
    ("zero_output_blocks_0_1",[Intervention("output_blocks.0.1","all", "zero")]),
    ("zero_input_blocks_5_1", [Intervention("input_blocks.5.1", "all", "zero")]),
    ("amplify_middle_block_1",[Intervention("middle_block.1",   "all", "amplify", magnitude=2.0)]),
]


def main():
    print("=" * 60)
    print(f"INTERVENTION DEMO — {len(TRIALS)} trials, seed {SEED}")
    print("=" * 60)
    model, clip, vae = load_model(None)
    for label, interventions in TRIALS:
        print(f"\n  {label}: {[iv.to_dict() for iv in interventions]}")
        r = generate_with_capture(
            model, clip, vae, prompt=PROMPT, neg_prompt=NEG, seed=SEED,
            interventions=interventions,
            image_path=OUT / f"{label}.png",
            profile_path=None, save_intermediates=False, capture_attn_summary=False,
        )
        print(f"    saved {label}.png ({r['elapsed']:.1f}s)")
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
