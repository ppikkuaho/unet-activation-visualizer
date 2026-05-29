# Architecture

How the visualizer captures, reduces, and renders SDXL U-Net activations.

```
ComfyUI SDXL inference
        │  forward hooks on every U-Net block
        ▼
activation_engine.py ──► full profile (per-channel, ~15 MB)  ─┐
        │                + cross-attention summaries           │
        │                + per-step decoded previews           │
        ▼                                                       │ lazy-loaded
preprocess_for_viz.py ─► slim *.viz.json (~550 KB) + index.json │ on drilldown
        │                                                       │
        ▼                                                       ▼
index.html (Three.js) ◄──────── fetch ◄──────── full profile (channel detail)
```

## 1. Capture — `activation_engine.py`

The engine drives SDXL inference through ComfyUI's `KSampler` and registers a `forward_hook` on every module under `input_blocks.` / `middle_block.` / `output_blocks.` whose class is one of `TimestepEmbedSequential`, `ResBlock`, `SpatialTransformer`, `Downsample`, `Upsample`. Across the nested block wrappers that is ~50 hook registrations, of which **32 distinct blocks emit per-step statistics** (matching the registry). Each of the **11** cross-attention `SpatialTransformer` blocks gets a second hook for the attention summary.

Per block, per captured step it records:

| Field | Meaning |
|---|---|
| `norm`, `mean`, `std`, `max` | scalar stats of the block's output tensor |
| `channel_norms` | per-channel L2 norm (reduced over spatial/sequence dims) |
| `cond_norm`, `uncond_norm`, `cfg_divergence` | the classifier-free-guidance split (see below) |

And for each `SpatialTransformer`, an `attn_summary`: `output_norm`, `output_std`, `has_cross_attn`. **`output_norm` is the L2 norm of the SpatialTransformer's full output** (`x + x_in` — the residual plus self-attention, cross-attention, and feed-forward). It is a coarse *activity* proxy for the attention-bearing blocks — **not** isolated to cross-attention, **not** a delta, and emphatically **not** the per-token `softmax(QKᵀ)` attention matrix.

**CFG split.** With classifier-free guidance the U-Net runs the conditioned and unconditioned latents together in one batch. A `set_model_unet_function_wrapper` records ComfyUI's `cond_or_uncond` ordering each forward pass, so the hook can split the batch and report `cond_norm`, `uncond_norm`, and their divergence per block — i.e. how much the prompt is pulling each block away from the unconditioned baseline.

**Interventions.** An `Intervention(block_name, channels, operation, magnitude, timesteps)` mutates a block's output in place during the forward pass — `zero`, `amplify`, or `clamp` on selected channels at selected steps. Because the edit happens inside the live forward pass, the downstream generation reflects it, which makes this a causal probe rather than a passive readout. The visualizer's UI builds these configs and exports them as JSON.

**Previews.** Denoised latents are cached on CPU during sampling and VAE-decoded to JPEGs *after* the loop, so the per-step preview images don't stall the GPU.

Everything is configurable via env vars (`COMFYUI_DIR`, `ACTIVATION_CKPT`, `LORA_DIR`); no model weights are bundled.

## 2. Reduce — `preprocess_for_viz.py`

A full profile carries a per-channel norm array for every block at every step — far too heavy for the browser. The preprocessor:

- keeps scalar stats and computes `log_norm = log1p(norm)` for display;
- bins `channel_norms` into fixed **64-channel groups** (so 320/640/1280-channel blocks → 5/10/20 bins). These are output-channel norms — 64 happens to be the SDXL attention-head width, but the values are conv output-channel magnitudes, *not* per-head attention activations, and the binning is applied to every block (including the non-attention ones);
- computes per-step and global **p5–p95 normalization bounds** (percentile-clipped so a few outlier blocks don't flatten the colour ramp);
- carries through the `attn_summaries` and the `cfg_split`.

Output: one `~550 KB *.viz.json` per profile plus an `index.json` registry. The heavy per-channel `baseline_<key>.json` (~15 MB) stays on disk and is fetched only when the user opens the channel drilldown.

### Data schema (`*.viz.json`)

```jsonc
{
  "profile_key": "alpaca_standing_s42",
  "schema_version": 2,
  "metadata": { "char", "pose", "seed", "prompt", "negative", "char_lora" },
  "timesteps": [1, …, 29],
  "normalization": { "global": {p5,p95,…}, "per_step": { "<t>": {p5,p95,…} } },
  "blocks": {                       // [step][block] scalar stats
    "<t>": { "<block>": { norm, mean, std, max, log_norm, shape,
                          head_norms[], cond_norm, uncond_norm, cfg_divergence,
                          in_registry } } },
  "attn_summaries": { "<t>": { "<block>": { output_norm, output_std, has_cross_attn } } },
  "cfg_split":      { "<t>": { "<block>": { cond_norm, uncond_norm, cfg_divergence } } }
}
```

`block_registry.json` is the canonical list of the 32 SDXL U-Net blocks with `side` (encoder/middle/decoder), `level`, channel count, resolution, and `has_cross_attn` — the source of truth for layout and for which blocks are expected to carry attention.

## 3. Render — `index.html`

A single-file ES-module Three.js app (Three.js 0.163, vendored). On load it fetches `index.json`, then the first profile's `*.viz.json`, and builds the scene:

- **Layout** — the 32 blocks are placed in a 3-D U-shape: encoder arm descending on one side, three middle blocks at the bottleneck, decoder arm ascending on the other, with skip connections bridging matching encoder/decoder levels. Tube radius narrows toward the bottleneck to echo the resolution change.
- **Block colour** — each block's `log_norm` is normalized against the per-step p5–p95 bounds and mapped along a cold→hot ramp, with emissive glow above a threshold.
- **Cross-attention arcs** — 11 curved lines from the attention blocks to a vertical "prompt field". Each arc's opacity is driven by that block's `output_norm`, min-max-normalized across blocks for the current step, so the arcs brighten where cross-attention is doing the most work.
- **Replay** — a signal pulse can travel the encoder→middle→decoder sequence, or step through diffusion timesteps, showing the order of computation.
- **Drilldown** — clicking a block opens its stats, CFG split, and per-channel norm groups (64-channel bins, lazy-loading the full profile), plus the intervention builder.
- **Compare** — load a second profile and colour each block by the activation-magnitude difference.

### Performance notes

- Core payload ≈ 10 MB (index + 2 slim profiles + preview JPEGs); the 2×15 MB full profiles load only on drilldown.
- Device pixel ratio is capped at 1.5×; geometry is instanced per block (32 draw calls — trivial for WebGL).
- No build step and no runtime CDN dependency — Three.js and the font are vendored.
