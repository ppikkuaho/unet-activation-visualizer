#!/usr/bin/env python3
"""
Activation Engine — capture and causally intervene in Stable Diffusion XL U-Net activations.

Registers forward hooks on every U-Net block during a ComfyUI SDXL denoising run and records,
per block per diffusion step:
  - scalar activation statistics: L2 norm, mean, std, max
  - per-channel L2 norms (the preprocessor aggregates these into per-attention-head groups)
  - attention-block output summaries for SpatialTransformer blocks: the L2 norm of the block's
    full output (x + x_in, i.e. residual + self-attention + cross-attention + feed-forward). This
    is a coarse ACTIVITY proxy for the attention-bearing blocks — it is NOT isolated to cross-
    attention, NOT a delta, and NOT a per-token attention-weight map.
  - the conditioned/unconditioned (CFG) split of each block's activation norm.

It can also causally intervene on activations mid-generation — zero / amplify / clamp specific
channels of specific blocks at specific timesteps — to probe what each part of the U-Net controls.

Configuration (all env-overridable; model weights are NOT shipped — supply your own checkpoint):
  COMFYUI_DIR      path to a ComfyUI install                (default: ~/ComfyUI)
  ACTIVATION_CKPT  path to an SDXL .safetensors checkpoint   (default: <COMFYUI_DIR>/models/checkpoints/sd_xl_base_1.0.safetensors)
  LORA_DIR         directory of optional LoRA files          (default: <COMFYUI_DIR>/models/loras)
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

# ComfyUI is an external runtime dependency, imported via sys.path (not pip-installable here).
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))
sys.path.insert(0, COMFYUI_DIR)

import comfy.sd
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management

# Checkpoint + LoRA dir are configurable; weights are user-supplied and not in this repo.
CHECKPOINT = Path(os.environ.get(
    "ACTIVATION_CKPT",
    os.path.join(COMFYUI_DIR, "models", "checkpoints", "sd_xl_base_1.0.safetensors"),
))
LORA_DIR = Path(os.environ.get("LORA_DIR", os.path.join(COMFYUI_DIR, "models", "loras")))

BASE_DIR = Path(__file__).parent.parent
PROFILES_DIR = BASE_DIR / "profiles"
IMAGES_DIR = BASE_DIR / "images"

# SDXL generation parameters
W, H = 832, 1216
STEPS = 29
CFG = 7.0
SAMPLER = "euler_ancestral"
SCHEDULER = "normal"

# Capture every diffusion step by default (the full denoising trajectory).
DEFAULT_CAPTURE_STEPS = list(range(1, STEPS + 1))

# U-Net module classes worth hooking.
HOOKABLE_CLASSES = {
    "TimestepEmbedSequential", "ResBlock", "SpatialTransformer", "Downsample", "Upsample",
}


@dataclass
class Intervention:
    """A causal edit applied to one block's activation during generation.

    operation is one of 'zero' | 'amplify' | 'clamp'. `channels` is a list of channel
    indices or the string 'all'. `timesteps` is a list of step numbers or ['all'].
    """
    block_name: str
    channels: list
    operation: str
    magnitude: float = 0.0          # multiplier for 'amplify', bound for 'clamp'
    timesteps: list = field(default_factory=lambda: ['all'])

    def applies_at(self, step_num):
        return 'all' in self.timesteps or step_num in self.timesteps

    def to_dict(self):
        return asdict(self)


class ActivationEngine:
    """Captures per-block activation statistics and applies interventions via forward hooks."""

    def __init__(self, capture_steps=None, capture_attn_summary=False, capture_cfg_split=True):
        self.capture_steps = set(capture_steps or DEFAULT_CAPTURE_STEPS)
        self.capture_attn_summary = capture_attn_summary
        self.capture_cfg_split = capture_cfg_split
        self.hooks = []
        self.current_step = -1
        self.activations = {}        # block_name -> stats for the current step
        self.step_captures = {}      # step_str -> {block_name: stats}
        self.attn_summaries = {}     # step_str -> {block_name: attention stats}
        self.interventions = []
        self._cond_or_uncond = None  # CFG batch ordering, set by the U-Net wrapper each forward

    def _apply_interventions(self, name, out):
        """Mutate `out` in place per any matching interventions. Returns True if modified."""
        modified = False
        for iv in self.interventions:
            if iv.block_name != name or not iv.applies_at(self.current_step):
                continue
            modified = True
            ch = slice(None) if iv.channels == 'all' else iv.channels
            if iv.operation == 'zero':
                if out.dim() == 4:   out[:, ch, :, :] = 0
                elif out.dim() == 3: out[:, :, ch] = 0
                elif out.dim() == 2: out[:, ch] = 0
            elif iv.operation == 'amplify':
                if out.dim() == 4:   out[:, ch, :, :] *= iv.magnitude
                elif out.dim() == 3: out[:, :, ch] *= iv.magnitude
                elif out.dim() == 2: out[:, ch] *= iv.magnitude
            elif iv.operation == 'clamp':
                m = iv.magnitude
                if out.dim() == 4:   out[:, ch, :, :] = out[:, ch, :, :].clamp(-m, m)
                elif out.dim() == 3: out[:, :, ch] = out[:, :, ch].clamp(-m, m)
                elif out.dim() == 2: out[:, ch] = out[:, ch].clamp(-m, m)
        return modified

    def _hook_fn(self, name):
        """Forward hook: apply interventions (every step) and capture stats (on capture steps)."""
        def fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, tuple):
                out = out[0] if out else None
            if out is None or not isinstance(out, torch.Tensor):
                return

            modified = self._apply_interventions(name, out)

            if self.current_step in self.capture_steps:
                with torch.no_grad():
                    f = out.float()
                    stats = {
                        "mean": f.mean().item(),
                        "std": f.std().item(),
                        "norm": f.norm().item(),
                        "max": f.abs().max().item(),
                        "shape": list(out.shape),
                    }
                    # Per-channel L2 norms (reduced over spatial / sequence dims).
                    if out.dim() == 4:
                        stats["channel_norms"] = f.norm(dim=(2, 3)).mean(dim=0).cpu().numpy()
                    elif out.dim() == 3:
                        stats["channel_norms"] = f.norm(dim=1).mean(dim=0).cpu().numpy()
                    elif out.dim() == 2:
                        stats["channel_norms"] = f.norm(dim=0).cpu().numpy()
                    else:
                        stats["channel_norms"] = np.array([f.norm().item()])

                    # CFG split: separate the conditioned vs unconditioned activation norms.
                    if (self.capture_cfg_split and self._cond_or_uncond is not None
                            and out.shape[0] >= 2):
                        cou = self._cond_or_uncond
                        ci = cou.index(0) if 0 in cou else 0
                        ui = cou.index(1) if 1 in cou else 1
                        cond, uncond = f[ci:ci + 1], f[ui:ui + 1]
                        stats["cond_norm"] = cond.norm().item()
                        stats["uncond_norm"] = uncond.norm().item()
                        stats["cfg_divergence"] = (cond - uncond).norm().item()

                    self.activations[name] = stats

            if modified:
                return (out,) + output[1:] if isinstance(output, tuple) else out
        return fn

    def _attn_hook_fn(self, block_name):
        """Forward hook on a SpatialTransformer: record the cross-attention output magnitude."""
        def fn(module, inp, output):
            if self.current_step not in self.capture_steps or not self.capture_attn_summary:
                return
            out = output[0] if isinstance(output, tuple) else output
            if out is None or not isinstance(out, torch.Tensor):
                return
            with torch.no_grad():
                f = out.float()
                self.attn_summaries.setdefault(str(self.current_step), {})[block_name] = {
                    "output_norm": f.norm().item(),
                    "output_std": f.std().item(),
                    "has_cross_attn": True,
                }
        return fn

    def register(self, unet):
        """Hook every input/middle/output block; add an attention summary hook per transformer."""
        self.clear_hooks()
        count = 0
        for name, module in unet.named_modules():
            if not any(name.startswith(p) for p in ("input_blocks.", "middle_block.", "output_blocks.")):
                continue
            if module.__class__.__name__ not in HOOKABLE_CLASSES:
                continue
            self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
            count += 1
            if module.__class__.__name__ == "SpatialTransformer" and self.capture_attn_summary:
                self.hooks.append(module.register_forward_hook(self._attn_hook_fn(name)))
        print(f"  Registered {count} block hooks on U-Net "
              f"(attn_summary={'on' if self.capture_attn_summary else 'off'}, "
              f"cfg_split={'on' if self.capture_cfg_split else 'off'})")

    def on_step(self, step_num):
        self.current_step = step_num
        if step_num in self.capture_steps:
            self.activations = {}

    def snapshot(self, step_num):
        """Freeze the current step's activations into the profile."""
        if not self.activations:
            return
        result = {}
        for name, s in self.activations.items():
            entry = {
                "mean": s["mean"], "std": s["std"], "norm": s["norm"],
                "max": s["max"], "shape": s["shape"],
                "channel_norms": s["channel_norms"].tolist(),
            }
            for k in ("cond_norm", "uncond_norm", "cfg_divergence"):
                if k in s:
                    entry[k] = s[k]
            result[name] = entry
        self.step_captures[str(step_num)] = result

    def get_raw_profile(self):
        return dict(self.step_captures)

    def get_attn_summaries(self):
        return dict(self.attn_summaries)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        self.clear_hooks()
        self.step_captures, self.attn_summaries, self.activations = {}, {}, {}
        self._cond_or_uncond, self.current_step = None, -1


def load_model(lora_file=None, lora_weight=1.0):
    """Load the SDXL checkpoint, optionally applying a LoRA. Returns (model_patcher, clip, vae)."""
    print(f"Loading checkpoint: {CHECKPOINT.name}")
    model_patcher, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        str(CHECKPOINT), output_vae=True, output_clip=True,
        output_clipvision=False, embedding_directory=None, output_model=True,
    )
    if lora_file:
        lora_path = LORA_DIR / lora_file
        if lora_path.exists():
            print(f"Loading LoRA: {lora_file} @ {lora_weight}")
            lora = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
            model_patcher, clip = comfy.sd.load_lora_for_models(
                model_patcher, clip, lora, lora_weight, lora_weight)
        else:
            print(f"WARNING: LoRA not found: {lora_path}")
    return model_patcher, clip, vae


def encode_prompt(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def generate_with_capture(model_patcher, clip, vae, prompt, neg_prompt, seed,
                          capture_steps=None, interventions=None,
                          image_path=None, profile_path=None,
                          save_intermediates=False, intermediates_dir=None,
                          capture_attn_summary=False, attn_path=None,
                          capture_cfg_split=True):
    """Generate one image with full activation capture (and optional interventions).

    Args:
        save_intermediates: if True, cache denoised latents and batch-decode them to JPEGs after
            sampling (avoids stalling the GPU loop with a VAE decode every step).
        capture_attn_summary: if True, record cross-attention output-magnitude summaries.
        capture_cfg_split: if True, record the cond/uncond split of each block's activation norm.

    Returns: {'image_path', 'profile_path', 'profile', 'attn_summaries', 'elapsed'}.
    """
    from PIL import Image as PILImage

    engine = ActivationEngine(capture_steps=capture_steps,
                              capture_attn_summary=capture_attn_summary,
                              capture_cfg_split=capture_cfg_split)
    if interventions:
        engine.interventions = interventions

    unet = model_patcher.model.diffusion_model
    engine.register(unet)

    # Intercept the CFG cond/uncond batch ordering before the hooks fire.
    if capture_cfg_split:
        def unet_wrapper(apply_func, params):
            engine._cond_or_uncond = params.get("cond_or_uncond", [0, 1])
            return apply_func(params["input"], params["timestep"], **params.get("c", {}))
        model_patcher.set_model_unet_function_wrapper(unet_wrapper)

    positive = encode_prompt(clip, prompt)
    negative = encode_prompt(clip, neg_prompt)

    device = comfy.model_management.intermediate_device()
    latent = torch.zeros([1, 4, H // 8, W // 8], device=device)
    noise = comfy.sample.prepare_noise(latent, seed)

    step_latents = {} if save_intermediates else None

    def step_callback(step_index, denoised, x, total_steps):
        completed = step_index + 1
        if engine.current_step in engine.capture_steps:
            engine.snapshot(engine.current_step)
        if save_intermediates and denoised is not None:
            step_latents[completed] = denoised[0:1].detach().to(torch.float16).cpu()
        engine.on_step(completed + 1)

    sampler = comfy.samplers.KSampler(
        model_patcher, steps=STEPS, device=comfy.model_management.get_torch_device(),
        sampler=SAMPLER, scheduler=SCHEDULER, denoise=1.0, model_options={})

    start = time.time()
    engine.on_step(1)
    samples = sampler.sample(noise, positive, negative, cfg=CFG, latent_image=latent,
                             force_full_denoise=True, denoise_mask=None,
                             callback=step_callback, seed=seed)
    if engine.current_step in engine.capture_steps:
        engine.snapshot(engine.current_step)
    elapsed = time.time() - start

    # Final image.
    img = vae.decode(samples)
    img_np = (img[0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    if image_path:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(img_np).save(str(image_path))

    # Intermediate images: batch-decode the cached latents after sampling.
    if save_intermediates and step_latents and intermediates_dir:
        intermediates_dir = Path(intermediates_dir)
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Decoding {len(step_latents)} intermediate images...")
        dev = comfy.model_management.intermediate_device()
        for s in sorted(step_latents):
            with torch.no_grad():
                dec = vae.decode(step_latents[s].to(torch.float32).to(dev))
            dec_np = (dec[0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            PILImage.fromarray(dec_np).save(str(intermediates_dir / f"step_{s:02d}.jpg"), quality=85)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    profile = engine.get_raw_profile()
    attn = engine.get_attn_summaries() if capture_attn_summary else {}

    if profile_path:
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as fp:
            json.dump(profile, fp)
    if attn_path and attn:
        Path(attn_path).parent.mkdir(parents=True, exist_ok=True)
        with open(attn_path, "w") as fp:
            json.dump(attn, fp)

    engine.clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "image_path": str(image_path) if image_path else None,
        "profile_path": str(profile_path) if profile_path else None,
        "profile": profile, "attn_summaries": attn, "elapsed": elapsed,
    }


def load_profile(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Activation Engine")
    print(f"  ComfyUI:    {COMFYUI_DIR}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Capture:    all {STEPS} steps ({W}x{H}, cfg {CFG}, {SAMPLER}/{SCHEDULER})")
