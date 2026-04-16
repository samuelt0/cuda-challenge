"""Dump (activation, weight) pairs from FLUX.1-schnell linear layers.

Runs one forward pass through the FLUX.1-schnell diffusion model and captures
input activations and weight matrices from specific linear layers. These are
used as benchmark data for the INT4 quantization + GEMM challenge.

Target GEMM shapes (M x N x K):
    4096 x  9216 x  3072  (attn_to_qkv)
    4096 x  3072 x  3072  (attn_to_out)
    4096 x 12288 x  3072  (ff_up)
    4096 x  3072 x 12288  (ff_down)

Output:
    flux_dump/weights.pt               -- {layer_name: [N, K] fp16}
    flux_dump/activations_1024x1024.pt -- {layer_name: [M, K] fp16}
"""

from pathlib import Path

import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OUT_DIR = Path("flux_dump")
OUT_DIR.mkdir(exist_ok=True, parents=True)

PROMPT = "A cat holding a sign that says hello world"
SEED = 0
DTYPE = torch.bfloat16

# 1024x1024 image -> M = (1024/16)^2 = 4096 image tokens
HEIGHT = 1024
WIDTH = 1024

# Target nn.Linear layers to hook (produce the 4 target (N, K) shapes).
# Layers chosen from deeper blocks where the accuracy gap between g=128 and g=64
# is larger, making correctness thresholds more meaningful.
# Q, K, V are hooked individually then merged into a single attn_to_qkv weight.
TARGET_LINEARS = {
    "single_transformer_blocks.10.attn.to_q": "_attn_to_q",   # [3072, 3072] \
    "single_transformer_blocks.10.attn.to_k": "_attn_to_k",   # [3072, 3072]  > merged -> [9216, 3072]
    "single_transformer_blocks.10.attn.to_v": "_attn_to_v",   # [3072, 3072] /
    "transformer_blocks.18.attn.to_out.0": "attn_to_out",     # [3072, 3072]
    "single_transformer_blocks.37.proj_mlp": "ff_up",          # [12288, 3072]
    "transformer_blocks.15.ff.net.2": "ff_down",               # [3072, 12288]
}

TARGET_M = 4096

captured_activations = {}
captured_weights = {}
capture_enabled = False
hooks = []


def make_linear_hook(full_name, short_name):
    """Forward pre-hook: captures input activation and weight from an nn.Linear."""
    def hook_fn(module, args):
        if not capture_enabled:
            return
        if short_name in captured_activations:
            return
        x = args[0]
        if x.dim() >= 3:
            x = x.reshape(-1, x.size(-1))
        if x.shape[0] > TARGET_M:
            x = x[:TARGET_M]
        captured_activations[short_name] = x.detach().cpu().to(torch.float16)
        captured_weights[short_name] = module.weight.data.detach().cpu().to(torch.float16)
    return hook_fn


print("Loading FLUX.1-schnell...")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
pipe.enable_model_cpu_offload()

for full_name, module in pipe.transformer.named_modules():
    if full_name in TARGET_LINEARS:
        short_name = TARGET_LINEARS[full_name]
        print(f"Hook: {full_name} -> {short_name}")
        hooks.append(module.register_forward_pre_hook(make_linear_hook(full_name, short_name)))

tag = f"{HEIGHT}x{WIDTH}"
print(f"\nRunning forward pass at {tag}...")

capture_enabled = True
generator = torch.Generator("cpu").manual_seed(SEED)
with torch.inference_mode():
    pipe(PROMPT, height=HEIGHT, width=WIDTH, guidance_scale=0.0,
         num_inference_steps=4, max_sequence_length=256, generator=generator)
capture_enabled = False

for h in hooks:
    h.remove()

# Merge Q, K, V into a single attn_to_qkv layer.
# All three share the same hidden_states input, so we use Q's activation.
captured_weights["attn_to_qkv"] = torch.cat([
    captured_weights.pop("_attn_to_q"),
    captured_weights.pop("_attn_to_k"),
    captured_weights.pop("_attn_to_v"),
], dim=0)  # [9216, 3072]
captured_activations["attn_to_qkv"] = captured_activations.pop("_attn_to_q")
del captured_activations["_attn_to_k"]
del captured_activations["_attn_to_v"]

# Some layers may receive small activations (e.g. timestep embedding, M=1).
# Use another activation with matching K as a proxy.
for name, act in list(captured_activations.items()):
    if act.shape[0] < TARGET_M:
        for donor_name, donor_act in captured_activations.items():
            if donor_name != name and donor_act.shape[1] == act.shape[1] and donor_act.shape[0] >= TARGET_M:
                captured_activations[name] = donor_act[:TARGET_M].clone()
                print(f"  {name}: proxied activation from {donor_name} (original M={act.shape[0]})")
                break

torch.save(captured_weights, OUT_DIR / "weights.pt")
torch.save(captured_activations, OUT_DIR / f"activations_{tag}.pt")

print(f"\nSaved to {OUT_DIR}/:")
print(f"  weights.pt               -- {len(captured_weights)} layers")
print(f"  activations_{tag}.pt -- {len(captured_activations)} layers")
print(f"\nTarget GEMM shapes:")
for name in ["attn_to_qkv", "attn_to_out", "ff_up", "ff_down"]:
    if name in captured_activations:
        a = captured_activations[name]
        w = captured_weights[name]
        print(f"  {name:15s}  {a.shape[0]:5d} x {w.shape[0]:5d} x {a.shape[1]:5d}")
