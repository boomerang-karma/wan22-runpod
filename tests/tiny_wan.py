"""Build a tiny, randomly initialised Wan 2.2 I2V checkpoint (same architecture classes, toy sizes) so the real engine
code can run end to end on a CPU in seconds. Also builds fake Lightning LoRAs in the original Wan key format."""
from __future__ import annotations

import json
import os

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

TEXT_DIM, HEADS, HEAD_DIM, FFN, LAYERS = 32, 2, 12, 32, 2
INNER = HEADS * HEAD_DIM


def tiny_transformer(seed: int) -> WanTransformer3DModel:
    torch.manual_seed(seed)
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
                                 in_channels=36, out_channels=16, text_dim=TEXT_DIM, freq_dim=256, ffn_dim=FFN,
                                 num_layers=LAYERS, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                                 rope_max_seq_len=32, image_dim=None)


def build_tiny_model_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(
        "a the cartoon hero runs with cake and balloons on sunny patio photo comes to life".split())}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
                            model_max_length=512).save_pretrained(os.path.join(path, "tokenizer"))
    torch.manual_seed(0)
    UMT5EncoderModel(UMT5Config(vocab_size=len(vocab), d_model=TEXT_DIM, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                                relative_attention_num_buckets=8)).save_pretrained(os.path.join(path, "text_encoder"))
    torch.manual_seed(0)
    AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                     temperal_downsample=[False, True, True]).save_pretrained(os.path.join(path, "vae"))
    UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0).save_pretrained(
        os.path.join(path, "scheduler"))
    tiny_transformer(1).save_pretrained(os.path.join(path, "transformer"))
    tiny_transformer(2).save_pretrained(os.path.join(path, "transformer_2"))
    with open(os.path.join(path, "model_index.json"), "w") as f:
        json.dump({"_class_name": "WanImageToVideoPipeline", "boundary_ratio": 0.9}, f)
    with open(os.path.join(path, "prepared.json"), "w") as f:
        json.dump({"tiny": True}, f)
    return path


def fake_lightning_lora(layers: int = LAYERS, rank: int = 4, style: str = "down_up_alpha", ffn: int = FFN,
                        inner: int = INNER) -> dict:
    """Original-Wan-format LoRA keys (diffusion_model.blocks.N.self_attn.q.lora_down.weight ...)."""
    down, up = ("lora_down", "lora_up") if style.startswith("down_up") else ("lora_A", "lora_B")
    g = torch.Generator().manual_seed(3)
    sd = {}
    for i in range(layers):
        shapes = {**{f"self_attn.{o}": (inner, inner) for o in "qkvo"},
                  **{f"cross_attn.{o}": (inner, inner) for o in "qkvo"},
                  "ffn.0": (inner, ffn), "ffn.2": (ffn, inner)}
        for name, (fan_in, fan_out) in shapes.items():
            base = f"diffusion_model.blocks.{i}.{name}"
            sd[f"{base}.{down}.weight"] = torch.randn(rank, fan_in, generator=g) * 0.05
            sd[f"{base}.{up}.weight"] = torch.randn(fan_out, rank, generator=g) * 0.05
            if style.endswith("alpha"):
                sd[f"{base}.alpha"] = torch.tensor(float(rank))
    return sd
