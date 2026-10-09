"""Check that a Lightning LoRA in the original Wan key layout maps onto every targeted module of the FULL-SIZE
Wan 2.2 A14B expert (40 blocks, 5120 wide). Built on the meta device, so no 28 GB allocation."""
import os
import sys

from accelerate import init_empty_weights
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel

sys.path.insert(0, os.path.dirname(__file__))
from tiny_wan import fake_lightning_lora  # noqa: E402

# Wan-AI/Wan2.2-I2V-A14B-Diffusers transformer/config.json (identical for transformer_2)
A14B = dict(added_kv_proj_dim=None, attention_head_dim=128, cross_attn_norm=True, eps=1e-06, ffn_dim=13824,
            freq_dim=256, image_dim=None, in_channels=36, num_attention_heads=40, num_layers=40, out_channels=16,
            patch_size=[1, 2, 2], pos_embed_seq_len=None, qk_norm="rms_norm_across_heads", rope_max_seq_len=1024,
            text_dim=4096)


def test_every_converted_key_targets_an_existing_linear():
    with init_empty_weights():
        model = WanTransformer3DModel(**A14B)
    modules = dict(model.named_modules())
    for style in ("down_up_alpha", "A_B"):
        sd = fake_lightning_lora(layers=40, rank=1, style=style, ffn=8, inner=8)   # shapes irrelevant to naming
        conv = WanImageToVideoPipeline.lora_state_dict(sd)
        conv = conv[0] if isinstance(conv, tuple) else conv
        targets = {k[len("transformer."):].rsplit(".lora_", 1)[0] for k in conv}
        missing = sorted(t for t in targets if t not in modules)
        assert not missing, missing[:5]
        assert len(targets) == 400
