"""Build a tiny, randomly initialised LTX-2.x image-to-video checkpoint (same classes, toy sizes) so the real engine
code can run end to end on a CPU in seconds. Component sizes follow diffusers' own LTX2 fast tests
(tests/pipelines/ltx2/testing_utils.py, Apache-2.0); the text encoder is hf-internal-testing/tiny-gemma3."""
from __future__ import annotations

import json
import os

import torch
from diffusers import (AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler,
                       LTX2ImageToVideoPipeline, LTX2VideoTransformer3DModel)
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

TINY_GEMMA = "hf-internal-testing/tiny-gemma3"


def tiny_pipeline() -> LTX2ImageToVideoPipeline:
    tokenizer = AutoTokenizer.from_pretrained(TINY_GEMMA)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(TINY_GEMMA)
    hidden = text_encoder.config.text_config.hidden_size
    torch.manual_seed(0)
    transformer = LTX2VideoTransformer3DModel(
        in_channels=4, out_channels=4, patch_size=1, patch_size_t=1, num_attention_heads=2, attention_head_dim=8,
        cross_attention_dim=16, audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
        audio_attention_head_dim=4, audio_cross_attention_dim=8, num_layers=2, qk_norm="rms_norm_across_heads",
        caption_channels=hidden, rope_double_precision=False, rope_type="split")
    torch.manual_seed(0)
    connectors = LTX2TextConnectors(
        caption_channels=hidden, text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
        video_connector_num_attention_heads=4, video_connector_attention_head_dim=8, video_connector_num_layers=1,
        video_connector_num_learnable_registers=None, audio_connector_num_attention_heads=4,
        audio_connector_attention_head_dim=8, audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=None, connector_rope_base_seq_len=32, rope_theta=10000.0,
        rope_double_precision=False, causal_temporal_positioning=False, rope_type="split")
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8,), decoder_block_out_channels=(8,),
        layers_per_block=(1,), decoder_layers_per_block=(1, 1), spatio_temporal_scaling=(True,),
        decoder_spatio_temporal_scaling=(True,), decoder_inject_noise=(False, False), downsample_type=("spatial",),
        upsample_residual=(False,), upsample_factor=(1,), timestep_conditioning=False, patch_size=1, patch_size_t=1,
        encoder_causal=True, decoder_causal=False)
    torch.manual_seed(0)
    audio_vae = AutoencoderKLLTX2Audio(
        base_channels=4, output_channels=2, ch_mult=(1,), num_res_blocks=1, attn_resolutions=None, in_channels=2,
        resolution=32, latent_channels=2, norm_type="pixel", causality_axis="height", dropout=0.0,
        mid_block_add_attention=False, sample_rate=16000, mel_hop_length=160, is_causal=True, mel_bins=8)
    torch.manual_seed(0)
    vocoder = LTX2Vocoder(
        in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins, hidden_channels=32, out_channels=2,
        upsample_kernel_sizes=[4, 4], upsample_factors=[2, 2], resnet_kernel_sizes=[3], resnet_dilations=[[1, 3, 5]],
        leaky_relu_negative_slope=0.1, output_sampling_rate=16000)
    return LTX2ImageToVideoPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, audio_vae=audio_vae,
                                    text_encoder=text_encoder, tokenizer=tokenizer, connectors=connectors,
                                    transformer=transformer, vocoder=vocoder)


def build_tiny_model_dir(path: str, marker: bool = True) -> str:
    """Lay out a checkpoint like Lightricks/LTX-2.5-Diffusers: pipeline components + latent_upsampler/."""
    tiny_pipeline().save_pretrained(path)
    torch.manual_seed(0)
    LTX2LatentUpsamplerModel(in_channels=4, mid_channels=32, num_blocks_per_stage=1).save_pretrained(
        os.path.join(path, "latent_upsampler"))
    if marker:
        with open(os.path.join(path, "prepared.json"), "w") as f:
            json.dump({"tiny": True}, f)
    return path
