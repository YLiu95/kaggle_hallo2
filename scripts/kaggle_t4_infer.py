#!/usr/bin/env python3
"""
Deterministic Hallo2 inference entrypoint for Kaggle T4 GPUs.

This script:
  1. Pulls all required checkpoints from Hugging Face with the Kaggle secret token.
  2. Symlinks the downloaded cache into ./pretrained_models.
  3. Generates a short talking-head video using the bundled inference pipeline.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
import sys

from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hallo.utils.util import merge_videos
from scripts.inference_long import inference_process

# -----------------------------------------------------------------------------
# Configurable parameters (edit values only; keep inline comments short & useful)
# -----------------------------------------------------------------------------
SOURCE_IMAGE = "/kaggle/input/female-voice/square2_512.png"  # Path to a frontal 512x512 portrait.
DRIVING_AUDIO = "/kaggle/input/female-voice/4s_ref_audio.wav"  # Clean mono speech waveform (WAV).
OUTPUT_ROOT = "/kaggle/working/hallo2_outputs"  # Folder where intermediate clips and the final mp4 are stored.
MODEL_DIR = "/kaggle/working/hallo2_models"  # Shared cache for all downloaded checkpoints.
BASE_CONFIG = "configs/inference/long.yaml"  # Existing YAML config used as a template.
INFERENCE_STEPS = 20  # Diffusion steps; increase for quality, lower for speed.
GUIDANCE_SCALE = 3.5  # Classifier-free guidance strength (higher -> sharper, lower -> lip-sync).
RESOLUTION = 512  # Render width/height in pixels (square images are required).
FPS = 25  # Frames per second for the exported video.
PRECISION = "fp16"  # Choose between fp16/bf16/fp32 depending on GPU memory.
CHUNK_FRAMES = 16  # Frames per diffusion window; keep <=16 on T4 to fit memory.
POSE_WEIGHT = 1.0  # Weight for pose motion control.
FACE_WEIGHT = 1.0  # Weight for face detail control.
LIP_WEIGHT = 1.0  # Weight for lip synchronization control.
FACE_EXPAND_RATIO = 1.2  # How much to expand the detected face crop for masks.
ENABLE_SUPER_RESOLUTION = False  # Upscale pass is memory heavy – keep False on T4.
USE_AUDIO_SEGMENTS = False  # Toggle 60s chunking for very long audios; keep False for short clips.


MODEL_DIR_PATH = Path(MODEL_DIR)
OUTPUT_ROOT_PATH = Path(OUTPUT_ROOT)


def _download(
    repo_id: str,
    local_dir: Path,
    allow: list[str] | None = None,
    ignore: list[str] | None = None,
    token: str | None = None,
) -> None:
    """Wrapper around snapshot_download with useful defaults."""
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow,
        ignore_patterns=ignore,
        token=token,
    )


def prepare_models(hf_token: str) -> None:
    """Fetch all checkpoints listed in the user instructions."""
    MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)

    sd_allow = [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/config.json",
        "text_encoder/*.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    ]

    sd_root = MODEL_DIR_PATH / "stable-diffusion-v1-5"
    sd_required = sd_root / "unet" / "diffusion_pytorch_model.safetensors"
    if sd_required.exists():
        print("Stable Diffusion v1.5 already cached.")
    else:
        print("Downloading Stable Diffusion v1.5 core weights...")
        try:
            _download(
                repo_id="runwayml/stable-diffusion-v1-5",
                local_dir=sd_root,
                allow=sd_allow,
                token=hf_token,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ SD download issue: {exc}")

    vae_root = MODEL_DIR_PATH / "sd-vae-ft-mse"
    if (vae_root / "diffusion_pytorch_model.safetensors").exists():
        print("VAE already cached.")
    else:
        print("Downloading MSE-finetuned VAE...")
        _download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=vae_root,
            allow=["config.json", "*.safetensors"],
            token=hf_token,
        )

    wav_root = MODEL_DIR_PATH / "wav2vec" / "wav2vec2-base-960h"
    if (wav_root / "pytorch_model.bin").exists():
        print("wav2vec2 already cached.")
    else:
        print("Downloading wav2vec2-base-960h...")
        _download(
            repo_id="facebook/wav2vec2-base-960h",
            local_dir=MODEL_DIR_PATH / "wav2vec" / "wav2vec2-base-960h",
            allow=["config.json", "*.json", "vocab.json", "*.safetensors", "pytorch_model.bin"],
            ignore=["*tf*", "*flax*"],
            token=hf_token,
        )

    def dl_hallo_subset(patterns: list[str]) -> None:
        _download(
            repo_id="fudan-generative-ai/hallo2",
            local_dir=MODEL_DIR_PATH,
            allow=patterns,
            ignore=["*.md", "*.txt", ".git*"],
            token=hf_token,
        )

    if not (MODEL_DIR_PATH / "hallo2" / "net.pth").exists():
        print("Downloading Hallo2 audio checkpoints...")
        dl_hallo_subset(["hallo2/net.pth"])
    if not (MODEL_DIR_PATH / "motion_module" / "mm_sd_v15_v2.ckpt").exists():
        print("Downloading motion module weights...")
        dl_hallo_subset(["motion_module/mm_sd_v15_v2.ckpt"])
    if not (MODEL_DIR_PATH / "face_analysis" / "models" / "scrfd_10g_bnkps.onnx").exists():
        print("Downloading face analysis package...")
        dl_hallo_subset(["face_analysis/models/*"])
    if not (MODEL_DIR_PATH / "facelib" / "detection_mobilenet0.25_Final.pth").exists():
        print("Downloading facelib detectors...")
        dl_hallo_subset(["facelib/*"])
    if not (MODEL_DIR_PATH / "realesrgan" / "RealESRGAN_x2plus.pth").exists():
        print("Downloading RealESRGAN weights...")
        dl_hallo_subset(["realesrgan/RealESRGAN_x2plus.pth"])

    pretrained_link = REPO_ROOT / "pretrained_models"
    if pretrained_link.exists() or pretrained_link.is_symlink():
        if pretrained_link.is_symlink() or pretrained_link.is_file():
            pretrained_link.unlink()
        else:
            shutil.rmtree(pretrained_link)
    pretrained_link.symlink_to(MODEL_DIR_PATH, target_is_directory=True)
    print(f"Symlinked {pretrained_link} -> {MODEL_DIR_PATH}")


def build_config() -> Path:
    """Create a temporary config file with Kaggle-specific overrides."""
    cfg = OmegaConf.load(REPO_ROOT / BASE_CONFIG)
    cfg.source_image = SOURCE_IMAGE
    cfg.driving_audio = DRIVING_AUDIO
    cfg.save_path = str(OUTPUT_ROOT_PATH)
    cfg.cache_path = str(OUTPUT_ROOT_PATH / ".cache")
    cfg.weight_dtype = PRECISION
    cfg.inference_steps = INFERENCE_STEPS
    cfg.cfg_scale = GUIDANCE_SCALE
    cfg.face_expand_ratio = FACE_EXPAND_RATIO
    cfg.pose_weight = POSE_WEIGHT
    cfg.face_weight = FACE_WEIGHT
    cfg.lip_weight = LIP_WEIGHT
    cfg.use_cut = USE_AUDIO_SEGMENTS
    cfg.enable_super_resolution = ENABLE_SUPER_RESOLUTION

    cfg.data.n_sample_frames = CHUNK_FRAMES
    cfg.data.source_image.width = RESOLUTION
    cfg.data.source_image.height = RESOLUTION
    cfg.data.export_video.fps = FPS

    cfg.base_model_path = str(REPO_ROOT / "pretrained_models/stable-diffusion-v1-5")
    cfg.motion_module_path = str(REPO_ROOT / "pretrained_models/motion_module/mm_sd_v15_v2.ckpt")
    cfg.vae.model_path = str(REPO_ROOT / "pretrained_models/sd-vae-ft-mse")
    cfg.wav2vec.model_path = str(REPO_ROOT / "pretrained_models/wav2vec/wav2vec2-base-960h")
    cfg.audio_ckpt_dir = str(REPO_ROOT / "pretrained_models/hallo2")

    tmp_dir = Path(tempfile.mkdtemp(prefix="hallo2_cfg_"))
    cfg_path = tmp_dir / "kaggle_long.yaml"
    OmegaConf.save(cfg, cfg_path)
    return cfg_path


def main() -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("⚠️ CUDA device visibility not set; Kaggle should expose T4 GPUs automatically.")

    token_client = UserSecretsClient()
    hf_token = token_client.get_secret("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing Hugging Face token. Set the HF_TOKEN secret in Kaggle first.")

    prepare_models(hf_token)
    OUTPUT_ROOT_PATH.mkdir(parents=True, exist_ok=True)

    cfg_path = build_config()

    args = argparse.Namespace(
        config=str(cfg_path),
        source_image=None,
        driving_audio=None,
        pose_weight=None,
        face_weight=None,
        lip_weight=None,
        face_expand_ratio=None,
        audio_ckpt_dir=None,
    )

    seg_dir = inference_process(args)
    final_video = Path(seg_dir).parent / "merge_video.mp4"
    merge_videos(seg_dir, str(final_video))
    print(f"All done! Final video saved to: {final_video}")


if __name__ == "__main__":
    main()
