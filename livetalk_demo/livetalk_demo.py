"""
LiveTalk Gradio Demo
Inputs:  image, audio, text prompt, video_duration
Output:  generated video
"""

import os
import sys
import argparse
import tempfile
import subprocess

import gradio as gr
import torch
import numpy as np

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = "/app/livetalk"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "OmniAvatar"))

# ── default config (mirrors causal_inference.yaml) ────────────────────────────
DEFAULT_CONFIG = {
    "dtype": "bf16",
    "text_encoder_path": "pretrained_checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "dit_path": "pretrained_checkpoints/LiveTalk-1.3B-V0.1/model.safetensors",
    "vae_path": "pretrained_checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    "wav2vec_path": "pretrained_checkpoints/wav2vec2",
    "image_path": "examples/inference/example1.jpg",
    "audio_path": "examples/inference/example1.wav",
    "prompt": (
        "A realistic video of a person speaking directly to the camera. "
        "The individual maintains steady eye contact with clear, expressive facial features. "
        "Their facial expressions are naturally animated and emotionally engaging, "
        "with precise lip movements perfectly synchronized to their speech."
    ),
    "output_path": "output_video.mp4",
    "video_duration": 5,          # Duration in seconds (should be 3n+2, e.g., 5, 8, 11, 14, 17, 20, ...)
    "max_hw": 720,                # 720: 480p; 1280: 720p
    "image_sizes_720": [[512, 512]],
    "fps": 16,
    "sample_rate": 16000,
    "num_steps": 4,
    "local_attn_size": 15,
    # causal inference params
    "denoising_step_list": [1000, 750, 500, 250],
    "warp_denoising_step": True,
    "num_transformer_blocks": 30,
    "frame_seq_length": 1024,
    "num_frame_per_block": 3,
    "independent_first_frame": False,
    # argparse extras
    "rank": 0,
    "world_size": 1,
    "local_rank": 0,
    "device": "cuda:0",
    "num_nodes": 1,
    "hparams": "",
    "debug": None,
    "infer": False,
    "exp_path": None,
    "input_file": None,
    "config": "configs/causal_inference.yaml",
}


def build_args(**overrides):
    """Build a fake argparse.Namespace from DEFAULT_CONFIG + caller overrides."""
    cfg = {**DEFAULT_CONFIG, **overrides}
    return argparse.Namespace(**cfg)


# ── model singleton ────────────────────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("[LiveTalk] Loading pipeline (first call) …")
    import OmniAvatar.utils.args_config as args_module
    base_args = build_args()

    # inference_example.py calls parse_args() at module level, which reads
    # sys.argv. Temporarily replace sys.argv so it won't choke on --port or
    # any other args we pass to this script.
    _saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "--config", "configs/causal_inference.yaml"]
    try:
        from scripts.inference_example import CausalInferencePipeline
    finally:
        sys.argv = _saved_argv

    # parse_args() inside the import overwrote args_module.args; restore ours.
    args_module.args = base_args

    device = torch.device("cuda:0")
    _pipeline = CausalInferencePipeline.from_pretrained(args=base_args, device=device)
    print("[LiveTalk] Pipeline ready.")
    return _pipeline


# ── inference function ─────────────────────────────────────────────────────────

def generate_video(image_input, audio_input, prompt):
    """
    image_input : str (filepath) or None
    audio_input : str (filepath) or None
    prompt      : str
    """
    import time
    video_duration = DEFAULT_DURATION

    # ── resolve paths ──────────────────────────────────────────────────────────
    image_path = image_input if image_input else os.path.join(PROJECT_ROOT, DEFAULT_CONFIG["image_path"])
    audio_path = audio_input if audio_input else os.path.join(PROJECT_ROOT, DEFAULT_CONFIG["audio_path"])
    prompt     = prompt.strip() if prompt.strip() else DEFAULT_CONFIG["prompt"]

    # ── build per-request args ─────────────────────────────────────────────────
    args = build_args(
        image_path=image_path,
        audio_path=audio_path,
        prompt=prompt,
        video_duration=video_duration,
    )

    # patch global args for any module that reads it directly
    import OmniAvatar.utils.args_config as args_module
    args_module.args = args

    pipeline = get_pipeline()
    # update pipeline args too
    pipeline.args = args

    device = torch.device("cuda:0")
    dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # num_frames = (n*fps + 4) // 4
    num_frames = (video_duration * args.fps + 4) // 4
    # align to num_frame_per_block
    rem = num_frames % args.num_frame_per_block
    if rem != 0:
        num_frames += args.num_frame_per_block - rem

    print(f"[LiveTalk] duration={video_duration}s  num_frames={num_frames}")

    noise = torch.randn([1, num_frames, 16, 64, 64], device=device, dtype=dtype)

    t_start = time.time()
    video = pipeline(
        noise=noise,
        text_prompts=prompt,
        image_path=image_path,
        audio_path=audio_path,
        initial_latent=None,
        return_latents=False,
    )

    # ── save video ─────────────────────────────────────────────────────────────
    import imageio
    video_np = (
        video.squeeze(0).permute(0, 2, 3, 1).cpu().float().numpy() * 255
    ).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_silent:
        tmp_silent_path = tmp_silent.name

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        output_path = tmp_out.name

    imageio.mimsave(
        tmp_silent_path,
        video_np,
        fps=args.fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", tmp_silent_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "96k",
            "-movflags", "+faststart",
            "-shortest",
            output_path,
        ],
        check=True,
    )
    os.remove(tmp_silent_path)

    elapsed = time.time() - t_start
    msg = f"✅ Done! (inference {elapsed:.1f}s)"
    print(f"[LiveTalk] Video saved → {output_path}  [{elapsed:.1f}s]")
    return output_path, msg


# ── Gradio UI ──────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = DEFAULT_CONFIG["prompt"]
DEFAULT_DURATION = 5  # Duration in seconds (should be 3n+2)

with gr.Blocks(title="LiveTalk Digital Human Demo") as demo:
    gr.Markdown("# 🎙️ LiveTalk Digital Human Demo")
    gr.Markdown(
        "Upload a portrait image and a speech audio clip, optionally edit the prompt, "
        "then click **Generate** to produce a talking-head video.\n\n"
        "Leave **Image** / **Audio** empty to use the built-in example files."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Portrait Image (leave empty to use example1.jpg)",
                type="filepath",
                value=None,
            )
            audio_input = gr.Audio(
                label="Speech Audio (leave empty to use example1.wav)",
                type="filepath",
                value=None,
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=4,
            )
            run_btn = gr.Button("🚀 Generate", variant="primary")

        with gr.Column(scale=1):
            video_output = gr.Video(label="Generated Video")
            status_output = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=generate_video,
        inputs=[image_input, audio_input, prompt_input],
        outputs=[video_output, status_output],
    )

    gr.Examples(
        examples=[
            [
                "/app/livetalk/examples/inference/example1.jpg",
                "/app/livetalk/examples/inference/example1.wav",
                DEFAULT_PROMPT,
            ],
        ],
        inputs=[image_input, audio_input, prompt_input],
        label="Examples",
    )

if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--port", type=int, default=8888)
    _args, _ = _parser.parse_known_args()

    # Pre-load pipeline at startup so the first request doesn't wait
    print("[LiveTalk] Pre-loading pipeline …")
    get_pipeline()
    print("[LiveTalk] Pipeline ready. Starting Gradio …")

    demo.launch(
        server_name="0.0.0.0",
        server_port=_args.port,
        share=False,
    )
