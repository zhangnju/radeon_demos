"""
Gradio demo for image editing with Qwen-Image-Edit via vllm-omni.

Run:
    python image_edit_demo.py --model /models/Qwen-Image-Edit/

"""

import argparse
import os
import time

import gradio as gr
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Sample assets
# ---------------------------------------------------------------------------
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "qwen-bear.png")
SAMPLE_PROMPT = "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'"


from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ---------------------------------------------------------------------------
# Global pipeline (loaded once at startup)
# ---------------------------------------------------------------------------
omni_pipeline: Omni = None
MODEL_PATH: str = ""


def load_pipeline(model: str, vae_use_slicing: bool, vae_use_tiling: bool, enable_cpu_offload: bool) -> None:
    global omni_pipeline, MODEL_PATH
    MODEL_PATH = model
    omni_pipeline = Omni(
        model=model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        enable_cpu_offload=enable_cpu_offload,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=1,
            ring_degree=1,
            cfg_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )
    print(f"Pipeline loaded: {model}")


# ---------------------------------------------------------------------------
# Inference function (called by Gradio)
# ---------------------------------------------------------------------------
def run_edit(
    input_image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    cfg_scale: float,
    guidance_scale: float,
    seed: int,
    negative_prompt: str,
) -> tuple[Image.Image, str]:
    if input_image is None:
        return None, "Please upload an input image."
    if not prompt.strip():
        return None, "Please enter a prompt."

    img = input_image.convert("RGB").resize((512, 512), Image.LANCZOS)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)

    neg = negative_prompt.strip() if negative_prompt.strip() else None

    t0 = time.perf_counter()
    outputs = omni_pipeline.generate(
        {
            "prompt": prompt,
            "negative_prompt": neg,
            "multi_modal_data": {"image": img},
        },
        OmniDiffusionSamplingParams(
            generator=generator,
            true_cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_outputs_per_prompt=1,
        ),
    )
    elapsed = time.perf_counter() - t0

    if not outputs:
        return None, "Generation failed: no output returned."

    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        return None, "Generation failed: no request_output."

    req_out = first_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images") or not req_out.images:
        return None, "Generation failed: no images in output."

    result_img = req_out.images[0]
    status = f"Done in {elapsed:.2f}s  |  model: {os.path.basename(MODEL_PATH)}  |  steps: {num_inference_steps}  |  cfg: {cfg_scale}"
    return result_img, status


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Qwen Image Edit Demo") as demo:
        gr.Markdown("# Qwen Image Edit Demo\nUpload a 512×512 image and describe the edit you want.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    image_mode="RGB",
                    width=512,
                    height=512,
                )
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Output Image",
                    type="pil",
                    width=512,
                    height=512,
                    interactive=False,
                )

        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe the edit, e.g. 'Let this mascot dance under the moon ...'",
            lines=3,
        )

        with gr.Row():
            num_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=5, value=50)
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, step=0.5, value=4.0)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.5, value=1.0)
            seed = gr.Number(label="Seed", value=0, precision=0)

        negative_prompt = gr.Textbox(
            label="Negative Prompt (optional)",
            placeholder="e.g. blurry, bad anatomy ...",
            lines=2,
        )

        run_btn = gr.Button("Edit Image", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=run_edit,
            inputs=[input_image, prompt, num_steps, cfg_scale, guidance_scale, seed, negative_prompt],
            outputs=[output_image, status_box],
        )

        gr.Examples(
            examples=[[SAMPLE_IMAGE_PATH, SAMPLE_PROMPT, 50, 4.0, 1.0, 0, ""]],
            inputs=[input_image, prompt, num_steps, cfg_scale, guidance_scale, seed, negative_prompt],
            label="Example",
            examples_per_page=1,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for Qwen-Image-Edit")
    parser.add_argument("--model", default="/models/Qwen-Image-Edit/", help="Model path or name")
    parser.add_argument("--vae-use-slicing", action="store_true", default=True)
    parser.add_argument("--vae-use-tiling", action="store_true", default=True)
    parser.add_argument("--enable-cpu-offload", action="store_true", default=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_pipeline(
        model=args.model,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
    )
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
