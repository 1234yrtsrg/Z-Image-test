"""Z-Image PyTorch Native Inference."""

import argparse
import os
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")


DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
    "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
    "Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
    "silhouetted tiered pagoda, blurred colorful distant lights."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one image with Z-Image.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt describing the image to generate.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--output", default="example.png", help="Output image path.")
    parser.add_argument("--model-path", default="ckpts/Z-Image-Turbo", help="Local model directory.")
    parser.add_argument("--repo-id", default="Tongyi-MAI/Z-Image-Turbo", help="Hugging Face repo to download if needed.")
    parser.add_argument("--height", type=int, default=1024, help="Image height. Must be divisible by 16.")
    parser.add_argument("--width", type=int, default=1024, help="Image width. Must be divisible by 16.")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="Use 0.0 for Z-Image-Turbo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--compile", action="store_true", help="Compile DiT and VAE for faster repeated runs.")
    parser.add_argument("--verify", action="store_true", help="Verify model files with checksums when available.")
    return parser.parse_args()


def select_device(torch):
    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        print("Chosen device: cuda")
        return "cuda"

    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print("Chosen device: tpu")
        return device
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            print("Chosen device: mps")
            return "mps"

    print("Chosen device: cpu")
    return "cpu"


def main():
    args = parse_args()

    import torch

    from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
    from zimage import generate

    model_path = ensure_model_weights(args.model_path, repo_id=args.repo_id, verify=args.verify)
    dtype = torch.bfloat16
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    device = select_device(torch)

    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=args.compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    start_time = time.time()
    images = generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        **components,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device).manual_seed(args.seed),
    )
    end_time = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Saved image to: {output_path}")

    # For best speed on Hopper GPUs (H100/H200/H800), set ZIMAGE_ATTENTION=_flash_3 and use --compile.


if __name__ == "__main__":
    main()
