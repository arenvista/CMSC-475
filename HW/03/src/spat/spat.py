import gc
import os
import torch
from diffusers import AutoPipelineForText2Image

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageOps


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CACHE_DIR = "hf_models_cache"

prompts = [
    "An elephant above a motorcycle.",
    "A teddy bear below a bed.",
    "A chair above a knife.",
    "A fork below a plate.",
    "A bus above a banana."
]

MODEL_SETTINGS = {
    "SD_2_1": {
        "id": "stabilityai/stable-diffusion-2-1",
        "steps": 30,
        "guidance": 7.5,
        "height": 512,
        "width": 512,
    },
    "SDXL_Base": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": 30,
        "guidance": 5.0,
        "height": 1024,
        "width": 1024,
    },
    "SD_1_5": {
        "id": "runwayml/stable-diffusion-v1-5",
        "steps": 30,
        "guidance": 7.5,
        "height": 512,
        "width": 512,
    },
    "OpenJourney_v4": {
        "id": "prompthero/openjourney-v4",
        "steps": 30,
        "guidance": 7.5,
        "height": 512,
        "width": 512,
    }
}


# Number of images per prompt per model
IMAGES_PER_PROMPT = 4
# Output directory
OUTPUT_DIR = "spatial_reasoning_results"

def load_pipeline(model_id):
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        cache_dir=CACHE_DIR,
    )
    pipe.set_progress_bar_config(disable=True)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    pipe = pipe.to(DEVICE)
    return pipe


def make_seed(model_idx, prompt_idx, image_idx):
    return 1000 + (model_idx * 10000) + (prompt_idx * 100) + image_idx


def main():
    if DEVICE == "cpu":
        print("[WARNING] No CUDA GPU detected")
    else:
        print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    error_log_path = os.path.join(OUTPUT_DIR, "errors.log")
    with open(error_log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n--- New local generation run ---\n")

    total_images = len(MODEL_SETTINGS) * len(prompts) * IMAGES_PER_PROMPT
    count = 1

    print(f"Starting local generation of {total_images} images...")

    for model_idx, (model_name, settings) in enumerate(MODEL_SETTINGS.items()):
        model_id = settings["id"]
        print(f"\n--- Loading Model: {model_name} ({model_id}) ---")

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        try:
            pipe = load_pipeline(model_id)
        except Exception as e:
            error_message = f"Failed to load model {model_name} ({model_id}). Error: {e}"
            print(f"[ERROR] {error_message}")
            with open(error_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(error_message + "\n")
            continue

        for prompt_idx, prompt in enumerate(prompts):
            print(f"  Prompt: '{prompt}'")

            for i in range(IMAGES_PER_PROMPT):
                filename = f"prompt{prompt_idx + 1}_{i + 1}.png"
                filepath = os.path.join(model_dir, filename)

                if os.path.exists(filepath):
                    print(f"    [{count}/{total_images}] Skipping (already exists): {filepath}")
                    count += 1
                    continue

                print(f"    [{count}/{total_images}] Generating: {filename}...")

                try:
                    seed = make_seed(model_idx, prompt_idx, i)
                    generator = torch.Generator().manual_seed(seed)

                    image = pipe(
                        prompt,
                        num_inference_steps=settings["steps"],
                        guidance_scale=settings["guidance"],
                        height=settings["height"],
                        width=settings["width"],
                        generator=generator,
                    ).images[0]

                    image.save(filepath)
                except Exception as e:
                    error_message = (
                        f"Failed to generate {filename} with {model_name}. "
                        f"Prompt: {prompt!r}. Error: {e}"
                    )
                    print(f"    [ERROR] {error_message}")
                    with open(error_log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(error_message + "\n")

                count += 1

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nLocal generation complete! Check the '{OUTPUT_DIR}' folder. Errors, if any, were logged to: {error_log_path}")

DEFAULT_PROMPTS = [
    "1. Elephant above\nmotorcycle",
    "2. Teddy bear below\nbed",
    "3. Chair above\nknife",
    "4. Fork below\nplate",
    "5. Bus above\nbanana",
]

def _make_contact_sheet(image_paths, columns=2, padding=8, background="white"):
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    if not images:
        return None

    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = ceil(len(images) / columns)

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + padding * (columns - 1),
            rows * cell_height + padding * (rows - 1),
        ),
        color=background,
    )

    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        tile = Image.new("RGB", (cell_width, cell_height), color=background)
        fitted = ImageOps.contain(image, (cell_width, cell_height))
        x_offset = (cell_width - fitted.width) // 2
        y_offset = (cell_height - fitted.height) // 2
        tile.paste(fitted, (x_offset, y_offset))
        sheet.paste(
            tile,
            (
                col * (cell_width + padding),
                row * (cell_height + padding),
            ),
        )

    return np.asarray(sheet)


def plot_model_prompt_image_table(
    results_dir="spatial_reasoning_results",
    prompt_labels=None,
    model_names=None,
    images_per_prompt=4,
    image_suffix=".png",
    save_path=None,
    dpi=200,
):
    sns.set_theme(style="white", context="notebook")

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    if prompt_labels is None:
        prompt_labels = DEFAULT_PROMPTS

    if model_names is None:
        model_names = sorted(path.name for path in results_path.iterdir() if path.is_dir())

    if not model_names:
        raise ValueError(f"No model directories found in {results_path}")

    nrows = len(model_names)
    ncols = len(prompt_labels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.2, nrows * 4.2),
        squeeze=False,
    )

    for row_idx, model_name in enumerate(model_names):
        model_dir = results_path / model_name

        for col_idx, prompt_label in enumerate(prompt_labels):
            ax = axes[row_idx, col_idx]
            prompt_idx = col_idx + 1

            image_paths = []
            for image_idx in range(1, images_per_prompt + 1):
                candidate = model_dir / f"prompt{prompt_idx}_{image_idx}{image_suffix}"
                if candidate.exists():
                    image_paths.append(candidate)

            if len(image_paths) < images_per_prompt:
                extras = sorted(
                    path
                    for path in model_dir.glob(f"prompt{prompt_idx}_*{image_suffix}")
                    if path not in image_paths
                )
                image_paths.extend(extras[: max(0, images_per_prompt - len(image_paths))])

            contact_sheet = _make_contact_sheet(image_paths[:images_per_prompt])

            if contact_sheet is None:
                ax.text(
                    0.5,
                    0.5,
                    "No images",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="gray",
                    transform=ax.transAxes,
                )
                ax.set_facecolor("#f7f7f7")
            else:
                ax.imshow(contact_sheet)

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#d9d9d9")
                spine.set_linewidth(0.8)

            if row_idx == 0:
                ax.set_title(prompt_label, fontsize=11, pad=10)

            if col_idx == 0:
                ax.set_ylabel(model_name, fontsize=12, labelpad=12)

    fig.suptitle("Spatial Reasoning Generations by Model and Prompt", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes

if __name__ == "__main__":
    # main()
    fig, axes = plot_model_prompt_image_table("spatial_reasoning_results", save_path="image_table.png")
    plt.show()
