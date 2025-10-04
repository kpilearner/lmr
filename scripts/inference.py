# Use the modified diffusers & peft library
import sys
import os
# workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

# if workspace_dir not in sys.path:
#     sys.path.insert(0, workspace_dir)
    
from diffusers import FluxFillPipeline

# Below is the original library
import torch
from PIL import Image
import numpy as np
import argparse
import random
    
parser = argparse.ArgumentParser() 
parser.add_argument("--image", type=str, help="Name of the image to be edited", required=True)
parser.add_argument("--instruction", type=str, help="Instruction for editing the image", required=True)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the output image")
parser.add_argument("--flux-path", type=str, default='black-forest-labs/flux.1-fill-dev', help="Path to the model")
parser.add_argument("--lora-path", type=str, default='RiverZ/normal-lora', help="Path to the LoRA weights")
parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
parser.add_argument("--semantic-image", type=str, default=None, help="Optional semantic segmentation image aligned with the visible image")


args = parser.parse_args()
pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(args.lora_path)

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload() 
else:
    pipe = pipe.to("cuda")

image = Image.open(args.image).convert("RGB")
semantic_image = None
if args.semantic_image is not None:
    semantic_image = Image.open(args.semantic_image).convert("RGB")

if image.size[0] != 512:
    print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
    new_width = 512
    scale = new_width / image.size[0]
    new_height = int(image.size[1] * scale)
    new_height = (new_height // 8) * 8  
    image = image.resize((new_width, new_height))
    print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")

instruction = args.instruction

print(f"Instruction: {instruction}")
if semantic_image is not None:
    instruction = (
        "A triptych showing a visible reference on the left and its panoptic segmentation on the right. "
        f"Modify the middle panel so that it {instruction}"
    )
else:
    instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}'

width, height = image.size
panels = [image, image.copy()]
if semantic_image is not None:
    if semantic_image.size != (width, height):
        semantic_image = semantic_image.resize((width, height))
    panels.append(semantic_image)

combined_width = width * len(panels)
combined_image = Image.new("RGB", (combined_width, height))
for idx, panel in enumerate(panels):
    combined_image.paste(panel, (idx * width, 0))

mask_array = np.zeros((height, combined_width), dtype=np.uint8)
middle_start = width
middle_end = width * 2
mask_array[:, middle_start:middle_end] = 255 
mask = Image.fromarray(mask_array)

if semantic_image is not None:
    print(
        "[semantic-debug] triptych size:", combined_image.size,
        "mask size:", mask.size,
    )

result_image = pipe(
    prompt=instruction,
    image=combined_image,
    mask_image=mask,
    height=height,
    width=combined_width,
    guidance_scale=50,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(args.seed) if args.seed is not None else None,
).images[0]

output_start = width
result_image = result_image.crop((output_start, 0, output_start + width, height))

os.makedirs(args.output_dir, exist_ok=True)

image_name = args.image.split("/")[-1]
result_image.save(os.path.join(args.output_dir, f"{image_name}"))
print(f"\033[92mResult saved as {os.path.abspath(os.path.join(args.output_dir, image_name))}\033[0m")
