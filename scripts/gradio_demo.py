'''
python scripts/gradio_demo.py 
'''

import sys
import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
    
from diffusers import FluxFillPipeline
import gradio as gr
import numpy as np
import torch
import spaces
import argparse
import random 
from diffusers import FluxFillPipeline
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


parser = argparse.ArgumentParser() 
parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio app")
parser.add_argument("--output-dir", type=str, default="gradio_results", help="Directory to save the output image")
parser.add_argument("--flux-path", type=str, default='black-forest-labs/flux.1-fill-dev', help="Path to the model")
parser.add_argument("--lora-path", type=str, default='sanaka87/ICEdit-MoE-LoRA', help="Path to the LoRA weights")
parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
args = parser.parse_args()

pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(args.lora_path)

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload() 
else:
    pipe = pipe.to("cuda")

def calculate_optimal_dimensions(image: Image.Image):
    # Extract the original dimensions
    original_width, original_height = image.size
    MIN_ASPECT_RATIO = 9 / 16
    MAX_ASPECT_RATIO = 16 / 9
    FIXED_DIMENSION = 1024

    original_aspect_ratio = original_width / original_height

    # Determine which dimension to fix
    if original_aspect_ratio > 1:  # Wider than tall
        width = FIXED_DIMENSION
        height = round(FIXED_DIMENSION / original_aspect_ratio)
    else:  # Taller than wide
        height = FIXED_DIMENSION
        width = round(FIXED_DIMENSION * original_aspect_ratio)

    # Ensure dimensions are multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Enforce aspect ratio limits
    calculated_aspect_ratio = width / height
    if calculated_aspect_ratio > MAX_ASPECT_RATIO:
        width = (height * MAX_ASPECT_RATIO // 8) * 8
    elif calculated_aspect_ratio < MIN_ASPECT_RATIO:
        height = (width / MIN_ASPECT_RATIO // 8) * 8

    # Ensure width and height remain above the minimum dimensions
    width = max(width, 576) if width == FIXED_DIMENSION else width
    height = max(height, 576) if height == FIXED_DIMENSION else height

    return width, height

@spaces.GPU
def infer(edit_images, 
          prompt, 
          seed=666, 
          randomize_seed=False, 
          width=1024, 
          height=1024, 
          guidance_scale=50, 
          num_inference_steps=28, 
          progress=gr.Progress(track_tqdm=True)
):
    
    image = edit_images["background"]
    image = image.convert("RGB")
    width, height = image.size
    image = image.resize((512, int(512 * height / width)))
    combined_image = Image.new("RGB", (width * 2, height))
    combined_image.paste(image, (0, 0)) 
    mask_array = np.zeros((height, width * 2), dtype=np.uint8)
    mask_array[:, width:] = 255 
    mask = Image.fromarray(mask_array)
    instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    image = pipe(
        prompt=instruction,
        image=combined_image,
        mask_image=mask,
        height=height,
        width=width*2,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

    w,h = image.size
    image = image.crop((w//2, 0, w, h))

    os.makedirs(args.output_dir, exist_ok=True)
        
    index = len(os.listdir(args.output_dir))
    image.save(f"{args.output_dir}/result_{index}.png")
    
    return image, seed
    
examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 1000px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# IC-Edit
A demo for [IC-Edit](https://arxiv.org/pdf/2504.20690).
More **open-source**, with **lower costs**, **faster speed** (it takes about 9 seconds to process one image), and **powerful performance**.
""")
        with gr.Row():
            with gr.Column():
                edit_image = gr.ImageEditor(
                    label='Upload and draw mask for inpainting',
                    type='pil',
                    sources=["upload", "webcam"],
                    image_mode='RGB',
                    layers=False,
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                    height=600
                )
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run")
                
            result = gr.Image(label="Result", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                    visible=False
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                    visible=False
                )
            
            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=50,
                    step=0.5,
                    value=50,
                )
  
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [edit_image, prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.launch(server_port=args.port)