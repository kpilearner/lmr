import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
# import cv2
import torch
import os
from datetime import datetime
import torchvision.transforms as T

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.to_pil = T.ToPILImage()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}",
                f"lora_{self.total_steps}",
                batch,
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        batch,
    ):
        try:
            image_tensor = batch["image"][0].detach().cpu()
            mask_tensor = batch["condition"][0].detach().cpu()
            prompt = batch["description"][0]
            condition_type = batch["condition_type"][0]
        except KeyError as exc:
            print("[callback] Missing keys in batch for sampling:", exc)
            return

        prompt = str(prompt)
        pl_module.flux_fill_pipe.transformer.eval()

        combined_image = self.to_pil(image_tensor)
        mask_np = mask_tensor.squeeze(0).clamp(0, 1).numpy()
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

        result = pl_module.flux_fill_pipe(
            prompt=prompt,
            image=combined_image,
            height=combined_image.height,
            width=combined_image.width,
            mask_image=mask_image,
            guidance_scale=50,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(666),
        ).images[0]

        mask_cols = np.where(mask_np.sum(axis=0) > 0)[0]
        if mask_cols.size > 0:
            right_start = int(mask_cols[0])
            right_end = int(mask_cols[-1]) + 1
        else:
            right_start = combined_image.width // 2
            right_end = combined_image.width
        panel_width = max(right_end - right_start, combined_image.width // 3)

        right_panel = result.crop(
            (right_start, 0, right_start + panel_width, combined_image.height)
        )

        os.makedirs(save_path, exist_ok=True)
        condition_str = str(condition_type)
        result.save(
            os.path.join(
                save_path,
                f"flux-fill-train-sample-{self.total_steps}-{condition_str}-full.png",
            )
        )
        right_panel.save(
            os.path.join(
                save_path,
                f"flux-fill-train-sample-{self.total_steps}-{condition_str}-right.png",
            )
        )

        pl_module.flux_fill_pipe.transformer.train()
        
