import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model_state_dict
import os
import prodigyopt

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, encode_images_fill, prepare_text_input


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        condition_size: int = 512,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.condition_size = condition_size

        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise

        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')

        # Initialize semantic conditioning
        use_semantic = self.model_config.get('use_semantic_conditioning', False)
        if use_semantic:
            fusion_method = self.model_config.get('semantic_fusion_method', 'fixed')
            print(f'[INFO] Semantic conditioning ENABLED')
            print(f'[INFO] Fusion method: {fusion_method}')

            if fusion_method == 'learnable':
                # Learnable weight (initialized at 0.5)
                self.semantic_weight = nn.Parameter(torch.tensor(0.5, dtype=dtype))
                print(f'[INFO] Using LEARNABLE semantic weight (init=0.5)')
            elif fusion_method == 'fixed':
                alpha = self.model_config.get('semantic_weight', 0.5)
                print(f'[INFO] Using FIXED semantic weight: {alpha}')
            else:
                raise ValueError(f'Unknown semantic_fusion_method: {fusion_method}. Use "learnable" or "fixed".')

        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = list(self.lora_layers)

        # Add learnable semantic weight to optimizer if exists
        if hasattr(self, 'semantic_weight'):
            self.trainable_params.append(self.semantic_weight)
            print('[INFO] Added semantic_weight to optimizer')

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        # Check if semantic conditioning is enabled
        use_semantic = self.model_config.get('use_semantic_conditioning', False)

        # Debug flag: print shapes only for first few batches
        debug_shapes = not hasattr(self, '_debug_printed') or self.global_step < 3
        if debug_shapes and not hasattr(self, '_debug_printed'):
            self._debug_printed = True

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )

            # Semantic conditioning: fuse in pixel space, then construct diptych
            if use_semantic and imgs.shape[-1] == self.condition_size * 3:
                if debug_shapes:
                    print(f"\n[DEBUG] ===== Semantic Conditioning =====")
                    print(f"[DEBUG] Input triptych shape: {imgs.shape}")

                # Split triptych: [visible | target_ir | semantic]
                visible_img = imgs[:, :, :, :self.condition_size]
                target_img = imgs[:, :, :, self.condition_size:self.condition_size*2]
                semantic_img = imgs[:, :, :, self.condition_size*2:self.condition_size*3]

                if debug_shapes:
                    print(f"[DEBUG] visible_img shape: {visible_img.shape}")
                    print(f"[DEBUG] target_img shape: {target_img.shape}")
                    print(f"[DEBUG] semantic_img shape: {semantic_img.shape}")

                # Get fusion weight
                if hasattr(self, 'semantic_weight'):
                    alpha = torch.sigmoid(self.semantic_weight)
                    if debug_shapes:
                        print(f"[DEBUG] Using learnable alpha: {alpha.item():.4f}")
                else:
                    alpha = self.model_config.get('semantic_weight', 0.5)
                    if debug_shapes:
                        print(f"[DEBUG] Using fixed alpha: {alpha}")

                # Fuse visible and semantic at pixel level
                enhanced_visible = (1 - alpha) * visible_img + alpha * semantic_img

                if debug_shapes:
                    print(f"[DEBUG] enhanced_visible shape: {enhanced_visible.shape}")

                # Construct enhanced diptych: [enhanced_visible | target_ir]
                # This matches the baseline diptych format exactly!
                enhanced_diptych = torch.cat([enhanced_visible, target_img], dim=-1)

                if debug_shapes:
                    print(f"[DEBUG] enhanced_diptych shape: {enhanced_diptych.shape}")
                    print(f"[DEBUG] mask_imgs shape: {mask_imgs.shape}")

                # Use standard encode_images_fill (same as baseline)
                x_0, x_cond, img_ids = encode_images_fill(
                    self.flux_fill_pipe,
                    enhanced_diptych,
                    mask_imgs,
                    prompt_embeds.dtype,
                    prompt_embeds.device
                )

                if debug_shapes:
                    print(f"[DEBUG] After encode_images_fill:")
                    print(f"[DEBUG]   x_0 shape: {x_0.shape}")
                    print(f"[DEBUG]   x_cond shape: {x_cond.shape}")
                    print(f"[DEBUG]   img_ids shape: {img_ids.shape}")

            else:
                # Standard diptych processing (backward compatible)
                if debug_shapes:
                    print(f"\n[DEBUG] ===== Standard Diptych =====")
                    print(f"[DEBUG] Input diptych shape: {imgs.shape}")

                x_0, x_cond, img_ids = encode_images_fill(
                    self.flux_fill_pipe, imgs, mask_imgs,
                    prompt_embeds.dtype, prompt_embeds.device
                )

                if debug_shapes:
                    print(f"[DEBUG] After encode_images_fill:")
                    print(f"[DEBUG]   x_0 shape: {x_0.shape}")
                    print(f"[DEBUG]   x_cond shape: {x_cond.shape}")

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            if debug_shapes:
                print(f"[DEBUG] x_t shape: {x_t.shape}")

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        hidden_states_input = torch.cat((x_t, x_cond), dim=2)

        if debug_shapes:
            print(f"[DEBUG] hidden_states (cat of x_t and x_cond) shape: {hidden_states_input.shape}")
            print(f"[DEBUG] ===================================\n")

        transformer_out = self.transformer(
            hidden_states=hidden_states_input,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()

        # Log learnable weight periodically
        if hasattr(self, 'semantic_weight') and self.global_step % 100 == 0:
            alpha = torch.sigmoid(self.semantic_weight).item()
            print(f'[Step {self.global_step}] Learned semantic weight: {alpha:.4f}')

        return loss
