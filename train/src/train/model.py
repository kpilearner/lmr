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
from ..flux.semantic_cross_attention import SemanticConditioningAdapter


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
        self.semantic_cross_attn = None

        if use_semantic:
            semantic_mode = self.model_config.get('semantic_mode', 'cross_attention')
            print(f'[INFO] Semantic conditioning ENABLED')
            print(f'[INFO] Semantic mode: {semantic_mode}')

            if semantic_mode == 'cross_attention':
                # Cross-attention based semantic injection
                num_layers = self.model_config.get('semantic_num_layers', 1)
                num_heads = self.model_config.get('semantic_num_heads', 8)

                # FLUX uses 64-dim packed latents
                self.semantic_cross_attn = SemanticConditioningAdapter(
                    dim=64,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=0.0
                )
                print(f'[INFO] Using Cross-Attention with {num_layers} layers, {num_heads} heads')

            elif semantic_mode == 'pixel_fusion':
                # Legacy pixel-level fusion (for comparison)
                fusion_method = self.model_config.get('semantic_fusion_method', 'fixed')
                if fusion_method == 'learnable':
                    self.semantic_weight = nn.Parameter(torch.tensor(0.5, dtype=dtype))
                    print(f'[INFO] Using LEARNABLE pixel fusion weight (init=0.5)')
                elif fusion_method == 'fixed':
                    alpha = self.model_config.get('semantic_weight', 0.5)
                    print(f'[INFO] Using FIXED pixel fusion weight: {alpha}')
            else:
                raise ValueError(f'Unknown semantic_mode: {semantic_mode}. Use "cross_attention" or "pixel_fusion".')

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

        # Add semantic cross-attention parameters if exists
        if self.semantic_cross_attn is not None:
            self.trainable_params.extend(list(self.semantic_cross_attn.parameters()))
            print(f'[INFO] Added {sum(p.numel() for p in self.semantic_cross_attn.parameters())} semantic cross-attention parameters to optimizer')

        # Add learnable semantic weight to optimizer if exists (for pixel fusion mode)
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
        semantic_mode = self.model_config.get('semantic_mode', 'cross_attention')
        semantic_tokens = None
        semantic_img = None  # Store semantic image for later encoding

        # Debug flag for first few steps
        debug = not hasattr(self, '_debug_done') or self.global_step < 2
        if debug and not hasattr(self, '_debug_done'):
            self._debug_done = True

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )

            # Process input based on semantic mode
            if use_semantic and imgs.shape[-1] == self.condition_size * 3:
                # Split triptych: [visible | target_ir | semantic]
                visible_img = imgs[:, :, :, :self.condition_size]
                target_img = imgs[:, :, :, self.condition_size:self.condition_size*2]
                semantic_img = imgs[:, :, :, self.condition_size*2:self.condition_size*3]

                if semantic_mode == 'cross_attention':
                    # Cross-attention mode: encode diptych, store semantic for later
                    if debug:
                        print(f"\n[DEBUG] ===== Cross-Attention Semantic Mode =====")
                        print(f"[DEBUG] visible_img: {visible_img.shape}")
                        print(f"[DEBUG] target_img: {target_img.shape}")
                        print(f"[DEBUG] semantic_img: {semantic_img.shape}")

                    # Encode visible+target as diptych
                    diptych = torch.cat([visible_img, target_img], dim=-1)
                    batch_size = diptych.shape[0]
                    mask_diptych = torch.zeros(
                        (batch_size, 1, self.condition_size, self.condition_size * 2),
                        dtype=diptych.dtype,
                        device=diptych.device
                    )
                    mask_diptych[:, :, :, self.condition_size:] = 1.0

                    if debug:
                        print(f"[DEBUG] diptych: {diptych.shape}")
                        print(f"[DEBUG] mask_diptych: {mask_diptych.shape}")

                    x_0, x_cond, img_ids = encode_images_fill(
                        self.flux_fill_pipe,
                        diptych,
                        mask_diptych,
                        prompt_embeds.dtype,
                        prompt_embeds.device
                    )

                    if debug:
                        print(f"[DEBUG] x_0: {x_0.shape}")
                        print(f"[DEBUG] x_cond: {x_cond.shape}")

                    # Semantic will be encoded outside no_grad to allow gradient flow
                    # semantic_img is already set above

                elif semantic_mode == 'pixel_fusion':
                    # Legacy pixel fusion mode
                    if hasattr(self, 'semantic_weight'):
                        alpha = torch.sigmoid(self.semantic_weight)
                    else:
                        alpha = self.model_config.get('semantic_weight', 0.5)

                    enhanced_visible = (1 - alpha) * visible_img + alpha * semantic_img
                    enhanced_diptych = torch.cat([enhanced_visible, target_img], dim=-1)

                    batch_size = enhanced_diptych.shape[0]
                    mask_diptych = torch.zeros(
                        (batch_size, 1, self.condition_size, self.condition_size * 2),
                        dtype=enhanced_diptych.dtype,
                        device=enhanced_diptych.device
                    )
                    mask_diptych[:, :, :, self.condition_size:] = 1.0

                    x_0, x_cond, img_ids = encode_images_fill(
                        self.flux_fill_pipe,
                        enhanced_diptych,
                        mask_diptych,
                        prompt_embeds.dtype,
                        prompt_embeds.device
                    )

            else:
                # Standard diptych processing (no semantic)
                x_0, x_cond, img_ids = encode_images_fill(
                    self.flux_fill_pipe, imgs, mask_imgs,
                    prompt_embeds.dtype, prompt_embeds.device
                )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Encode semantic image OUTSIDE no_grad for cross-attention
        # This allows gradients to flow through cross-attention
        if semantic_mode == 'cross_attention' and semantic_img is not None:
            with torch.no_grad():
                # VAE is frozen, so no_grad here
                semantic_tokens, _ = encode_images(self.flux_fill_pipe, semantic_img)

            if debug:
                print(f"[DEBUG] semantic_tokens (encoded outside no_grad): {semantic_tokens.shape}")

        # Forward pass
        hidden_states = torch.cat((x_t, x_cond), dim=2)

        # Apply semantic cross-attention if available
        if self.semantic_cross_attn is not None and semantic_tokens is not None:
            # Inject semantic guidance via cross-attention
            # hidden_states: [B, 2048, 384], semantic_tokens: [B, 2048, 64]
            # We need to apply cross-attn to the x_t part (first 64 dims)
            if debug:
                print(f"\n[DEBUG] Applying semantic cross-attention:")
                print(f"[DEBUG] x_t: {x_t.shape}")
                print(f"[DEBUG] semantic_tokens: {semantic_tokens.shape}")
                # Check scale parameter
                for i, layer in enumerate(self.semantic_cross_attn.layers):
                    print(f"[DEBUG] Layer {i} scale: {layer.scale.item():.6f}")

            x_t_enhanced = self.semantic_cross_attn(x_t, semantic_tokens)
            hidden_states = torch.cat((x_t_enhanced, x_cond), dim=2)

            if debug:
                print(f"[DEBUG] x_t_enhanced: {x_t_enhanced.shape}")
                print(f"[DEBUG] hidden_states (final): {hidden_states.shape}")
                print(f"[DEBUG] ==========================================\n")

        transformer_out = self.transformer(
            hidden_states=hidden_states,
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
