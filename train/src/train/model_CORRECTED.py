"""
修正版 model.py - 解决维度不匹配问题

主要修改:
1. 移除concat模式 (会破坏FLUX输入维度)
2. 改进weighted模式,支持可学习融合权重
3. 新增pixel_fusion模式 (在像素空间融合)
4. 添加详细的维度检查和错误提示

修改标记: 🔧 表示新增/修改的代码
"""

import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
import torch
import torch.nn as nn  # 🔧 NEW: 用于可学习参数
from peft import LoraConfig, get_peft_model_state_dict
import os
import prodigyopt

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, encode_images_fill, prepare_text_input, encode_single_image


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

        # 🔧 NEW: Initialize learnable fusion weight
        use_semantic = self.model_config.get('use_semantic_conditioning', False)
        fusion_method = self.model_config.get('semantic_fusion_method', 'weighted')

        if use_semantic:
            print(f'[INFO] Semantic conditioning ENABLED')
            print(f'[INFO] Fusion method: {fusion_method}')

            # 🔧 NEW: Validate fusion method
            if fusion_method == 'concat':
                print('[WARNING] ⚠️  concat mode is DEPRECATED due to dimension mismatch!')
                print('[WARNING] Falling back to weighted mode.')
                self.model_config['semantic_fusion_method'] = 'weighted'
                fusion_method = 'weighted'

            if fusion_method == 'learnable_weighted':
                # 🔧 NEW: Learnable weight (initialized at 0.5)
                self.semantic_weight = nn.Parameter(torch.tensor(0.5, dtype=dtype))
                print(f'[INFO] Using LEARNABLE semantic weight (init=0.5)')
            elif fusion_method == 'weighted':
                alpha = self.model_config.get('semantic_weight', 0.5)
                print(f'[INFO] Using FIXED semantic weight: {alpha}')
            elif fusion_method == 'pixel_fusion':
                alpha = self.model_config.get('semantic_weight', 0.5)
                print(f'[INFO] Using PIXEL-level fusion with weight: {alpha}')
            else:
                raise ValueError(f'Unknown semantic_fusion_method: {fusion_method}')

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
        if self.model_config.get('use_sep', False):  # 🔧 FIX: use .get() to avoid KeyError
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # 🔧 NEW: Add learnable semantic weight to optimizer
        if hasattr(self, 'semantic_weight'):
            self.trainable_params = list(self.trainable_params) + [self.semantic_weight]

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

        # 🔧 NEW: Log learnable weight
        if hasattr(self, 'semantic_weight') and batch_idx % 100 == 0:
            alpha = torch.sigmoid(self.semantic_weight).item()
            print(f'[Step {batch_idx}] Learned semantic weight: {alpha:.4f}')

        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        # Check if semantic conditioning is enabled
        use_semantic = self.model_config.get('use_semantic_conditioning', False)
        semantic_fusion = self.model_config.get('semantic_fusion_method', 'weighted')

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )

            # 🔧 MODIFIED: Semantic conditioning with corrected fusion logic
            if use_semantic and imgs.shape[-1] == self.condition_size * 3:
                # imgs shape: [B, 3, H, W*3] where W=512
                # Split: [visible | target_ir | semantic]
                visible_img = imgs[:, :, :, :self.condition_size]
                target_img = imgs[:, :, :, self.condition_size:self.condition_size*2]
                semantic_img = imgs[:, :, :, self.condition_size*2:self.condition_size*3]

                # 🔧 NEW: Pixel-level fusion option
                if semantic_fusion == 'pixel_fusion':
                    # Fuse BEFORE encoding to maintain dimension compatibility
                    if hasattr(self, 'semantic_weight'):
                        alpha = torch.sigmoid(self.semantic_weight)  # Learnable
                    else:
                        alpha = self.model_config.get('semantic_weight', 0.5)

                    # Fuse visible and semantic at pixel level
                    fused_condition = (1 - alpha) * visible_img + alpha * semantic_img

                    # Encode target image (ground truth)
                    x_0, _ = encode_single_image(
                        self.flux_fill_pipe, target_img,
                        prompt_embeds.dtype, prompt_embeds.device
                    )

                    # Encode fused condition
                    x_cond, img_ids = encode_single_image(
                        self.flux_fill_pipe, fused_condition,
                        prompt_embeds.dtype, prompt_embeds.device
                    )

                    # 🔧 NEW: Dimension check
                    assert x_0.shape == x_cond.shape, \
                        f"Shape mismatch: x_0 {x_0.shape} vs x_cond {x_cond.shape}"
                    assert x_0.shape[2] == 64, \
                        f"Channel dimension must be 64, got {x_0.shape[2]}"

                else:  # weighted or learnable_weighted
                    # Encode all three images separately
                    x_0, _ = encode_single_image(
                        self.flux_fill_pipe, target_img,
                        prompt_embeds.dtype, prompt_embeds.device
                    )

                    vis_tokens, img_ids = encode_single_image(
                        self.flux_fill_pipe, visible_img,
                        prompt_embeds.dtype, prompt_embeds.device
                    )

                    sem_tokens, _ = encode_single_image(
                        self.flux_fill_pipe, semantic_img,
                        prompt_embeds.dtype, prompt_embeds.device
                    )

                    # 🔧 MODIFIED: Feature-level weighted fusion
                    if hasattr(self, 'semantic_weight'):
                        # Learnable weight with sigmoid activation
                        alpha = torch.sigmoid(self.semantic_weight)
                    else:
                        # Fixed weight
                        alpha = self.model_config.get('semantic_weight', 0.5)

                    # Fuse at latent level
                    # 🔧 NEW: Add residual connection to preserve more information
                    use_residual = self.model_config.get('use_residual_fusion', False)
                    if use_residual:
                        # Residual fusion: keep base visible, add semantic modulation
                        x_cond = vis_tokens + alpha * (sem_tokens - vis_tokens)
                    else:
                        # Standard weighted sum
                        x_cond = (1 - alpha) * vis_tokens + alpha * sem_tokens

                    # 🔧 NEW: Dimension checks
                    assert x_0.shape == x_cond.shape == vis_tokens.shape == sem_tokens.shape, \
                        f"Shape mismatch: x_0={x_0.shape}, x_cond={x_cond.shape}, vis={vis_tokens.shape}, sem={sem_tokens.shape}"
                    assert x_cond.shape[2] == 64, \
                        f"❌ CRITICAL: Channel dimension must be 64 for FLUX, got {x_cond.shape[2]}"

            else:
                # Standard diptych processing (backward compatible)
                x_0, x_cond, img_ids = encode_images_fill(
                    self.flux_fill_pipe, imgs, mask_imgs,
                    prompt_embeds.dtype, prompt_embeds.device
                )

            # 🔧 NEW: Final dimension validation before Transformer
            expected_channels = 64  # FLUX expects 64 channels
            if x_cond.shape[2] != expected_channels:
                raise RuntimeError(
                    f"❌ CRITICAL ERROR: Condition tensor has wrong channel dimension!\n"
                    f"   Expected: [B, seq_len, {expected_channels}]\n"
                    f"   Got: {x_cond.shape}\n"
                    f"   This will cause Transformer input mismatch!"
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

        # 🔧 MODIFIED: Forward pass with dimension check
        try:
            # Expected shape: [B, seq_len, 128] after cat([x_t, x_cond], dim=2)
            hidden_states_input = torch.cat((x_t, x_cond), dim=2)

            # 🔧 NEW: Validate concatenated shape
            expected_concat_channels = 128  # 64 + 64
            if hidden_states_input.shape[2] != expected_concat_channels:
                raise RuntimeError(
                    f"❌ Concatenated hidden_states has wrong shape!\n"
                    f"   Expected: [B, seq_len, {expected_concat_channels}]\n"
                    f"   Got: {hidden_states_input.shape}"
                )

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
        except RuntimeError as e:
            print(f"[ERROR] Transformer forward failed!")
            print(f"  x_t shape: {x_t.shape}")
            print(f"  x_cond shape: {x_cond.shape}")
            print(f"  hidden_states shape: {hidden_states_input.shape}")
            print(f"  img_ids shape: {img_ids.shape}")
            raise e

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
