"""
调试脚本: 检查prepare_mask_latents的行为
"""
from diffusers.pipelines import FluxFillPipeline
import torch

# 加载pipeline
print("Loading pipeline...")
pipe = FluxFillPipeline.from_pretrained(
    "/root/autodl-tmp/qyt/hfd_model",
    torch_dtype=torch.bfloat16
).to("cuda")

print("\n" + "="*60)
print("测试 prepare_mask_latents 的行为")
print("="*60)

# 创建测试图像: [B=1, C=3, H=512, W=1024] (diptych)
test_image = torch.randn(1, 3, 512, 1024, dtype=torch.bfloat16).to("cuda")
test_mask = torch.zeros(1, 1, 512, 1024, dtype=torch.bfloat16).to("cuda")
test_mask[:, :, :, 512:] = 1.0  # 右半部分mask

print(f"\n输入:")
print(f"  image shape: {test_image.shape}")
print(f"  mask shape: {test_mask.shape}")

# Preprocess
image = pipe.image_processor.preprocess(test_image, height=512, width=1024)
mask_image = pipe.mask_processor.preprocess(test_mask, height=512, width=1024)

print(f"\nPreprocess后:")
print(f"  image shape: {image.shape}")
print(f"  mask shape: {mask_image.shape}")

# Masked image
masked_image = image * (1 - mask_image)
print(f"  masked_image shape: {masked_image.shape}")

# 调用prepare_mask_latents
mask, masked_image_latents = pipe.prepare_mask_latents(
    mask_image,
    masked_image,
    batch_size=1,
    num_channels_latents=pipe.vae.config.latent_channels,
    num_images_per_prompt=1,
    height=512,
    width=1024,
    dtype=torch.bfloat16,
    device="cuda",
    generator=None,
)

print(f"\nprepare_mask_latents输出:")
print(f"  mask shape: {mask.shape}")
print(f"  masked_image_latents shape: {masked_image_latents.shape}")

# Cat together (x_cond)
x_cond = torch.cat((masked_image_latents, mask), dim=-1)
print(f"\nx_cond (cat后) shape: {x_cond.shape}")

print("\n" + "="*60)
print("预期:")
print("  如果输入是512x1024,VAE 16x下采样")
print("  latent应该是: [1, 16, 32, 64]")
print("  flatten后: [1, 2048, 16]")
print("  mask也是: [1, 2048, 1]")
print("  x_cond应该是: [1, 2048, 17]")
print("="*60)

# 额外测试: 检查VAE的实际下采样因子
print("\nVAE配置:")
print(f"  vae_scale_factor: {pipe.vae_scale_factor}")
print(f"  latent_channels: {pipe.vae.config.latent_channels}")
