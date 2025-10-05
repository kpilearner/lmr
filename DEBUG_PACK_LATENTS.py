"""
调试脚本: 检查_pack_latents和VAE编码的完整流程
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
print("测试完整的encode流程")
print("="*60)

# 创建测试图像: [B=1, C=3, H=512, W=1024] (diptych)
test_image = torch.randn(1, 3, 512, 1024, dtype=torch.bfloat16).to("cuda")

print(f"\n1. 输入图像 shape: {test_image.shape}")

# Step 1: Preprocess
images = pipe.image_processor.preprocess(test_image)
print(f"2. Preprocess后 shape: {images.shape}")

# Step 2: VAE encode
images = images.to(pipe.device).to(pipe.dtype)
latents = pipe.vae.encode(images).latent_dist.sample()
print(f"3. VAE encode后 (latents) shape: {latents.shape}")
print(f"   应该是: [B, 16, H/8, W/8] = [1, 16, 64, 128]")

# Step 3: Normalize
latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
print(f"4. Normalize后 shape: {latents.shape}")

# Step 4: Pack latents
images_tokens = pipe._pack_latents(latents, *latents.shape)
print(f"5. Pack latents后 (images_tokens) shape: {images_tokens.shape}")
print(f"   应该是: [B, H/8 * W/8, 16] = [1, 64*128, 16] = [1, 8192, 16]")

# Step 5: Prepare image IDs
images_ids = pipe._prepare_latent_image_ids(
    latents.shape[0],
    latents.shape[2],
    latents.shape[3],
    pipe.device,
    pipe.dtype,
)
print(f"6. Image IDs shape: {images_ids.shape}")

# Check mismatch
print(f"\n7. 检查维度匹配:")
print(f"   images_tokens.shape[1] = {images_tokens.shape[1]}")
print(f"   images_ids.shape[0] = {images_ids.shape[0]}")
if images_tokens.shape[1] != images_ids.shape[0]:
    print(f"   ❌ 不匹配! 会触发 //2 逻辑")
    images_ids_fixed = pipe._prepare_latent_image_ids(
        latents.shape[0],
        latents.shape[2] // 2,
        latents.shape[3] // 2,
        pipe.device,
        pipe.dtype,
    )
    print(f"   修正后 image_ids shape: {images_ids_fixed.shape}")
else:
    print(f"   ✅ 匹配!")

print("\n" + "="*60)
print("VAE scale factor: " + str(pipe.vae_scale_factor))
print("Latent channels: " + str(pipe.vae.config.latent_channels))
print("="*60)
