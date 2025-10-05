"""
调试脚本: 查看_pack_latents的具体实现
"""
from diffusers.pipelines import FluxFillPipeline
import torch
import inspect

# 加载pipeline
print("Loading pipeline...")
pipe = FluxFillPipeline.from_pretrained(
    "/root/autodl-tmp/qyt/hfd_model",
    torch_dtype=torch.bfloat16
).to("cuda")

print("\n" + "="*60)
print("_pack_latents 源码:")
print("="*60)

# 打印_pack_latents的源码
print(inspect.getsource(pipe._pack_latents))

print("\n" + "="*60)
print("测试不同输入的pack结果:")
print("="*60)

# Test case 1: 正方形
latents_square = torch.randn(1, 16, 64, 64)
packed_square = pipe._pack_latents(latents_square, *latents_square.shape)
print(f"\n输入: {latents_square.shape}")
print(f"输出: {packed_square.shape}")
print(f"预期如果是简单flatten: [1, 64*64, 16] = [1, 4096, 16]")
print(f"预期如果是2x2 patch: [1, 32*32, 64] = [1, 1024, 64]")

# Test case 2: 长方形 (我们的情况)
latents_rect = torch.randn(1, 16, 64, 128)
packed_rect = pipe._pack_latents(latents_rect, *latents_rect.shape)
print(f"\n输入: {latents_rect.shape}")
print(f"输出: {packed_rect.shape}")
print(f"预期如果是简单flatten: [1, 64*128, 16] = [1, 8192, 16]")
print(f"预期如果是2x2 patch: [1, 32*64, 64] = [1, 2048, 64]")
