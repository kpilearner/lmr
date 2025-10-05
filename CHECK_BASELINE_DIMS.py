"""
检查baseline在相同输入下的维度输出
"""
import sys
sys.path.insert(0, '/root/autodl-tmp/qyt/ICEdit_raw/train')

from diffusers.pipelines import FluxFillPipeline
import torch
from src.flux.pipeline_tools import encode_images_fill
import gc

# 清理显存
torch.cuda.empty_cache()
gc.collect()

# 加载pipeline
print("Loading pipeline...")
pipe = FluxFillPipeline.from_pretrained(
    "/root/autodl-tmp/qyt/hfd_model",
    torch_dtype=torch.bfloat16
).to("cuda")

print("\n" + "="*60)
print("测试 baseline encode_images_fill 的输出")
print("="*60)

# 创建512x1024 diptych (和baseline一样)
diptych = torch.randn(4, 3, 512, 1024, dtype=torch.bfloat16).to("cuda")
mask = torch.zeros(4, 1, 512, 1024, dtype=torch.bfloat16).to("cuda")
mask[:, :, :, 512:] = 1.0

print(f"\n输入:")
print(f"  diptych shape: {diptych.shape}")
print(f"  mask shape: {mask.shape}")

# 调用encode_images_fill
x_0, x_cond, img_ids = encode_images_fill(
    pipe,
    diptych,
    mask,
    torch.bfloat16,
    "cuda"
)

print(f"\n输出:")
print(f"  x_0 shape: {x_0.shape}")
print(f"  x_cond shape: {x_cond.shape}")
print(f"  img_ids shape: {img_ids.shape}")

print("\n" + "="*60)
print("如果baseline也是2048序列长度,说明这是正确的!")
print("如果x_cond也是320维,说明mask确实有256个channel!")
print("="*60)
