"""
调试脚本: 检查image_processor.preprocess的行为
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
print("测试 image_processor.preprocess 的行为")
print("="*60)

# 创建测试图像: [B=1, C=3, H=512, W=1024] (diptych)
test_image = torch.randn(1, 3, 512, 1024, dtype=torch.bfloat16).to("cuda")
print(f"\n输入图像 shape: {test_image.shape}")

# 测试1: 不带参数的preprocess
result1 = pipe.image_processor.preprocess(test_image)
print(f"preprocess() 无参数 -> shape: {result1.shape}")

# 测试2: 带height和width参数
result2 = pipe.image_processor.preprocess(test_image, height=512, width=1024)
print(f"preprocess(height=512, width=1024) -> shape: {result2.shape}")

# 测试3: 检查image_processor的配置
print(f"\n" + "="*60)
print("image_processor 配置:")
print("="*60)
print(f"Type: {type(pipe.image_processor)}")
if hasattr(pipe.image_processor, 'config'):
    config = pipe.image_processor.config
    print(f"Config: {config}")
elif hasattr(pipe.image_processor, '__dict__'):
    for key, value in pipe.image_processor.__dict__.items():
        print(f"  {key}: {value}")

# 测试4: 检查do_resize参数
print(f"\n" + "="*60)
print("测试 do_resize 参数:")
print("="*60)

if hasattr(pipe.image_processor, 'do_resize'):
    print(f"do_resize 默认值: {pipe.image_processor.do_resize}")

    # 尝试关闭resize
    result3 = pipe.image_processor.preprocess(test_image, height=512, width=1024)
    print(f"With do_resize={pipe.image_processor.do_resize} -> shape: {result3.shape}")

print("\n" + "="*60)
print("完成!")
print("="*60)
