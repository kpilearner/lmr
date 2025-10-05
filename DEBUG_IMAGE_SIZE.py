"""
调试脚本: 检查数据集中图像的实际尺寸
"""

from datasets import load_dataset
import os

# 加载数据集
dataset_path = "/root/autodl-tmp/qyt_1/dataset/pid_llvip_dataset.parquet"  # 改成你的路径
dataset = load_dataset("parquet", data_files=dataset_path, split="train")

# 检查前几个样本
print("=" * 60)
print("数据集图像尺寸检查")
print("=" * 60)

for i in range(min(5, len(dataset))):
    sample = dataset[i]

    src_img = sample.get("src_img")
    edited_img = sample.get("edited_img")
    panoptic_img = sample.get("panoptic_img")

    print(f"\n样本 {i}:")
    if src_img:
        print(f"  src_img size: {src_img.size}")  # (width, height)
    if edited_img:
        print(f"  edited_img size: {edited_img.size}")
    if panoptic_img:
        print(f"  panoptic_img size: {panoptic_img.size}")

print("\n" + "=" * 60)
print("预期尺寸: 所有图像应该是 (512, 512) 或相同的宽高比")
print("如果宽高不一致,会导致编码后的序列长度错误!")
print("=" * 60)
