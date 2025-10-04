#!/bin/bash

# Visible-to-Infrared inference helper.
# Edit the variables below to match your files, then run:
#   bash scripts/run_vis2ir.sh

IMAGE_PATH="/root/autodl-tmp/test/190025.jpg"             # 可见光输入
SEMANTIC_PATH="/root/autodl-tmp/test_seg/190025_mask_rgb.png"         # 语义/全景分割图 (留空则不使用)
OUTPUT_DIR="outputs1"                             # 输出目录
INSTRUCTION="translate visible image to infrared"

# 模型与权重路径
FLUX_PATH="/root/autodl-tmp/qyt/hfd_model"
LORA_PATH="/root/autodl-tmp/qyt_1/ICEdit_lmr/ICEdit_lmr/train/runs/20251003-151749/ckpt/4000/pytorch_lora_weights.safetensors"

# 其他可选参数
SEED=42
ENABLE_CPU_OFFLOAD=false    # 改成 true 启用 --enable-model-cpu-offload

CMD=(python scripts/inference.py
  --image "$IMAGE_PATH"
  --instruction "$INSTRUCTION"
  --output-dir "$OUTPUT_DIR"
  --flux-path "$FLUX_PATH"
  --lora-path "$LORA_PATH"
  --seed "$SEED"
)

if [[ -n "$SEMANTIC_PATH" ]]; then
  CMD+=(--semantic-image "$SEMANTIC_PATH")
fi

if [[ "$ENABLE_CPU_OFFLOAD" == "true" ]]; then
  CMD+=(--enable-model-cpu-offload)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
