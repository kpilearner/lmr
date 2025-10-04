#!/bin/bash

# Visible-to-Infrared with Semantic Guidance Training Script
# usage: bash script/train_vis2ir.sh [CONFIG_FILE] [PORT]
# example: bash script/train_vis2ir.sh vis2ir_semantic.yaml 41353

CONFIG_FILE=${1:-"vis2ir_semantic.yaml"}
PORT=${2:-41353}
DATA_CACHE=${3:-"/root/autodl-tmp/.cache"}

export XFL_CONFIG=./train/config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.

if [[ -n "$DATA_CACHE" ]]; then
  export HF_DATASETS_CACHE="$DATA_CACHE"
  export HUGGINGFACE_HUB_CACHE="$DATA_CACHE"
  echo "Datasets cache redirected to: $DATA_CACHE"
fi

# IMPORTANT: Change CUDA_VISIBLE_DEVICES to your GPU ID
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port ${PORT} -m src.train.train
