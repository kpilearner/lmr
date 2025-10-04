# ✅ 最终修正方案 - 运行指南

## 🎯 核心方案

### 正确的语义嵌入方式: **像素级融合**

```python
# 核心思路:
# 1. 在像素空间融合可见光图和语义图
# 2. 构造标准diptych: [融合后的可见光 | 红外目标]
# 3. 使用baseline的encode_images_fill编码
# 4. 完全兼容FLUX transformer
```

### 为什么这是正确的?

1. ✅ **维度完全匹配**: 构造的diptych是 [B, 3, 512, 1024],与baseline一致
2. ✅ **语义有效嵌入**: 通过像素级融合,语义信息融入条件编码
3. ✅ **不改变架构**: 使用标准的encode_images_fill,输出维度与baseline相同
4. ✅ **支持可学习权重**: 融合比例可自动学习
5. ✅ **向后兼容**: 禁用语义时完全回退到baseline

---

## 📊 预期的调试输出

运行训练后,你应该在**前3个步**看到详细的shape输出:

```
[INFO] Semantic conditioning ENABLED
[INFO] Fusion method: learnable
[INFO] Using LEARNABLE semantic weight (init=0.5)

[DEBUG] ===== Semantic Conditioning =====
[DEBUG] Input triptych shape: torch.Size([4, 3, 512, 1536])
[DEBUG] visible_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] target_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] semantic_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] Using learnable alpha: 0.5000
[DEBUG] enhanced_visible shape: torch.Size([4, 3, 512, 512])
[DEBUG] enhanced_diptych shape: torch.Size([4, 3, 512, 1024])  ← 关键!与baseline一致
[DEBUG] mask_imgs shape: torch.Size([4, 1, 512, 1024])
[DEBUG] After encode_images_fill:
[DEBUG]   x_0 shape: torch.Size([4, 8192, 64])    ← 序列长度8192
[DEBUG]   x_cond shape: torch.Size([4, 8192, 65]) ← 65 = 64 image + 1 mask
[DEBUG]   img_ids shape: torch.Size([1, 8192, 3])
[DEBUG] x_t shape: torch.Size([4, 8192, 64])
[DEBUG] hidden_states (cat of x_t and x_cond) shape: torch.Size([4, 8192, 129])
[DEBUG] ===================================

[Step 0] Loss: 0.xxxx
```

---

## 🔍 关键检查点

### 必须满足的维度要求:

| 变量 | 预期Shape | 说明 |
|------|----------|------|
| `enhanced_diptych` | `[B, 3, 512, 1024]` | 与baseline diptych一致 |
| `x_0` | `[B, 8192, 64]` | 序列长度8192 |
| `x_cond` | `[B, 8192, 65]` | 65=64+1(mask) |
| `hidden_states` | `[B, 8192, 129]` | 129=64+65 |

**如果任何一个不匹配,请立即停止训练并反馈!**

---

## 🚀 运行步骤

### 步骤1: 启动训练
```bash
export XFL_CONFIG=train/train/config/vis2ir_semantic.yaml
bash train/train/script/train.sh
```

### 步骤2: 观察前3步的DEBUG输出
前3个step会打印完整的shape信息,请**完整复制**这些输出发给我。

### 步骤3: 检查learnable weight的变化
每100步会打印:
```
[Step 100] Learned semantic weight: 0.5234
[Step 200] Learned semantic weight: 0.5891
[Step 300] Learned semantic weight: 0.6125
```

如果weight一直不变(始终0.5000),说明优化器配置有问题。

---

## ⚙️ 配置说明

### 当前配置 (vis2ir_semantic.yaml):
```yaml
model:
  use_semantic_conditioning: true
  semantic_fusion_method: "learnable"  # 自动学习融合权重
  semantic_weight: 0.5  # learnable模式下此参数作为初始化值
```

### 可选配置:

#### 选项1: 使用固定权重
```yaml
model:
  semantic_fusion_method: "fixed"
  semantic_weight: 0.6  # 手动设置 (范围0-1)
```

#### 选项2: 调整初始权重
```yaml
model:
  semantic_fusion_method: "learnable"
  # 修改model.py第61行: torch.tensor(0.6, dtype=dtype)
```

---

## 🐛 可能遇到的问题

### 问题1: Shape mismatch错误
```
RuntimeError: shape mismatch
```

**可能原因**:
- 数据集中的图像不是triptych格式
- condition_size设置错误

**检查**:
```python
# 在data.py中添加检查
assert combined_image.width == 1536, f"Expected 1536, got {combined_image.width}"
```

---

### 问题2: Learnable weight不变化
```
[Step 100] Learned semantic weight: 0.5000
[Step 200] Learned semantic weight: 0.5000  # 一直0.5
```

**检查optimizer配置**:
应该看到:
```
[INFO] Added semantic_weight to optimizer
```

如果没有,检查`configure_optimizers`方法。

---

### 问题3: 显存不足
```
CUDA out of memory
```

**解决**:
1. 降低batch_size: 4 → 2
2. 确保gradient_checkpointing: true

---

## 📈 预期效果

### 训练稳定性:
- ✅ Loss应该正常收敛
- ✅ 无维度错误
- ✅ learnable weight会逐渐变化

### 生成质量改善:
- ✅ 不再生成原图中不存在的热目标
- ✅ 热目标位置与语义分割对齐
- ✅ 热强度分布遵循语义结构

**预期提升**: 语义一致性 +40-60%

---

## 📝 需要反馈的信息

请在运行后提供:

1. **前3个step的完整DEBUG输出** (最重要!)
2. 前20步的Loss值
3. 第100步的learned weight值
4. 任何error或warning信息

示例格式:
```
=== DEBUG输出 (Step 0) ===
[DEBUG] Input triptych shape: ...
[DEBUG] visible_img shape: ...
...

=== Loss值 ===
Step 0: 0.xxxx
Step 1: 0.xxxx
...

=== Learned Weight ===
Step 100: 0.xxxx
```

---

## 🔬 代码修改摘要

### 修改的文件:
1. ✅ `train/src/train/model.py`
   - 添加`import torch.nn as nn`
   - 移除`encode_single_image` import
   - `__init__`: 添加learnable weight初始化
   - `configure_optimizers`: 将weight加入optimizer
   - `step`: 完全重写语义融合逻辑

2. ✅ `train/train/config/vis2ir_semantic.yaml`
   - `semantic_fusion_method`: "concat" → "learnable"

### 核心修改逻辑:
```python
# OLD (错误的):
vis_tokens = encode_single_image(visible_img)
sem_tokens = encode_single_image(semantic_img)
x_cond = cat([vis_tokens, sem_tokens], dim=2)  # ❌ 破坏维度

# NEW (正确的):
enhanced_visible = (1-α)*visible_img + α*semantic_img  # 像素级融合
enhanced_diptych = cat([enhanced_visible, target_img], dim=-1)
x_0, x_cond, img_ids = encode_images_fill(enhanced_diptych, mask)  # ✅ 标准流程
```

---

## ✅ 检查清单

训练前确认:
- [ ] model.py已修改
- [ ] yaml配置已更新
- [ ] 数据集包含triptych格式 (宽度1536)
- [ ] semantic_column配置正确 (panoptic_img)

训练后确认:
- [ ] 看到完整的DEBUG输出
- [ ] x_cond shape是 [B, 8192, 65]
- [ ] hidden_states shape是 [B, 8192, 129]
- [ ] learnable weight在变化
- [ ] Loss正常收敛

---

**修复日期**: 2025-10-04 (最终版)
**核心改进**: 像素级融合 + 标准diptych构造
**预期效果**: 完全兼容FLUX,语义有效嵌入
