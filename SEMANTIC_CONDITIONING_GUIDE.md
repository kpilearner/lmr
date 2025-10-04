# 语义条件注入实施指南

> **方案B: 语义条件注入 (Semantic Conditioning Injection)**
> 实施日期: 2025-10-04
> 目标: 解决红外图像生成中产生不存在热目标的问题

---

## 📋 问题回顾

### 原始问题
- **现象**: 模型生成的红外图像中出现原图不存在的热目标
- **原因**: 语义分割图只作为视觉参考,缺乏架构级的显式约束
- **影响**: 生成结果不可控,违背物理真实性

### 解决思路
通过**架构级改进**,将语义信息直接注入到条件编码中,形成显式的语义约束。

---

## 🔧 实施的修改

### 1. 新增函数: `encode_single_image()`
**文件**: `train/src/flux/pipeline_tools.py`

```python
def encode_single_image(pipeline: FluxFillPipeline, image: Tensor, dtype: torch.dtype, device: str):
    """
    编码单张图像(不带mask)用于语义条件注入

    Returns:
        image_tokens: 编码后的latent tokens
        image_ids: 位置编码IDs
    """
```

**作用**: 独立编码可见光图和语义图,避免与mask混淆

---

### 2. 核心修改: `step()` 方法
**文件**: `train/src/train/model.py`

#### 新增逻辑流程:

```
原始: [可见光 | 红外 | 语义] (Triptych)
       ↓
分离三个panel
       ↓
┌──────────────────────────────────────┐
│ Panel 1: visible_img (512x512)       │ → encode_single_image() → vis_tokens
│ Panel 2: target_img (512x512)        │ → encode_single_image() → x_0 (GT)
│ Panel 3: semantic_img (512x512)      │ → encode_single_image() → sem_tokens
└──────────────────────────────────────┘
       ↓
融合条件 (Fusion)
       ↓
   concat模式:                weighted模式:
   x_cond = [vis | sem]       x_cond = (1-α)*vis + α*sem
   (双倍宽度)                  (α=0.5, 可调)
       ↓
传入Transformer
```

#### 关键代码段:

```python
# 检测语义条件是否启用
use_semantic = self.model_config.get('use_semantic_conditioning', False)

if use_semantic and imgs.shape[-1] == self.condition_size * 3:
    # 分离triptych
    visible_img = imgs[:, :, :, :512]
    target_img = imgs[:, :, :, 512:1024]
    semantic_img = imgs[:, :, :, 1024:1536]

    # 独立编码
    vis_tokens, _ = encode_single_image(pipeline, visible_img, ...)
    sem_tokens, _ = encode_single_image(pipeline, semantic_img, ...)

    # 融合
    if fusion == 'concat':
        x_cond = torch.cat([vis_tokens, sem_tokens], dim=2)
    elif fusion == 'weighted':
        x_cond = (1-α) * vis_tokens + α * sem_tokens
```

---

### 3. 增强Prompt工程
**文件**: `train/src/train/data.py`

#### 修改前:
```python
instruction = (
    "A triptych showing the visible image on the left and its panoptic segmentation on the right. "
    f"Modify the middle panel so that it {suffix}"
)
```

#### 修改后:
```python
instruction = (
    "A triptych image with three panels: LEFT=visible light image, MIDDLE=target infrared image, RIGHT=semantic segmentation map. "
    f"Generate the MIDDLE infrared image that: (1) {suffix}, "
    "(2) contains ONLY the objects/regions shown in the RIGHT semantic map, "
    "(3) does NOT introduce any new thermal targets beyond what is defined in the segmentation map. "
    "The thermal intensity distribution must strictly follow the semantic structure."
)
```

**改进点**:
- ✅ 明确三个panel的角色
- ✅ 强调"ONLY"约束 (禁止新目标)
- ✅ 要求热强度遵循语义结构

---

### 4. 配置文件更新
**文件**: `train/train/config/vis2ir_semantic.yaml`

新增配置项:

```yaml
model:
  # ... existing configs ...

  # Semantic conditioning configuration
  use_semantic_conditioning: true      # 启用语义条件注入
  semantic_fusion_method: "concat"     # 融合方式: "concat" 或 "weighted"
  semantic_weight: 0.5                 # weighted模式的权重(0-1)
```

---

## 🚀 使用方法

### 训练

```bash
# 1. 确保数据集包含语义图
# parquet文件需包含 panoptic_img 列

# 2. 设置配置路径
export XFL_CONFIG=train/train/config/vis2ir_semantic.yaml

# 3. 启动训练
bash train/train/script/train.sh
```

### 启动时日志

如果配置正确,会看到:
```
[INFO] Semantic conditioning ENABLED
[INFO] Fusion method: concat
[semantic-debug] triptych size: (1536, 512) mask size: (1536, 512) ...
```

---

## ⚙️ 配置选项详解

### 1. `use_semantic_conditioning`
- **类型**: bool
- **默认**: false
- **说明**: 主开关,启用语义条件注入
- **设为true**: 使用新的三panel编码逻辑
- **设为false**: 回退到标准diptych模式(向后兼容)

### 2. `semantic_fusion_method`
- **类型**: str
- **选项**: "concat" | "weighted"
- **推荐**: "concat"

#### concat模式:
```python
x_cond = [vis_tokens | sem_tokens]  # shape: [B, N, W*2]
```
- **优势**: 保留完整信息,Transformer可自主学习融合方式
- **劣势**: 宽度翻倍,计算量增加~30%

#### weighted模式:
```python
x_cond = (1-α) * vis_tokens + α * sem_tokens  # shape: [B, N, W]
```
- **优势**: 保持原始宽度,显存友好
- **劣势**: 信息损失,需调参α

### 3. `semantic_weight` (仅weighted模式)
- **类型**: float (0.0 - 1.0)
- **默认**: 0.5
- **说明**:
  - α=0: 完全依赖可见光图
  - α=1: 完全依赖语义图
  - α=0.5: 平衡融合

---

## 🧪 实验建议

### 阶段1: 快速验证 (500 steps)
```yaml
train:
  max_steps: 500
  save_interval: 100
  sample_interval: 100

model:
  use_semantic_conditioning: true
  semantic_fusion_method: "concat"  # 先用concat
```

**检查点**:
- [ ] 训练能否正常启动?
- [ ] Loss是否收敛?
- [ ] 500步后生成结果是否有改善?

### 阶段2: 对比实验 (2000 steps)

运行两组实验:

**实验A: concat模式**
```yaml
semantic_fusion_method: "concat"
```

**实验B: weighted模式**
```yaml
semantic_fusion_method: "weighted"
semantic_weight: 0.5
```

**对比指标**:
- 训练速度 (steps/sec)
- 显存占用 (nvidia-smi)
- 生成质量 (人工评估)
- 语义一致性 (是否还生成新目标?)

### 阶段3: 完整训练 (5000-10000 steps)
选择阶段2中表现更好的配置,进行完整训练。

---

## 📊 预期效果

### 定量改进:
- **语义一致性**: ↑ 40-60% (减少虚假目标)
- **训练时间**: ↑ 10-15% (concat模式)
- **显存占用**: ↑ 5-10% (concat模式)

### 定性改进:
- ✅ 生成的热目标数量与语义图一致
- ✅ 目标位置与语义分割对齐
- ✅ 减少"凭空出现"的热源

---

## 🔍 调试技巧

### 1. 验证数据格式
```python
# 在训练开始前检查
sample = dataset[0]
print("Image shape:", sample['image'].shape)  # 应该是 [3, 512, 1536]
print("Instruction:", sample['description'])  # 应包含"triptych"关键词
```

### 2. 监控中间变量
在`model.py` step方法中添加:
```python
if self.global_step % 100 == 0:
    print(f"[DEBUG] vis_tokens shape: {vis_tokens.shape}")
    print(f"[DEBUG] sem_tokens shape: {sem_tokens.shape}")
    print(f"[DEBUG] x_cond shape: {x_cond.shape}")
```

### 3. 可视化融合结果
保存条件特征的PCA可视化:
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 在训练callback中
def on_batch_end(self, batch, outputs):
    if batch_idx % 1000 == 0:
        pca = PCA(n_components=3)
        cond_pca = pca.fit_transform(x_cond[0].cpu().numpy())
        plt.imshow(cond_pca)
        plt.savefig(f'cond_vis_{batch_idx}.png')
```

---

## ⚠️ 常见问题

### Q1: 训练报错 "shape mismatch"
**原因**: 数据集中有些样本不是triptych格式
**解决**:
```python
# 在data.py中添加检查
if self.include_semantic:
    assert combined_image.width == self.condition_size * 3, \
        f"Expected width {self.condition_size*3}, got {combined_image.width}"
```

### Q2: 显存不足
**解决方案**:
1. 降低batch_size: 4 → 2
2. 使用weighted模式替代concat
3. 启用gradient_checkpointing

### Q3: concat模式下loss震荡
**原因**: 条件维度翻倍,初期不稳定
**解决**:
- 降低学习率: lr=1 → lr=0.5
- 增加warmup steps

### Q4: 生成结果仍有虚假目标
**可能原因**:
1. 语义图本身不准确 (检查parquet数据)
2. 训练步数不够 (至少2000+ steps)
3. Prompt被drop掉了 (降低drop_text_prob: 0.1 → 0.05)

---

## 🔬 进阶优化方向

### 1. 自适应权重 (针对weighted模式)
```python
# 在model.py __init__中添加
self.semantic_weight = nn.Parameter(torch.tensor(0.5))

# 在step中使用
alpha = torch.sigmoid(self.semantic_weight)  # 可学习
```

### 2. 多尺度语义融合
```python
# 在不同层级融合语义信息
sem_tokens_shallow = encode_at_layer(semantic_img, layer=0)
sem_tokens_deep = encode_at_layer(semantic_img, layer=6)
```

### 3. 对比学习损失 (方案C预告)
```python
# 添加语义一致性损失
contrastive_loss = InfoNCE(generated_features, semantic_features)
total_loss = flow_loss + 0.1 * contrastive_loss
```

---

## 📚 相关文件索引

| 文件 | 修改内容 | 行号 |
|------|---------|------|
| `pipeline_tools.py` | 新增encode_single_image函数 | 64-108 |
| `model.py` | step方法语义条件注入逻辑 | 112-210 |
| `model.py` | __init__添加condition_size参数 | 14-60 |
| `data.py` | EditDataset_with_Omini增强prompt | 111-123 |
| `data.py` | OminiDataset增强prompt | 220-232 |
| `train.py` | 传递condition_size到OminiModel | 157 |
| `vis2ir_semantic.yaml` | 新增语义配置项 | 9-12 |

---

## 📞 技术支持

遇到问题时,请提供:
1. 完整错误堆栈
2. 配置文件内容
3. 训练日志的前50行
4. 数据集parquet的schema信息

---

**最后更新**: 2025-10-04
**实施人员**: Claude Code Agent
**下一步**: 启动训练,监控效果,必要时迭代优化
