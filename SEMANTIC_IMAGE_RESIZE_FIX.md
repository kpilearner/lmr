# 🔧 语义图像尺寸修复

## 问题诊断

### 症状
```
❌ 错误输出:
[DEBUG] x_0 shape: torch.Size([4, 2048, 64])    # 应该是8192!
[DEBUG] x_cond shape: torch.Size([4, 2048, 320]) # 应该是8192, 65!
```

### 根本原因

**序列长度 2048 而不是 8192 意味着图像尺寸是预期的一半!**

经过分析发现,`data.py`中存在关键BUG:

```python
# 第71-73行: crop操作只应用于visible和target图像
if self.crop_the_noise and split <= 1:
    image = image.crop((0, 0, image.width, image.height - image.height // 32))
    edited_image = edited_image.crop((0, 0, edited_image.width, edited_image.height - edited_image.height // 32))
    # ❌ semantic_image没有被crop!

# 第75-80行: resize操作
image = image.resize((self.condition_size, self.condition_size)).convert("RGB")
edited_image = edited_image.resize((self.target_size, self.target_size)).convert("RGB")
if self.include_semantic:
    if semantic_image is None:
        semantic_image = image.copy()
    semantic_image = semantic_image.resize((self.condition_size, self.condition_size)).convert("RGB")
    # ❌ semantic_image从不同的原始尺寸resize,可能导致维度不匹配!
```

**问题**:
1. 当`crop_the_noise=True`且`split<=1`时,只有visible和target图像被裁剪
2. semantic_image保持原始尺寸
3. 三个图像从不同的原始尺寸resize到目标尺寸
4. 如果semantic_image原始尺寸与其他两个不同,会导致最终拼接的triptych尺寸错误

---

## 修复方案

### 修复1: EditDataset_with_Omini类 (第74-76行)

```python
if self.crop_the_noise and split <= 1:
    image = image.crop((0, 0, image.width, image.height - image.height // 32))
    edited_image = edited_image.crop((0, 0, edited_image.width, edited_image.height - edited_image.height // 32))
    # ✅ 新增: 同样裁剪semantic图像!
    if self.include_semantic and semantic_image is not None:
        semantic_image = semantic_image.crop((0, 0, semantic_image.width, semantic_image.height - semantic_image.height // 32))
```

### 修复2: OminiDataset类 (第190-191行)

```python
if self.include_semantic:
    if semantic_image is None:  # ✅ 新增安全检查
        semantic_image = image.copy()
    semantic_image = semantic_image.resize((self.condition_size, self.condition_size)).convert("RGB")
```

---

## 验证步骤

### 1. 重新启动训练

```bash
export XFL_CONFIG=train/train/config/vis2ir_semantic.yaml
bash train/train/script/train.sh
```

### 2. 检查Step 0-2的DEBUG输出

**现在应该看到**:
```
✅ 正确输出:
[DEBUG] ===== Semantic Conditioning =====
[DEBUG] Input triptych shape: torch.Size([4, 3, 512, 1536])
[DEBUG] visible_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] target_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] semantic_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] Using learnable alpha: 0.5000
[DEBUG] enhanced_visible shape: torch.Size([4, 3, 512, 512])
[DEBUG] enhanced_diptych shape: torch.Size([4, 3, 512, 1024])
[DEBUG] mask_diptych shape (corrected): torch.Size([4, 1, 512, 1024])
[DEBUG] After encode_images_fill:
[DEBUG]   x_0 shape: torch.Size([4, 8192, 64])     ← 修复!
[DEBUG]   x_cond shape: torch.Size([4, 8192, 65])  ← 修复!
[DEBUG]   img_ids shape: torch.Size([1, 8192, 3])
[DEBUG] x_t shape: torch.Size([4, 8192, 64])
[DEBUG] hidden_states (cat of x_t and x_cond) shape: torch.Size([4, 8192, 129])
[DEBUG] ===================================
```

### 3. 检查维度关键指标

- [ ] enhanced_diptych: `[4, 3, 512, 1024]` ✅
- [ ] x_0: `[4, 8192, 64]` ✅
- [ ] x_cond: `[4, 8192, 65]` ✅
- [ ] hidden_states: `[4, 8192, 129]` ✅

---

## 为什么会有这个问题?

**数据集特性**:
- 用户的数据集中,图像不是标准的512x512
- visible, target, semantic三个图像可能有不同的原始尺寸
- crop操作遗漏了semantic_image

**维度计算**:
- FLUX VAE: 8x下采样
- 512x1024 diptych → (512/8) × (1024/8) = 64 × 128 = 8192 序列长度
- 如果图像是256x512 → (256/8) × (512/8) = 32 × 64 = 2048 序列长度 ❌

---

## 已修复的文件

- ✅ `train/src/train/data.py` (第74-76行, 190-191行)

---

## 预期效果

修复后,所有三个图像(visible, target, semantic)将:
1. 应用相同的crop操作
2. resize到相同的目标尺寸 (512x512)
3. 正确拼接为1536宽度的triptych
4. 在model.py中正确构造1024宽度的diptych
5. 产生正确的序列长度8192

**现在请重新运行训练,并提供完整的Step 0 DEBUG输出!** 🚀
