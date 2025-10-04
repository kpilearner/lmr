# 🚨 紧急修复: Mask维度不匹配问题

## 问题诊断

从你的输出中发现了关键问题:

```
❌ 错误的输出:
[DEBUG] enhanced_diptych shape: torch.Size([4, 3, 512, 1024])  ✅ diptych正确
[DEBUG] mask_imgs shape: torch.Size([4, 1, 512, 1536])  ❌ mask是triptych的!
[DEBUG]   x_cond shape: torch.Size([4, 2048, 320])  ❌❌ 完全错误!

✅ 预期输出:
[DEBUG] enhanced_diptych shape: torch.Size([4, 3, 512, 1024])
[DEBUG] mask_diptych shape: torch.Size([4, 1, 512, 1024])  ← 应该匹配diptych
[DEBUG]   x_cond shape: torch.Size([4, 8192, 65])  ← 正确维度
```

## 根本原因

**data.py构造的mask是针对1536宽度的triptych,但model.py构造了1024宽度的diptych!**

维度不匹配导致encode_images_fill输出错误的x_cond维度。

---

## ✅ 修复方案

已完成以下修复:

### 修复1: model.py - 重新构造正确的mask

```python
# 在model.py第193-202行添加:

# Create correct mask for diptych (not triptych!)
batch_size = enhanced_diptych.shape[0]
mask_diptych = torch.zeros(
    (batch_size, 1, 512, 1024),  # 匹配diptych宽度
    dtype=enhanced_diptych.dtype,
    device=enhanced_diptych.device
)
# Mark right half as 1 (to be inpainted)
mask_diptych[:, :, :, 512:] = 1.0

# 然后使用mask_diptych而不是mask_imgs
x_0, x_cond, img_ids = encode_images_fill(
    self.flux_fill_pipe,
    enhanced_diptych,
    mask_diptych,  # ← 使用正确的mask!
    dtype, device
)
```

### 修复2: data.py - 简化Prompt (避免CLIP截断)

```python
# 原来的prompt太长(96个token),被CLIP截断
# 改为简短版本:
instruction = (
    f"Infrared image: {suffix}. Follow semantic map strictly, no extra targets."
)
```

---

## 🔄 重新运行

请重新启动训练,现在应该看到:

```
✅ 正确的输出:
[DEBUG] enhanced_diptych shape: torch.Size([4, 3, 512, 1024])
[DEBUG] mask_diptych shape (corrected): torch.Size([4, 1, 512, 1024])  ← 匹配!
[DEBUG]   x_0 shape: torch.Size([4, 8192, 64])    ← seq_len=8192
[DEBUG]   x_cond shape: torch.Size([4, 8192, 65]) ← channels=65
[DEBUG] hidden_states shape: torch.Size([4, 8192, 129])  ← 129=64+65

不再有CLIP截断警告
```

---

## 关键检查点

运行后必须确认:
- [ ] mask_diptych宽度是1024 (不是1536)
- [ ] x_0 shape是 `[B, 8192, 64]`
- [ ] x_cond shape是 `[B, 8192, 65]`
- [ ] hidden_states是 `[B, 8192, 129]`
- [ ] 无CLIP token截断警告

如果还有问题,立即反馈新的DEBUG输出!

---

**修复完成时间**: 刚刚
**需要重新运行**: 是
**预期修复**: 维度完全匹配baseline
