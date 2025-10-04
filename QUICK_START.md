# ⚡ 快速启动指南

## 1️⃣ 确认修改完成

```bash
# 检查model.py是否有learnable weight
grep "semantic_weight = nn.Parameter" train/src/train/model.py

# 应该输出:
# self.semantic_weight = nn.Parameter(torch.tensor(0.5, dtype=dtype))

# 检查配置文件
grep "semantic_fusion_method" train/train/config/vis2ir_semantic.yaml

# 应该输出:
# semantic_fusion_method: "learnable"
```

---

## 2️⃣ 启动训练

```bash
export XFL_CONFIG=train/train/config/vis2ir_semantic.yaml
bash train/train/script/train.sh
```

---

## 3️⃣ 观察输出

### 启动时应看到:
```
[INFO] Semantic conditioning ENABLED
[INFO] Fusion method: learnable
[INFO] Using LEARNABLE semantic weight (init=0.5)
[INFO] Added semantic_weight to optimizer  ← 必须看到!
```

### Step 0-2 应看到完整DEBUG:
```
[DEBUG] ===== Semantic Conditioning =====
[DEBUG] Input triptych shape: torch.Size([4, 3, 512, 1536])
[DEBUG] visible_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] target_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] semantic_img shape: torch.Size([4, 3, 512, 512])
[DEBUG] Using learnable alpha: 0.5000
[DEBUG] enhanced_visible shape: torch.Size([4, 3, 512, 512])
[DEBUG] enhanced_diptych shape: torch.Size([4, 3, 512, 1024])  ← 关键!
[DEBUG] mask_imgs shape: torch.Size([4, 1, 512, 1024])
[DEBUG] After encode_images_fill:
[DEBUG]   x_0 shape: torch.Size([4, 8192, 64])    ← 必须是8192!
[DEBUG]   x_cond shape: torch.Size([4, 8192, 65]) ← 必须是65!
[DEBUG]   img_ids shape: torch.Size([1, 8192, 3])
[DEBUG] x_t shape: torch.Size([4, 8192, 64])
[DEBUG] hidden_states (cat of x_t and x_cond) shape: torch.Size([4, 8192, 129])
[DEBUG] ===================================
```

---

## 4️⃣ 关键检查点

### ✅ 正确的输出:
- enhanced_diptych: `[4, 3, 512, 1024]`
- x_0: `[4, 8192, 64]`
- x_cond: `[4, 8192, 65]`
- hidden_states: `[4, 8192, 129]`

### ❌ 错误的输出(如果看到这些,立即停止):
- x_cond: `[4, 4096, 64]` ← 序列长度错误
- x_cond: `[4, 8192, 128]` ← channels错误
- hidden_states: `[4, 8192, 192]` ← 总维度错误

---

## 5️⃣ 提供反馈

请完整复制Step 0的所有DEBUG输出,发送给我检查。

格式:
```
=== Step 0 DEBUG输出 ===
[DEBUG] ===== Semantic Conditioning =====
[DEBUG] Input triptych shape: ...
...

=== Step 1 Loss ===
Step 1: 0.xxxx
```

---

## 🔧 快速故障排除

| 问题 | 检查 | 解决 |
|------|------|------|
| 未看到DEBUG输出 | 检查global_step | 正常,step 3后会停止打印 |
| x_cond不是65维 | 检查encode_images_fill | 可能mask有问题 |
| OOM | 显存 | batch_size: 4→2 |
| Weight不变 | optimizer | 重启训练,检查日志 |

---

## 📁 修改的文件清单

- ✅ `train/src/train/model.py`
- ✅ `train/train/config/vis2ir_semantic.yaml`

**未修改**:
- ❌ `train/src/flux/pipeline_tools.py` (不需要改)
- ❌ `train/src/train/data.py` (已有triptych支持)
- ❌ `train/src/train/train.py` (不需要改)

---

## 📚 文档索引

- **快速启动**: 本文档
- **详细指南**: `FINAL_CORRECTED_GUIDE.md`
- **改动对比**: `CHANGES_BEFORE_AFTER.md`
- **问题分析**: `CRITICAL_ISSUES_AND_FIXES.md`

---

**准备好了吗? 运行训练,把DEBUG输出发给我!** 🚀
