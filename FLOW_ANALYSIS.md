# 🔍 Cross-Attention实现流程分析

## 完整训练流程

### 1. 数据加载
```
Triptych from dataset: [B, 3, 512, 1536]
  = [visible_img | target_img | semantic_img]
    [0:512]      [512:1024]    [1024:1536]
```

### 2. 编码阶段 (在no_grad内)

```python
with torch.no_grad():
    # 2.1 编码文本
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(...)

    # 2.2 分离图像
    visible_img = triptych[:, :, :, 0:512]      # [B, 3, 512, 512]
    target_img = triptych[:, :, :, 512:1024]    # [B, 3, 512, 512]
    semantic_img = triptych[:, :, :, 1024:1536] # [B, 3, 512, 512]

    # 2.3 构造diptych并编码
    diptych = cat([visible_img, target_img], dim=-1)  # [B, 3, 512, 1024]

    x_0, x_cond, img_ids = encode_images_fill(diptych, mask)
    # x_0: [B, 2048, 64]    - target latent (ground truth)
    # x_cond: [B, 2048, 320] - masked condition (64 image + 256 mask)

    # 2.4 准备噪声和插值
    x_1 = randn_like(x_0)  # [B, 2048, 64] - pure noise
    t = random_timestep     # [B]
    x_t = (1-t) * x_0 + t * x_1  # [B, 2048, 64] - noisy latent
```

### 3. 语义编码 (在no_grad外,但VAE仍然frozen)

```python
# 在with torch.no_grad()块外
if semantic_mode == 'cross_attention' and semantic_img is not None:
    with torch.no_grad():
        # VAE是frozen的,所以仍然no_grad
        semantic_tokens, _ = encode_images(semantic_img)
        # semantic_tokens: [B, 2048, 64]
```

**关键**: semantic_tokens虽然在no_grad内生成,但:
- VAE本身是frozen的,不需要梯度
- Cross-attention的梯度可以正常回传
- 这就像一个detached的condition signal

### 4. Cross-Attention注入

```python
# 初始hidden_states
hidden_states = cat([x_t, x_cond], dim=2)  # [B, 2048, 384]

# 应用cross-attention (有梯度!)
if semantic_cross_attn is not None and semantic_tokens is not None:
    x_t_enhanced = semantic_cross_attn(x_t, semantic_tokens)
    # x_t: [B, 2048, 64] (query)
    # semantic_tokens: [B, 2048, 64] (key, value)
    # x_t_enhanced: [B, 2048, 64] (output)

    # 重新组合
    hidden_states = cat([x_t_enhanced, x_cond], dim=2)  # [B, 2048, 384]
```

**Cross-Attention内部**:
```python
class SemanticCrossAttention:
    def forward(image_feat, semantic_feat):
        # Normalize
        image_norm = LayerNorm(image_feat)
        semantic_norm = LayerNorm(semantic_feat)

        # Project
        Q = Linear_q(image_norm)    # [B, 2048, 64]
        K = Linear_k(semantic_norm) # [B, 2048, 64]
        V = Linear_v(semantic_norm) # [B, 2048, 64]

        # Multi-head attention (8 heads)
        Q = reshape(Q, [B, num_heads, 2048, head_dim])  # [B, 8, 2048, 8]
        K = reshape(K, [B, num_heads, 2048, head_dim])
        V = reshape(V, [B, num_heads, 2048, head_dim])

        # Attention
        scores = Q @ K.T * scale  # [B, 8, 2048, 2048]
        attn = softmax(scores)
        out = attn @ V            # [B, 8, 2048, 8]

        # Reshape and project
        out = reshape(out, [B, 2048, 64])
        out = Linear_out(out)

        # Residual with learnable scale
        return image_feat + scale * out  # scale初始化为0
```

### 5. Transformer Forward

```python
transformer_out = transformer(
    hidden_states=hidden_states,  # [B, 2048, 384] (x_t_enhanced + x_cond)
    timestep=t,
    guidance=guidance,
    pooled_projections=pooled_prompt_embeds,
    encoder_hidden_states=prompt_embeds,
    txt_ids=text_ids,
    img_ids=img_ids,
)

pred = transformer_out[0]  # [B, 2048, 64] - 预测的速度场
```

### 6. Loss计算

```python
# Flow Matching训练目标
target = x_1 - x_0  # [B, 2048, 64] - 速度场ground truth
loss = MSE(pred, target)
```

---

## 梯度流分析

### ✅ 有梯度的部分:
1. **Cross-Attention模块** (semantic_cross_attn)
   - Q, K, V projections
   - Output projection
   - LayerNorm parameters
   - Scale parameter

2. **Transformer LoRA层**
   - 所有LoRA adapter weights

### ❌ 无梯度的部分(frozen):
1. VAE encoder/decoder
2. CLIP text encoder
3. T5 text encoder
4. Transformer backbone (除了LoRA)

### 梯度回传路径:
```
Loss (MSE)
  ↓
pred (transformer output)
  ↓
transformer blocks + LoRA
  ↓
hidden_states [x_t_enhanced | x_cond]
  ↓
x_t_enhanced (cross-attention output)
  ↓
semantic_cross_attn (Q,K,V projections + scale)
  ↓
x_t (detached, 因为来自no_grad)
semantic_tokens (detached, 因为来自no_grad)
```

**关键**: 虽然x_t和semantic_tokens是detached的,但cross-attention的**参数**仍然可以学习!

这就像:
```python
x = torch.randn(10).detach()  # 无梯度
w = nn.Parameter(torch.randn(10))  # 有梯度
y = (x * w).sum()  # y对w有梯度!
y.backward()  # w.grad不为None
```

---

## 为什么这样设计是合理的?

### 1. 训练目标一致性
- Transformer的任务: 预测速度场 `v = x_1 - x_0`
- Cross-attention的作用: 提供语义引导,让预测更准确
- Loss仍然是标准Flow Matching loss

### 2. 类似ControlNet
```
ControlNet:
  condition_image → ControlNet encoder → features
  features + UNet_features → UNet decoder → prediction

Ours:
  semantic_image → VAE encoder → semantic_tokens
  semantic_tokens + x_t → Cross-Attention → x_t_enhanced
  x_t_enhanced → Transformer → prediction
```

### 3. 梯度流合理
- Cross-attention学习如何从semantic提取有用信息
- Transformer LoRA学习如何利用增强后的特征
- 两者联合优化,实现端到端训练

---

## 推理流程

推理时,流程类似:

```python
# 1. Encode inputs
visible_img = preprocess(input_image)
semantic_img = get_semantic_map(input_image)

# 2. Encode condition
diptych = cat([visible_img, zeros_like(visible_img)], dim=-1)
x_cond = encode_with_mask(diptych)

# 3. Encode semantic
semantic_tokens = encode_images(semantic_img)

# 4. Sampling loop
x_t = randn(...)  # 初始噪声
for t in reversed(timesteps):
    # Apply cross-attention
    x_t_enhanced = semantic_cross_attn(x_t, semantic_tokens)

    # Transformer forward
    hidden = cat([x_t_enhanced, x_cond])
    pred = transformer(hidden, t, ...)

    # Update x_t (Euler或其他sampler)
    x_t = update_step(x_t, pred, t)

# 5. Decode
output = vae.decode(x_t)
```

---

## 关键检查点

### 启动时应该看到:
```
[INFO] Semantic mode: cross_attention
[INFO] Using Cross-Attention with 1 layers, 8 heads
[INFO] Added 33088 semantic cross-attention parameters to optimizer
```

### Step 0-1 DEBUG输出:
```
[DEBUG] visible_img: [4, 3, 512, 512]
[DEBUG] target_img: [4, 3, 512, 512]
[DEBUG] semantic_img: [4, 3, 512, 512]
[DEBUG] diptych: [4, 3, 512, 1024]
[DEBUG] x_0: [4, 2048, 64]
[DEBUG] x_cond: [4, 2048, 320]
[DEBUG] semantic_tokens (encoded outside no_grad): [4, 2048, 64]

[DEBUG] Applying semantic cross-attention:
[DEBUG] x_t: [4, 2048, 64]
[DEBUG] semantic_tokens: [4, 2048, 64]
[DEBUG] Layer 0 scale: 0.000000

[DEBUG] x_t_enhanced: [4, 2048, 64]
[DEBUG] hidden_states (final): [4, 2048, 384]
```

### 训练中(~100 steps):
```
[Step 100] Loss: 0.xxxx
[DEBUG] Layer 0 scale: 0.00x  # 应该逐渐增大
```

### 训练后(~1000 steps):
```
[DEBUG] Layer 0 scale: 0.1-0.5  # 稳定在某个值
```

---

## 总结

✅ **实现正确性**:
- 梯度流畅通 (cross-attention有梯度)
- 训练目标一致 (仍然是Flow Matching)
- 架构合理 (类似ControlNet)

✅ **创新性**:
- 显式语义引导 (vs 简单fusion)
- 可解释 (attention可视化)
- 端到端优化

✅ **实用性**:
- 训练稳定 (zero-init scale)
- 参数量小 (~33K)
- 推理简单 (直接复用)

现在可以放心训练了! 🚀
