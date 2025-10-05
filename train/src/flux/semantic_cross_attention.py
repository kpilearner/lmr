"""
Semantic Cross-Attention Module for FLUX
Enables explicit semantic guidance in infrared image generation
"""

import torch
import torch.nn as nn
from typing import Optional


class SemanticCrossAttention(nn.Module):
    """
    Cross-attention layer to inject semantic guidance into image features.

    Args:
        dim: Feature dimension (64 for FLUX packed latents)
        num_heads: Number of attention heads (default: 8)
        dropout: Attention dropout rate (default: 0.0)

    Input shapes:
        image_feat: [B, seq_len, dim] e.g., [4, 2048, 64]
        semantic_feat: [B, sem_seq_len, dim] e.g., [4, 2048, 64]

    Output shape:
        [B, seq_len, dim] - same as image_feat
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # Cross-attention components
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Normalization
        self.norm_image = nn.LayerNorm(dim)
        self.norm_semantic = nn.LayerNorm(dim)

        # Learnable scale (initialized to 0 for stable training)
        self.scale = nn.Parameter(torch.zeros(1))

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        image_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_feat: [B, seq_len, dim] - image features (query)
            semantic_feat: [B, sem_seq_len, dim] - semantic features (key/value)
            attention_mask: Optional mask for attention

        Returns:
            [B, seq_len, dim] - image features enhanced by semantic guidance
        """
        B, seq_len, dim = image_feat.shape
        _, sem_seq_len, _ = semantic_feat.shape

        # Normalize inputs
        image_norm = self.norm_image(image_feat)
        semantic_norm = self.norm_semantic(semantic_feat)

        # Project to Q, K, V
        # Q from image: [B, seq_len, dim]
        Q = self.q_proj(image_norm)
        # K, V from semantic: [B, sem_seq_len, dim]
        K = self.k_proj(semantic_norm)
        V = self.v_proj(semantic_norm)

        # Reshape for multi-head attention
        # [B, seq_len, dim] -> [B, num_heads, seq_len, head_dim]
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, sem_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, sem_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # [B, num_heads, seq_len, head_dim] @ [B, num_heads, head_dim, sem_seq_len]
        # -> [B, num_heads, seq_len, sem_seq_len]
        scale_factor = self.head_dim ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale_factor

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # [B, num_heads, seq_len, sem_seq_len] @ [B, num_heads, sem_seq_len, head_dim]
        # -> [B, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        # [B, num_heads, seq_len, head_dim] -> [B, seq_len, dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection with learnable scale
        # scale initialized to 0, so initially this has no effect
        output = image_feat + self.scale * attn_output

        return output


class SemanticConditioningAdapter(nn.Module):
    """
    Wrapper that adds semantic cross-attention to FLUX transformer blocks.

    This module can be inserted after transformer blocks to inject semantic guidance.

    Args:
        dim: Feature dimension (default: 64)
        num_layers: Number of cross-attention layers to create (default: 1)
        num_heads: Number of attention heads (default: 8)
    """

    def __init__(
        self,
        dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            SemanticCrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        image_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_feat: [B, seq_len, dim]
            semantic_feat: [B, sem_seq_len, dim]

        Returns:
            [B, seq_len, dim] - semantically guided features
        """
        output = image_feat
        for layer in self.layers:
            output = layer(output, semantic_feat)
        return output


def test_semantic_cross_attention():
    """Test function to verify dimensions"""
    print("Testing SemanticCrossAttention...")

    # Create module
    cross_attn = SemanticCrossAttention(dim=64, num_heads=8)

    # Test inputs (FLUX packed latent dimensions)
    B, seq_len, dim = 4, 2048, 64
    sem_seq_len = 2048

    image_feat = torch.randn(B, seq_len, dim)
    semantic_feat = torch.randn(B, sem_seq_len, dim)

    print(f"Input image_feat: {image_feat.shape}")
    print(f"Input semantic_feat: {semantic_feat.shape}")

    # Forward pass
    output = cross_attn(image_feat, semantic_feat)

    print(f"Output: {output.shape}")
    print(f"Scale parameter: {cross_attn.scale.item():.6f}")

    assert output.shape == image_feat.shape, "Output shape mismatch!"
    print("✅ Test passed!")


if __name__ == "__main__":
    test_semantic_cross_attention()
