"""Shared transformer components for TransformerAE and TransformerMAE."""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image into non-overlapping patches and embed them.

    Args:
        patch_size: Size of each square patch
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        embed_dim: Dimension of patch embeddings
    """

    def __init__(self, patch_size=4, in_channels=1, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_dim = patch_size * patch_size * in_channels

        # Linear projection from patch to embedding
        self.projection = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, height, width)

        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        batch_size, channels, height, width = x.shape
        assert channels == self.in_channels
        assert height % self.patch_size == 0 and width % self.patch_size == 0

        # Calculate number of patches
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Extract patches using reshape and permute
        # (B, C, H, W) -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.reshape(batch_size, channels, num_patches_h, self.patch_size,
                     num_patches_w, self.patch_size)

        # (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        # -> (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)

        # (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        # -> (B, num_patches, patch_dim)
        x = x.reshape(batch_size, num_patches, self.patch_dim)

        # Project to embedding dimension
        # (B, num_patches, patch_dim) -> (B, num_patches, embed_dim)
        embeddings = self.projection(x)

        return embeddings


class PositionalEmbedding(nn.Module):
    """Learned 1D positional embeddings.

    Args:
        num_patches: Number of patches (e.g., 64 for 32x32 image with 4x4 patches)
        embed_dim: Dimension of embeddings
    """

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Initialize with truncated normal
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, indices=None):
        """
        Args:
            indices: Optional indices to select specific positions (for masked encoding)
                    If None, returns all positional embeddings

        Returns:
            pos_emb: (1, num_patches, embed_dim) or (1, len(indices), embed_dim)
        """
        if indices is None:
            return self.pos_embedding
        else:
            return self.pos_embedding[:, indices, :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, embed_dim, num_heads=6, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            out: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        # (B, seq_len, embed_dim) -> (B, seq_len, 3 * embed_dim)
        qkv = self.qkv(x)

        # (B, seq_len, 3 * embed_dim) -> (B, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # (B, seq_len, 3, num_heads, head_dim) -> (3, B, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into Q, K, V: each (B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        # (B, num_heads, seq_len, head_dim) @ (B, num_heads, head_dim, seq_len)
        # -> (B, num_heads, seq_len, seq_len)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, head_dim)
        # -> (B, num_heads, seq_len, head_dim)
        out = attn @ v

        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads, head_dim)
        out = out.transpose(1, 2)

        # (B, seq_len, num_heads, head_dim) -> (B, seq_len, embed_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)

        # Output projection
        out = self.proj(out)
        out = self.dropout(out)

        return out


class MLP(nn.Module):
    """Feed-forward MLP with GELU activation.

    Args:
        embed_dim: Input/output dimension
        mlp_ratio: Ratio of hidden dimension to embed_dim (typically 4)
        dropout: Dropout probability
    """

    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            out: (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder/decoder block with pre-normalization.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embed_dim
        dropout: Dropout probability
    """

    def __init__(self, embed_dim, num_heads=6, mlp_ratio=4, dropout=0.0):
        super().__init__()

        # Pre-normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            out: (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class PatchReconstruction(nn.Module):
    """Convert patch embeddings back to image.

    Args:
        embed_dim: Dimension of patch embeddings
        patch_size: Size of each square patch
        in_channels: Number of output channels
        image_size: Size of output image (height and width)
    """

    def __init__(self, embed_dim, patch_size, in_channels, image_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_dim = patch_size * patch_size * in_channels

        # Linear projection from embedding to patch
        self.projection = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_patches, embed_dim)

        Returns:
            image: (batch_size, input_dim) - flattened image
        """
        batch_size, num_patches, embed_dim = x.shape

        # Project to patch dimension
        # (B, num_patches, embed_dim) -> (B, num_patches, patch_dim)
        patches = self.projection(x)

        # Calculate grid dimensions
        num_patches_h = self.image_size // self.patch_size
        num_patches_w = self.image_size // self.patch_size
        assert num_patches == num_patches_h * num_patches_w

        # Reshape to image
        # (B, num_patches, patch_dim) -> (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        patches = patches.reshape(batch_size, num_patches_h, num_patches_w,
                                 self.in_channels, self.patch_size, self.patch_size)

        # (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        # -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        image = patches.permute(0, 3, 1, 4, 2, 5)

        # (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        # -> (B, C, H, W)
        image = image.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        # Flatten to match expected output format
        # (B, C, H, W) -> (B, C * H * W)
        image = image.view(batch_size, -1)

        return image


def initialize_weights(module):
    """Initialize weights following ViT/MAE best practices.

    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)
