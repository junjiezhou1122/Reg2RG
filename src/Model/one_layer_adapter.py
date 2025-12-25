"""
One-Layer Adapter for Minimal-Capacity Probing

Based on Perceiver Resampler architecture but with depth=1.
Follows the Minimal-Capacity Probing Principle inspired by FAE (Feature Adaptation Encoder):
- Shallow adapter forced to preserve critical information in one pass
- No "layer laziness" - cannot defer compression to later layers
- More "honest" compression for evaluation purposes

Author: Junjie Zhou
Date: 2025-12-25
Experiment: LIT Exp5 (Minimal-Capacity Adapter Comparison)
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat


class OneLayerAdapter(nn.Module):
    """
    1-layer cross-attention adapter for minimal-capacity probing.

    This adapter uses a single Perceiver-style cross-attention block to compress
    vision tokens into a fixed number of latents. By limiting to one layer, we
    force the model to preserve all critical information in a single compression
    step, avoiding the "layer laziness" phenomenon observed in deep adapters.

    Architecture:
        1. Learnable latent queries (parameters)
        2. Single cross-attention block (queries attend to vision tokens)
        3. Single feed-forward network
        4. Layer normalization

    Key difference from 6-layer Perceiver:
        - No iterative refinement across multiple layers
        - Direct gradient flow (no gradient dilution)
        - Must learn complete compression strategy upfront

    Args:
        dim (int): Token embedding dimension (default: 768 for ViT)
        num_latents (int): Number of compressed latent tokens (default: 32)
        heads (int): Number of attention heads (default: 8)
        ff_mult (int): Feed-forward network expansion factor (default: 4)

    Input:
        x: (B, T, F, v, D) - vision tokens
            B: batch size
            T: number of temporal/media items
            F: number of frames
            v: number of vision tokens per frame
            D: token dimension

    Output:
        (B, T, num_latents, D) - compressed latent representation
    """

    def __init__(self, dim=768, num_latents=32, heads=8, ff_mult=4):
        super().__init__()

        # Store configuration
        self.dim = dim
        self.num_latents = num_latents
        self.heads = heads

        # Learnable latent queries (compressed representation "seeds")
        # These are randomly initialized and optimized during training
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Positional embeddings (optional, for compatibility with Perceiver)
        # These are set to None but can be added if needed
        self.frame_embs = None
        self.media_time_embs = None

        # Layer normalization for inputs
        self.norm_media = nn.LayerNorm(dim)     # Normalize vision tokens
        self.norm_latents = nn.LayerNorm(dim)   # Normalize query latents

        # Attention configuration
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5  # Scaling factor for attention scores

        # Cross-attention projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)      # Query projection
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False) # Key-Value projection
        self.to_out = nn.Linear(inner_dim, dim, bias=False)    # Output projection

        # Feed-forward network (MLP)
        hidden_dim = int(dim * ff_mult)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

        # Final output normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass: compress vision tokens to latent representation.

        Args:
            x: (B, T, F, v, D) - vision tokens from ViT encoder

        Returns:
            (B, T, num_latents, D) - compressed latent tokens
        """

        # Extract dimensions
        B, T, F, v = x.shape[:4]

        # Flatten spatial/temporal dimensions: (B, T, F, v, D) -> (B, T, F*v, D)
        # This creates a single sequence of tokens per temporal item
        x = rearrange(x, "b T F v d -> b T (F v) d")

        # Initialize latents by repeating for batch and temporal dimensions
        # (num_latents, D) -> (B, T, num_latents, D)
        latents = repeat(self.latents, "n d -> b T n d", b=B, T=T)

        # === NORMALIZATION ===
        x_norm = self.norm_media(x)          # Normalize input tokens
        latents_norm = self.norm_latents(latents)  # Normalize queries

        # === CROSS-ATTENTION (SINGLE LAYER!) ===
        h = self.heads

        # Project queries from latents
        q = self.to_q(latents_norm)  # (B, T, num_latents, inner_dim)

        # Concatenate input tokens and latents for key-value computation
        # This allows latents to attend to both input tokens AND themselves
        kv_input = torch.cat((x_norm, latents_norm), dim=-2)  # (B, T, F*v + num_latents, D)

        # Project to keys and values
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)  # Each: (B, T, F*v + num_latents, inner_dim)

        # Reshape for multi-head attention
        q = rearrange(q, "b T n (h d) -> b h T n d", h=h)
        k = rearrange(k, "b T n (h d) -> b h T n d", h=h)
        v = rearrange(v, "b T n (h d) -> b h T n d", h=h)

        # Scaled dot-product attention
        q = q * self.scale
        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)  # (B, h, T, num_latents, F*v + num_latents)

        # Numerical stability: subtract max before softmax
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # Attention weights
        attn = sim.softmax(dim=-1)  # (B, h, T, num_latents, F*v + num_latents)

        # Apply attention to values
        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)  # (B, h, T, num_latents, d)

        # Reshape back and project
        out = rearrange(out, "b h T n d -> b T n (h d)")

        # Residual connection
        latents = latents + self.to_out(out)

        # === FEED-FORWARD NETWORK (SINGLE LAYER!) ===
        latents = latents + self.ff(latents)

        # Final normalization
        return self.norm(latents)

    def extra_repr(self):
        """String representation for debugging."""
        return (
            f"dim={self.dim}, num_latents={self.num_latents}, "
            f"heads={self.heads}, depth=1 (MINIMAL-CAPACITY)"
        )
