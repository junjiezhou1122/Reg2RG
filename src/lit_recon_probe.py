"""
LIT Reconstruction Probe for Medical Imaging (RadGenome)

Main Purpose:
This module implements a Linear Information Theory (LIT) probe to test whether
region-specific medical imaging features can be reconstructed from compressed
representations. It uses:
1. A 3D Vision Transformer (ViT) to encode CT scan volumes
2. A Perceiver Resampler to compress tokens into a compact representation
3. A Cross-Attention Decoder to reconstruct the original features

The reconstruction quality indicates how much anatomical information is preserved
in the compressed representation.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os  # Operating system operations (file paths, directory creation)
import signal  # Handle Ctrl+C / SIGTERM for graceful checkpointing
import time  # Timestamp for checkpoint filenames
from dataclasses import dataclass, field  # For creating configuration classes
from typing import Dict, List, Optional, Tuple  # Type hints for better code documentation

import torch  # PyTorch main library for tensor operations and neural networks
import torch.nn.functional as F  # Functional interface for operations like MSE, normalize
from torch import nn  # Neural network modules (Linear, LayerNorm, etc.)
from torch.utils.data import DataLoader, Subset, random_split  # Data loading and splitting utilities
import transformers  # Hugging Face library for argument parsing

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None

# Custom dataset for loading RadGenome CT scans and region masks
from Dataset.radgenome_dataset_train import RadGenomeDataset_Train
# Perceiver Resampler: compresses variable-length sequences to fixed-length
from Model.helpers import PerceiverResampler
# 1-layer adapter for minimal-capacity probing (Experiment 5)
from Model.one_layer_adapter import OneLayerAdapter
# Adapter weight loading utilities (Experiment 5a)
from Model.adapter_utils import load_first_layer_from_6layer_checkpoint
# 3D positional encodings for volumetric data
from Model.position_encoding import PositionEmbeddingLearned3d
# 3D Vision Transformer for processing CT volumes
from Model.vit_3d import ViT
# Unified configuration (HuggingFace-compatible Arguments)
from config import ModelArguments, DataArguments, TrainArguments


# ============================================================================
# ANATOMICAL REGIONS
# ============================================================================

# List of anatomical regions that can be segmented and analyzed
# These correspond to different organs/structures in chest CT scans
REGIONS = [
    "abdomen",              # Abdominal region
    "bone",                 # Bone structures
    "breast",               # Breast tissue
    "esophagus",            # Esophagus
    "heart",                # Heart
    "lung",                 # Lung tissue
    "mediastinum",          # Middle chest cavity
    "pleura",               # Membrane around lungs
    "thyroid",              # Thyroid gland
    "trachea and bronchie", # Airways
]


# NOTE: ModelArguments, DataArguments, TrainArguments are now imported from config
# See: src/config/arguments.py


# ============================================================================
# DATA COLLATION
# ============================================================================

class LITDataCollator:
    """
    Custom collate function for batching medical imaging data.

    Purpose: Combines multiple dataset samples into a batch, handling:
    - Variable numbers of regions per sample (some scans may not contain all organs)
    - Zero-padding for missing regions to maintain consistent batch structure
    - Filtering out regions that don't appear in any sample in the batch

    PyTorch DataLoader calls this function to create batches from individual samples.
    """

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of dataset instances into a batched dictionary.

        Args:
            instances: List of dictionaries, each containing:
                - vision_x: Dict mapping region names to CT volume tensors
                - mask_x: Dict mapping region names to binary mask tensors
                - bbox_sizes: Dict mapping region names to bbox size info
                - region2area: Metadata about regions (not used in this function)

        Returns:
            Batched dictionary with keys:
                - vision_x: Dict of batched region volumes
                - mask_x: Dict of batched region masks
                - bbox_sizes: List of per-sample bbox size dicts
                - region2area: List of metadata
        """

        # Extract the components from each instance
        vision_xs, mask_xs, region2areas = tuple(
            [instance[key] for instance in instances]  # List comprehension
            for key in ("vision_x", "mask_x", "region2area")  # For each key
        )

        # Extract bbox_sizes if available (for compression ratio analysis)
        bbox_sizes_list = [instance.get("bbox_sizes", {}) for instance in instances]

        # Initialize temporary storage for each anatomical region
        # Dict comprehension: {key: value for key in iterable}
        vision_temp = {area: [] for area in REGIONS}  # Empty list for each region
        mask_temp = {area: [] for area in REGIONS}

        # Get shape of vision and mask tensors from first sample's first region
        # next(iter(...)) gets the first item from dictionary values
        vision_shape = next(iter(vision_xs[0].values())).shape  # e.g., (1, 256, 256, 64)
        mask_shape = next(iter(mask_xs[0].values())).shape

        # Track regions that don't appear in ANY sample in this batch
        useless_regions = []

        # For each anatomical region
        for area in REGIONS:
            flag = False  # Tracks if this region appears in at least one sample

            # For each sample in the batch
            for i in range(len(vision_xs)):
                # Check if current sample contains this region
                if area in vision_xs[i]:
                    # Region exists: add actual data
                    vision_temp[area].append(vision_xs[i][area])
                    mask_temp[area].append(mask_xs[i][area])
                    flag = True  # Mark that this region is present
                else:
                    # Region missing: add zero-filled placeholder
                    # torch.zeros() creates a tensor of zeros with specified shape
                    vision_temp[area].append(torch.zeros(vision_shape))
                    mask_temp[area].append(torch.zeros(mask_shape))

            # If region never appeared in any sample, mark for removal
            if not flag:
                useless_regions.append(area)

        # Concatenate full CT images from all samples into a batch
        # torch.cat() concatenates tensors along specified dimension
        # unsqueeze(0) adds a dimension at position 0: (C,H,W,D) -> (1,C,H,W,D)
        # dim=0 stacks along batch dimension: (1,...) + (1,...) -> (2,...)
        images = torch.cat([vision["image"].unsqueeze(0) for vision in vision_xs], dim=0)

        # Remove regions that don't exist in any sample (saves memory/compute)
        for area in useless_regions:
            vision_temp.pop(area)  # Remove key from dictionary
            mask_temp.pop(area)

        # Get list of regions that actually exist in this batch
        useful_regions = list(vision_temp.keys())

        # Stack region-specific data into batched tensors
        # Dict comprehension with torch.cat to batch each region's data
        vision_xs = {
            area: torch.cat([_.unsqueeze(0) for _ in vision_temp[area]], dim=0)
            for area in useful_regions
        }
        vision_xs["image"] = images  # Add full image to dictionary

        # Same for masks
        mask_xs = {
            area: torch.cat([_.unsqueeze(0) for _ in mask_temp[area]], dim=0)
            for area in useful_regions
        }

        # Return batched data as dictionary
        return dict(
            vision_x=vision_xs,
            mask_x=mask_xs,
            bbox_sizes=bbox_sizes_list,  # List of dicts for compression ratio analysis
            region2area=region2areas
        )


# ============================================================================
# DECODER ARCHITECTURE
# ============================================================================

class CrossAttentionDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention and cross-attention.

    Architecture (standard transformer decoder):
    1. Self-attention: Query tokens attend to each other
    2. Cross-attention: Query tokens attend to memory (encoder output)
    3. Feed-forward network: Two-layer MLP with GELU activation

    Each sublayer uses residual connections and layer normalization.

    PyTorch nn.Module: Base class for all neural network modules.
    Must implement __init__() and forward().
    """

    def __init__(self, dim: int, heads: int, ff_mult: int):
        """
        Initialize decoder layer components.

        Args:
            dim: Token embedding dimension
            heads: Number of parallel attention heads
            ff_mult: Feedforward network expansion factor (hidden_dim = dim * ff_mult)
        """
        super().__init__()  # Initialize parent nn.Module class

        # Self-attention: queries attend to queries
        # nn.MultiheadAttention: PyTorch's built-in multi-head attention
        # batch_first=True: expects input shape (batch, seq_len, dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # Cross-attention: queries attend to memory (encoder outputs)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # Feed-forward network (two-layer MLP)
        # nn.Sequential: Container that chains modules in order
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),  # Linear: y = xW^T + b (expand)
            nn.GELU(),                       # GELU: Gaussian Error Linear Unit activation
            nn.Linear(dim * ff_mult, dim),  # Contract back to original dimension
        )

        # Layer normalization: normalizes across feature dimension
        # nn.LayerNorm: normalize to mean=0, std=1, then scale and shift
        self.norm1 = nn.LayerNorm(dim)  # After self-attention
        self.norm2 = nn.LayerNorm(dim)  # After cross-attention
        self.norm3 = nn.LayerNorm(dim)  # After feedforward

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x: Query tokens, shape (batch, num_queries, dim)
            memory: Memory tokens from encoder, shape (batch, num_memory, dim)

        Returns:
            Updated query tokens, shape (batch, num_queries, dim)
        """

        # Self-attention with residual connection and layer norm
        # self_attn(q, k, v) where q=k=v=x (queries attend to themselves)
        # [0] extracts attention output (ignores attention weights)
        # need_weights=False: don't return attention weights (saves memory)
        # Residual: x = norm(x + attn(x))
        x = self.norm1(x + self.self_attn(x, x, x, need_weights=False)[0])

        # Cross-attention with residual connection and layer norm
        # cross_attn(q=x, k=memory, v=memory): queries attend to memory
        # Residual: x = norm(x + cross_attn(x, memory))
        x = self.norm2(x + self.cross_attn(x, memory, memory, need_weights=False)[0])

        # Feedforward network with residual connection and layer norm
        # Residual: x = norm(x + ff(x))
        x = self.norm3(x + self.ff(x))

        return x


class ProbeDecoder(nn.Module):
    """
    Reconstruction decoder using cross-attention.

    Purpose: Given a compressed representation (memory), reconstruct the original
    token sequence by cross-attending from learned query tokens to memory.

    This is the "probe" in LIT (Linear Information Theory) probing - it tests
    whether information can be linearly decoded from the compressed representation.

    Architecture:
    1. Learnable query tokens (parameters that adapt during training)
    2. 3D positional embeddings (encode spatial location in volume)
    3. Stack of cross-attention decoder layers
    4. Final layer normalization
    """

    def __init__(
        self,
        num_tokens: int,    # Number of tokens to reconstruct (e.g., 8*8*16=1024)
        dim: int,           # Token dimension (must match memory dimension)
        depth: int,         # Number of decoder layers
        heads: int,         # Number of attention heads per layer
        ff_mult: int,       # Feedforward expansion factor
    ):
        """Initialize decoder components."""
        super().__init__()

        self.num_tokens = num_tokens  # Store for validation in forward()

        # Learnable query tokens: initialized randomly, optimized during training
        # nn.Parameter: marks this tensor as a model parameter (will be optimized)
        # torch.randn: sample from normal distribution N(0,1)
        # Shape: (1, num_tokens, dim) - batch dimension 1 will be expanded
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim))

        # 3D positional embeddings for volumetric data
        # Encodes (height, width, depth) position of each token
        # dim // 3: each spatial dimension gets 1/3 of the embedding dimension
        self.pos_embed = PositionEmbeddingLearned3d(
            dim // 3,              # Channels per spatial dimension
            h_patch_num=16,        # Number of patches in height
            w_patch_num=16,        # Number of patches in width
            d_patch_num=128        # Number of patches in depth
        )

        # Stack of cross-attention decoder layers
        # nn.ModuleList: container for storing submodules (registers parameters)
        # List comprehension creates 'depth' decoder layers
        self.layers = nn.ModuleList(
            [CrossAttentionDecoderLayer(dim, heads, ff_mult) for _ in range(depth)]
        )

        # Final layer normalization for output stabilization
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        memory: torch.Tensor,         # Compressed representation to decode from
        grid: Tuple[int, int, int],   # 3D grid dimensions (h, w, d)
        batch_size: int               # Batch size for expanding queries
    ) -> torch.Tensor:
        """
        Reconstruct token sequence from compressed memory.

        Args:
            memory: Compressed representation, shape (batch, num_memory_tokens, dim)
            grid: Tuple of (height, width, depth) patch counts
            batch_size: Batch size to expand query tokens

        Returns:
            Reconstructed tokens, shape (batch, num_tokens, dim)
        """

        # Unpack grid dimensions
        h, w, d = grid

        # Calculate expected number of tokens from grid
        n_tokens = h * w * d  # Total tokens = height * width * depth

        # Validate that grid matches initialized query token count
        if n_tokens != self.num_tokens:
            raise ValueError(
                f"Query tokens mismatch: expected {self.num_tokens}, got {n_tokens}"
            )

        # Generate 3D positional embeddings
        # Shape: (batch_size, h*w*d, dim)
        pos = self.pos_embed(batch_size, h, w, d, memory)

        # Prepare query tokens with positional embeddings
        # self.query[:, :n_tokens, :]: slice to match actual token count
        # .expand(batch_size, -1, -1): repeat for batch (doesn't copy memory)
        #   -1 means keep original dimension size
        # + pos: add positional information to queries
        q = self.query[:, :n_tokens, :].expand(batch_size, -1, -1) + pos

        # Pass through decoder layers sequentially
        # Each layer refines queries by attending to memory
        for layer in self.layers:
            q = layer(q, memory)  # Update queries via cross-attention

        # Apply final normalization and return
        return self.norm(q)


# ============================================================================
# MAIN MODEL
# ============================================================================

class LITProbeModel(nn.Module):
    """
    Complete LIT (Linear Information Theory) probe model.

    Purpose: Test whether anatomical information from CT scans can be reconstructed
    from compressed representations learned by a vision-language model.

    Pipeline:
    1. Vision Encoder (ViT3D): CT volume → token sequence
    2. Adapter (Perceiver): token sequence → compressed representation
    3. Projection (FC): compressed tokens → LLM embedding space
    4. Decoder (ProbeDecoder): compressed representation → reconstructed tokens

    Only the decoder is trained; encoder/adapter/projection are frozen.
    This tests if information is linearly accessible in the compressed space.
    """

    def __init__(
        self,
        vis_dim: int,           # Vision token dimension (768)
        llm_dim: int,           # LLM embedding dimension (4096)
        perceiver_num: int,     # Number of compressed tokens (32)
        decoder_layers: int,    # Decoder depth (2)
        decoder_heads: int,     # Decoder attention heads (8)
        decoder_ff_mult: int,   # Decoder FFN multiplier (4)
        decode_mode: str,       # "pre_proj" or "post_proj"
        adapter_depth: int = 6, # Adapter depth (6 for Perceiver, 1 for OneLayer)
        separate_adapters: bool = False,  # Exp9: separate global/local adapters
    ):
        """Initialize all model components."""
        super().__init__()

        # Store patch sizes for calculating grid dimensions
        self.patch_size = 32          # Spatial patch size (H, W)
        self.frame_patch_size = 4     # Temporal/depth patch size (D)

        # Store dimensions for later use
        self.vis_dim = vis_dim
        self.llm_dim = llm_dim
        self.decode_mode = decode_mode  # Where to decode from
        self.adapter_depth = adapter_depth  # Store for reference
        self.separate_adapters = separate_adapters  # Exp9: separate paths

        # ===== 1. VISION ENCODER (3D ViT) =====
        # Processes 3D CT volumes into token sequences
        self.vision_encoder = ViT(
            image_size=512,                      # Input spatial resolution
            frames=512,                          # Input depth (number of slices)
            image_patch_size=self.patch_size,    # Patch size for H, W
            frame_patch_size=self.frame_patch_size,  # Patch size for D
            dim=vis_dim,                         # Token embedding dimension
            depth=12,                            # Number of transformer layers
            heads=8,                             # Number of attention heads
            mlp_dim=2048,                        # MLP hidden dimension
            dropout=0.1,                         # Dropout rate
            emb_dropout=0.1,                     # Embedding dropout rate
        )

        # ===== 2. ADAPTER (Perceiver Resampler or OneLayer) =====
        # Compresses variable-length token sequence to fixed number of latents
        # Choice depends on adapter_depth and separate_adapters:
        #   - separate_adapters=True (Exp9): Two 1-layer adapters (global + local)
        #   - depth=6: Use PerceiverResampler (baseline, 6 layers of refinement)
        #   - depth=1: Use OneLayerAdapter (minimal-capacity probe, Exp5)
        if separate_adapters:
            # ===== Exp9: Separate Global/Local Adapters =====
            if adapter_depth != 1:
                print(f"[WARN] separate_adapters=True requires adapter_depth=1, got {adapter_depth}")
                print(f"[INFO] Forcing adapter_depth=1 for separate adapters")
            print(f"[INFO] 🔬 Using SEPARATE 1-layer adapters (Exp9)")
            print(f"[INFO]    Global adapter: 32 latents, random init")
            print(f"[INFO]    Local adapter: 32 latents, random init")
            self.global_adapter = OneLayerAdapter(
                dim=vis_dim,
                num_latents=perceiver_num,
                heads=8,
                dim_head=64,
                ff_mult=4
            )
            self.local_adapter = OneLayerAdapter(
                dim=vis_dim,
                num_latents=perceiver_num,
                heads=8,
                dim_head=64,
                ff_mult=4
            )
            # For compatibility, also set self.adapter to global_adapter
            self.adapter = self.global_adapter
        elif adapter_depth == 1:
            print(f"[INFO] 🔬 Using 1-layer adapter (minimal-capacity probe)")
            self.adapter = OneLayerAdapter(
                dim=vis_dim,
                num_latents=perceiver_num,
                heads=8,
                dim_head=64,  # Must match PerceiverAttention for weight loading
                ff_mult=4
            )
        else:
            print(f"[INFO] Using {adapter_depth}-layer Perceiver Resampler")
            self.adapter = PerceiverResampler(
                dim=vis_dim,
                depth=adapter_depth,
                num_latents=perceiver_num
            )

        # ===== 3. PROJECTION LAYER =====
        # Projects from vision space to LLM embedding space
        # nn.Linear: fully connected layer (y = xW^T + b)
        self.fc = nn.Linear(vis_dim, llm_dim)

        # ===== 4. MEMORY PROJECTION FOR DECODER =====
        # Depends on decode mode:
        # - "pre_proj": decode from Perceiver output (vis_dim) → Identity (no-op)
        # - "post_proj": decode from FC output (llm_dim) → project back to vis_dim
        self.mem_proj = (
            nn.Identity()  # No operation (passes input unchanged)
            if decode_mode == "pre_proj"
            else nn.Linear(llm_dim, vis_dim)  # Project from LLM space back to vision space
        )

        # ===== 5. RECONSTRUCTION DECODER =====
        # Calculate grid dimensions for region volumes (smaller than full image)
        grid = self._patch_grid(256, 256, 64)  # Region volume size
        num_tokens = grid[0] * grid[1] * grid[2]  # Total tokens to reconstruct

        if separate_adapters:
            # ===== Exp9: Separate Global/Local Decoders =====
            print(f"[INFO]    Global decoder: {decoder_layers} layers")
            print(f"[INFO]    Local decoder: {decoder_layers} layers")
            self.global_decoder = ProbeDecoder(
                num_tokens=num_tokens,
                dim=vis_dim,
                depth=decoder_layers,
                heads=decoder_heads,
                ff_mult=decoder_ff_mult,
            )
            self.local_decoder = ProbeDecoder(
                num_tokens=num_tokens,
                dim=vis_dim,
                depth=decoder_layers,
                heads=decoder_heads,
                ff_mult=decoder_ff_mult,
            )
            # For compatibility, also set self.decoder to global_decoder
            self.decoder = self.global_decoder
        else:
            self.decoder = ProbeDecoder(
                num_tokens=num_tokens,        # Number of tokens to reconstruct
                dim=vis_dim,                  # Token dimension (must match encoder)
                depth=decoder_layers,         # Number of decoder layers
                heads=decoder_heads,          # Number of attention heads
                ff_mult=decoder_ff_mult,      # FFN expansion factor
            )

    @staticmethod  # Method doesn't need access to instance (self)
    def _patch_grid(h: int, w: int, d: int) -> Tuple[int, int, int]:
        """
        Calculate number of patches in each dimension.

        Args:
            h, w, d: Height, width, depth of input volume

        Returns:
            Tuple of (num_h_patches, num_w_patches, num_d_patches)
        """
        # Integer division: number of patches = volume_size / patch_size
        return h // 32, w // 32, d // 4  # (8, 8, 16) for region volumes

    def encode_tokens(
        self, volume: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Encode CT volume into token sequence using ViT.

        Args:
            volume: CT volume tensor, shape (batch, num_regions, 1, H, W, D)

        Returns:
            tokens: Token sequence, shape (batch*num_regions, num_tokens, dim)
            grid: Tuple of patch counts (h, w, d)
        """

        # Unpack tensor dimensions
        # b: batch size, s: number of regions/sequences per batch
        # c: channels (1 for CT), h,w,d: spatial dimensions
        b, s, c, h, w, d = volume.shape

        # Reshape: merge batch and sequence dimensions for parallel processing
        # (b, s, c, h, w, d) → (b*s, c, h, w, d)
        # ViT processes each region independently
        volume = volume.reshape(b * s, c, h, w, d)

        # Pass through vision encoder
        # Returns: tokens (features), _ (ignore second return value)
        tokens, _ = self.vision_encoder(volume)

        # Calculate grid dimensions for this volume
        grid = self._patch_grid(h, w, d)

        return tokens, grid

    def compress_tokens(self, tokens: torch.Tensor, is_global: bool = True) -> torch.Tensor:
        """
        Compress token sequence using Perceiver adapter.

        Args:
            tokens: Token sequence, shape (batch, num_tokens, dim)
            is_global: If True, use global adapter; if False, use local adapter (Exp9)

        Returns:
            Compressed representation, shape (batch, perceiver_num, dim or llm_dim)
        """

        # Perceiver expects shape: (batch, num_media, num_frames, num_tokens, dim)
        # unsqueeze(1).unsqueeze(1): add two dimensions
        # (b, n, d) → (b, 1, 1, n, d)
        x = tokens.unsqueeze(1).unsqueeze(1)

        # Select adapter based on separate_adapters mode and is_global flag
        if self.separate_adapters:
            adapter = self.global_adapter if is_global else self.local_adapter
        else:
            adapter = self.adapter

        # Perceiver Resampler: compress to fixed number of latent tokens
        # Output: (batch, 1, perceiver_num, dim)
        # squeeze(1): remove media dimension → (batch, perceiver_num, dim)
        z = adapter(x).squeeze(1)

        # If decode mode is "post_proj", project to LLM space
        if self.decode_mode == "post_proj":
            z = self.fc(z)  # (batch, perceiver_num, llm_dim)

        return z

    def decode_tokens(
        self, memory: torch.Tensor, grid: Tuple[int, int, int], is_global: bool = True
    ) -> torch.Tensor:
        """
        Reconstruct original tokens from compressed representation.

        Args:
            memory: Compressed representation, shape (batch, perceiver_num, dim)
            grid: Tuple of patch counts for reconstruction
            is_global: If True, use global decoder; if False, use local decoder (Exp9)

        Returns:
            Reconstructed tokens, shape (batch, h*w*d, vis_dim)
        """

        # Project memory back to vision space if needed
        # pre_proj: memory is already in vis_dim space (Identity does nothing)
        # post_proj: memory is in llm_dim space, project back to vis_dim
        memory = self.mem_proj(memory)

        # Select decoder based on separate_adapters mode and is_global flag
        if self.separate_adapters:
            decoder = self.global_decoder if is_global else self.local_decoder
        else:
            decoder = self.decoder

        # Use decoder to reconstruct tokens via cross-attention
        return decoder(memory, grid, memory.shape[0])


# ============================================================================
# LOSS FUNCTIONS AND METRICS
# ============================================================================

def l2_error_topk(x_hat: torch.Tensor, x: torch.Tensor, ratio: float = 0.01) -> float:
    """
    Calculate mean L2 error of the top-k worst predictions.

    Purpose: Identify regions where reconstruction is poorest (useful diagnostic).

    Args:
        x_hat: Predicted tokens, shape (batch, num_tokens, dim)
        x: Ground truth tokens, shape (batch, num_tokens, dim)
        ratio: Fraction of tokens to consider (default 1% worst)

    Returns:
        Mean L2 error of top-k worst predictions (scalar)
    """

    # Calculate L2 norm (Euclidean distance) for each token
    # torch.norm(x, dim=-1): L2 norm across feature dimension
    # err shape: (batch * num_tokens,) after reshape
    err = torch.norm(x_hat - x, dim=-1).reshape(-1)

    # Determine k: number of worst tokens to consider
    # max(1, ...): ensure at least 1 token selected
    # err.numel(): total number of elements in error tensor
    k = max(1, int(err.numel() * ratio))

    # Select top-k largest errors
    # .topk(k): returns largest k values
    # .values: extract values (ignore indices)
    # .mean(): average the top-k errors
    # .item(): convert single-element tensor to Python float
    return err.topk(k).values.mean().item()


def recon_loss(
    x_hat: torch.Tensor,  # Predicted tokens
    x: torch.Tensor,      # Ground truth tokens
    ln: nn.LayerNorm      # Layer normalization module
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Calculate reconstruction loss combining MSE and cosine similarity.

    Purpose: Measures both magnitude error (MSE) and directional error (1 - cosine).

    Args:
        x_hat: Predicted tokens, shape (batch, num_tokens, dim)
        x: Ground truth tokens, shape (batch, num_tokens, dim)
        ln: LayerNorm for normalizing before MSE calculation

    Returns:
        Tuple of (loss, mse, cosine_similarity, top1_error)
    """

    # ===== MSE LOSS (after LayerNorm) =====
    # Normalize both predictions and targets before computing MSE
    # This makes loss invariant to magnitude scaling
    x_ln = ln(x)          # Normalize ground truth
    x_hat_ln = ln(x_hat)  # Normalize predictions

    # F.mse_loss: mean squared error = mean((x_hat - x)^2)
    mse = F.mse_loss(x_hat_ln, x_ln)

    # ===== COSINE SIMILARITY LOSS =====
    # F.normalize: normalize to unit vectors (L2 norm = 1)
    # dim=-1: normalize across feature dimension
    x_norm = F.normalize(x, dim=-1)          # shape: (batch, num_tokens, dim)
    x_hat_norm = F.normalize(x_hat, dim=-1)

    # Cosine similarity = dot product of unit vectors
    # (x_norm * x_hat_norm): element-wise multiplication
    # .sum(dim=-1): sum across feature dimension → (batch, num_tokens)
    # .mean(): average across all tokens and batches
    cos = (x_norm * x_hat_norm).sum(dim=-1).mean()

    # ===== COMBINED LOSS =====
    # MSE: measures magnitude error
    # (1 - cos): measures directional error (0 = perfect alignment)
    loss = mse + (1 - cos)

    # ===== DIAGNOSTIC METRIC =====
    # Calculate top-1% worst L2 errors
    top1 = l2_error_topk(x_hat, x, ratio=0.01)

    # Return loss tensor and metrics as Python floats
    # .item(): extract scalar value from 0-dimensional tensor
    return loss, mse.item(), cos.item(), top1


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Freeze or unfreeze all parameters in a module.

    Purpose: Control which parts of the model are trainable.

    Args:
        module: PyTorch module (nn.Module)
        requires_grad: True to enable gradient computation, False to freeze
    """

    # Iterate over all parameters in the module
    # module.parameters(): generator yielding all Parameter objects
    for param in module.parameters():
        # Set requires_grad flag
        # When False: gradients won't be computed (saves memory and compute)
        # When True: gradients computed and parameter updated during optimization
        param.requires_grad = requires_grad


def warmup_monai_cache(
    dataset: torch.utils.data.Dataset,
    desc: str,
    show_progress: bool,
    max_items: Optional[int],
    num_workers: int,
    priority: Optional[int] = None,
) -> None:
    """
    Warm up MONAI PersistentDataset cache by triggering _transform.

    RadGenomeDataset_Train inherits from PersistentDataset with a MapTransform-based
    cacheable transform. This function triggers MONAI's _transform method which loads
    NIfTI files and applies preprocessing, caching the results for future use.

    We call _transform instead of __getitem__ to skip text tokenization which is not cacheable.

    Progress bar improvements:
    - Pre-scan to count already-cached vs uncached samples
    - Display real-time statistics (cached_skip / processed)
    - Show accurate completion estimates based on actual work

    Args:
        dataset: Dataset to warm up
        desc: Description for progress bar
        show_progress: Whether to show tqdm progress bar
        max_items: Optional max samples to process
        num_workers: Number of DataLoader workers
        priority: Process nice priority (-20 to 19, None=no change)
    """

    # Set process priority if requested
    if priority is not None:
        try:
            import os
            current_nice = os.nice(0)  # Get current
            os.nice(priority - current_nice)  # Adjust
            final_nice = os.nice(0)
            if final_nice < 0:
                print(f"🚀 Process priority set to {final_nice} (HIGH priority)")
            elif final_nice == 0:
                print(f"⚙️  Process priority set to {final_nice} (normal priority)")
            else:
                print(f"🐌 Process priority set to {final_nice} (low priority)")
        except PermissionError:
            print(f"⚠️  Cannot set priority to {priority} (need sudo for negative values)")
        except Exception as e:
            print(f"⚠️  Failed to set priority: {e}")

    base_ds: torch.utils.data.Dataset = dataset
    indices: List[int]
    if isinstance(dataset, Subset):
        base_ds = dataset.dataset  # type: ignore[assignment]
        indices = list(dataset.indices)  # type: ignore[attr-defined]
    else:
        indices = list(range(len(dataset)))

    if max_items is not None:
        indices = indices[: max_items]

    # Pre-scan to count cached vs uncached samples
    if hasattr(base_ds, 'cache_exists'):
        print(f"🔍 Scanning cache status for {len(indices)} samples...")
        uncached_indices = []
        cached_count = 0

        scan_iterator = indices
        if show_progress and tqdm is not None:
            scan_iterator = tqdm(
                indices,
                desc=f"📊 {desc} [scanning]",
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )

        for idx in scan_iterator:
            if base_ds.cache_exists(idx):  # type: ignore[attr-defined]
                cached_count += 1
            else:
                uncached_indices.append(idx)

        uncached_count = len(uncached_indices)
        cache_percent = (cached_count / len(indices) * 100) if len(indices) > 0 else 0
        print(f"✨ Cache status: {cached_count}/{len(indices)} already cached ({cache_percent:.1f}%), "
              f"{uncached_count} need processing")

        # If everything is cached, skip warmup
        if uncached_count == 0:
            print(f"🎉 All samples already cached! Skipping warmup.")
            return
    else:
        # Fallback if cache_exists not available
        uncached_indices = indices
        uncached_count = len(indices)
        cached_count = 0

    # Check if dataset is a PersistentDataset with _transform method
    if num_workers <= 0:
        # Single-process warmup with smart skipping and detailed progress
        iterator = indices
        if show_progress and tqdm is not None:
            iterator = tqdm(
                indices,
                desc=f"⚡ {desc}",
                leave=True,
                total=len(indices),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                colour='green'
            )

        cached_skip = 0
        processed = 0

        for idx in iterator:
            # Fast check: skip if cache already exists (only compute missing ones)
            if hasattr(base_ds, 'cache_exists') and base_ds.cache_exists(idx):  # type: ignore[attr-defined]
                cached_skip += 1
                if show_progress and tqdm is not None:
                    iterator.set_postfix(  # type: ignore[attr-defined]
                        {'✓cached': cached_skip, '🔥new': processed},
                        refresh=False
                    )
                continue  # Skip - cache exists (几毫秒)

            # Trigger MONAI's caching by accessing the item (calls __getitem__ -> _cachecheck)
            _ = base_ds[idx]  # Only process if cache missing (5-10秒)
            processed += 1
            if show_progress and tqdm is not None:
                iterator.set_postfix(  # type: ignore[attr-defined]
                    {'✓cached': cached_skip, '🔥new': processed},
                    refresh=False
                )

        if show_progress:
            print(f"✅ Warmup complete: {cached_skip} cached (skipped), {processed} newly processed")
        return

    # Parallel warmup: avoid transferring large tensors back to the main process.
    # Each worker triggers MONAI's caching and returns a tiny value.
    class _WarmupDataset(torch.utils.data.Dataset):
        def __init__(self, base: torch.utils.data.Dataset, idxs: List[int]) -> None:
            self.base = base
            self.idxs = idxs

        def __len__(self) -> int:  # noqa: D401
            return len(self.idxs)

        def __getitem__(self, i: int) -> Tuple[int, int]:
            """Returns (was_cached, idx) tuple for statistics tracking."""
            idx = self.idxs[i]
            # Smart skip: only process if cache doesn't exist
            if hasattr(self.base, 'cache_exists') and self.base.cache_exists(idx):  # type: ignore[attr-defined]
                return (1, idx)  # Cache exists, skip (fast)
            # Trigger MONAI's caching by accessing the item
            _ = self.base[idx]  # type: ignore[index]
            return (0, idx)  # Processed

    warmup_ds = _WarmupDataset(base_ds, indices)
    loader = DataLoader(
        warmup_ds,
        batch_size=1,  # Keep batch_size=1 for cache warmup to minimize RAM usage
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=1,  
        persistent_workers=False,
    )

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(
            loader,
            total=len(warmup_ds),
            desc=f"⚡ {desc} [{num_workers}w]",
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            colour='cyan'
        )

    cached_skip = 0
    processed = 0

    for was_cached, _ in iterator:
        # batch_size=1, so was_cached is a single value
        if was_cached.item():
            cached_skip += 1
        else:
            processed += 1

        if show_progress and tqdm is not None:
            iterator.set_postfix(  # type: ignore[attr-defined]
                {'✓cached': cached_skip, '🔥new': processed},
                refresh=False
            )

    if show_progress:
        print(f"✅ Warmup complete: {cached_skip} cached (skipped), {processed} newly processed")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run_training(
    model_args: ModelArguments,
    data_args: DataArguments,
    train_args: TrainArguments
) -> None:
    """
    Run training with the given arguments.

    This is the main training function that can be called directly with
    Arguments objects (from run_experiment.py) or indirectly via main()
    which parses CLI args first.

    Args:
        model_args: Model architecture configuration
        data_args: Dataset paths configuration
        train_args: Training hyperparameters
    """

    # ===== 1. SET RANDOM SEEDS =====
    # Ensures reproducibility: same seed → same random numbers → same results
    torch.manual_seed(train_args.seed)              # CPU random number generator
    torch.cuda.manual_seed_all(train_args.seed)     # All GPU random number generators

    # ===== 2. DEVICE SETUP =====
    # torch.cuda.is_available(): check if CUDA-capable GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[device] GPU count: {torch.cuda.device_count()}")
        print(f"[device] Current GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")

    # Create output directory if it doesn't exist
    # exist_ok=True: don't raise error if directory already exists
    os.makedirs(train_args.output_dir, exist_ok=True)

    # ===== 2.5. EXPERIMENT TRACKING (W&B) =====
    wandb_run = None
    if train_args.use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases logging requested but `wandb` is not installed. "
                "Install it (e.g., `pip install wandb`) or set --use_wandb False."
            ) from exc

        wandb_run = wandb.init(
            project=train_args.wandb_project,
            entity=train_args.wandb_entity,
            name=train_args.wandb_run_name,
            mode=train_args.wandb_mode,
            config={
                "model": vars(model_args),
                "data": vars(data_args),
                "train": vars(train_args),
            },
        )

    # ===== 4. DATASET LOADING =====
    # Load RadGenome dataset with CT scans, region masks, and reports
    train_dataset = RadGenomeDataset_Train(
        text_tokenizer=model_args.tokenizer_path,  # For text preprocessing
        data_folder=data_args.data_folder,         # CT scan volumes
        mask_folder=data_args.mask_folder,         # Region masks
        csv_file=data_args.report_file,            # Radiology reports
        cache_dir=data_args.monai_cache_dir,       # MONAI cache for faster loading
    )

    def infer_val_paths_from_train(args: DataArguments) -> Optional[Tuple[str, str, str]]:
        """
        Try to infer validation paths from train paths (train_* -> valid_*).
        Returns (val_data_folder, val_mask_folder, val_report_file) if all exist.
        """
        data_folder = args.data_folder
        mask_folder = args.mask_folder
        report_file = args.report_file

        if data_folder.endswith("/train_preprocessed"):
            data_folder = data_folder[: -len("/train_preprocessed")] + "/valid_preprocessed"
        if mask_folder.endswith("/train_region_mask"):
            mask_folder = mask_folder[: -len("/train_region_mask")] + "/valid_region_mask"
        if report_file.endswith("/train_region_report.csv"):
            report_file = (
                report_file[: -len("/train_region_report.csv")] + "/valid_region_report.csv"
            )

        if all(os.path.exists(p) for p in (data_folder, mask_folder, report_file)):
            return data_folder, mask_folder, report_file
        return None

    # ===== 5. TRAIN/VAL DATASETS =====
    # Prefer dedicated validation set if provided (or inferable); otherwise split train.
    val_paths: Optional[Tuple[str, str, str]] = None
    any_val_arg_set = any(
        v is not None
        for v in (data_args.val_data_folder, data_args.val_mask_folder, data_args.val_report_file)
    )
    if any_val_arg_set:
        missing = [
            name
            for name, value in (
                ("val_data_folder", data_args.val_data_folder),
                ("val_mask_folder", data_args.val_mask_folder),
                ("val_report_file", data_args.val_report_file),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "If setting any val_* args, you must set all of them. Missing: "
                + ", ".join(missing)
            )
        val_paths = (
            data_args.val_data_folder,
            data_args.val_mask_folder,
            data_args.val_report_file,
        )
    else:
        val_paths = infer_val_paths_from_train(data_args)

    if val_paths is not None:
        val_dataset = RadGenomeDataset_Train(
            text_tokenizer=model_args.tokenizer_path,
            data_folder=val_paths[0],
            mask_folder=val_paths[1],
            csv_file=val_paths[2],
            cache_dir=data_args.monai_cache_dir,
        )
        train_set, val_set = train_dataset, val_dataset
        print(f"Using dedicated validation set: data_folder={val_paths[0]}")
    else:
        # Fall back: split train into train/val.
        val_size = int(len(train_dataset) * data_args.val_split)  # e.g., 5% for validation
        train_size = len(train_dataset) - val_size                 # Remaining for training

        train_set, val_set = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # Fixed seed
        )
        print(f"Using random_split for validation: val_split={data_args.val_split}")

    # ===== 5.5. OPTIONAL: WARM UP MONAI PERSISTENT CACHE =====
    precache_mode = (train_args.precache_splits or "none").lower()
    valid_modes = {"none", "train", "val", "both"}
    if precache_mode not in valid_modes:
        raise ValueError(
            f"Invalid --precache_splits {train_args.precache_splits!r}. "
            f"Choose one of: {', '.join(sorted(valid_modes))}"
        )

    if precache_mode in {"train", "both"}:
        print("Warming up MONAI cache for training set...")
        warmup_monai_cache(
            train_set,
            desc="[cache] train",
            show_progress=train_args.show_progress,
            max_items=train_args.precache_max_items,
            num_workers=train_args.precache_num_workers,
            priority=train_args.warmup_priority,
        )
    if precache_mode in {"val", "both"}:
        print("Warming up MONAI cache for validation set...")
        warmup_monai_cache(
            val_set,
            desc="[cache] val",
            show_progress=train_args.show_progress,
            max_items=train_args.precache_max_items,
            num_workers=train_args.precache_num_workers,
            priority=train_args.warmup_priority,
        )

    if train_args.precache_only and precache_mode != "none":
        print("Cache warmup complete (--precache_only True). Exiting without training.")
        return

    # ===== 6. DATA LOADERS =====
    # Initialize custom collator for batching
    collator = LITDataCollator()

    # Training data loader
    train_loader = DataLoader(
        train_set,                                  # Training subset
        batch_size=train_args.batch_size,          # Samples per batch
        shuffle=True,                               # Shuffle each epoch (important!)
        num_workers=train_args.dataloader_num_workers,  # Parallel data loading
        collate_fn=collator,                        # Custom batching function
        pin_memory=True,                            # Speed up CPU->GPU transfer
        persistent_workers=True if train_args.dataloader_num_workers > 0 else False,  # Reuse workers
        prefetch_factor=2 if train_args.dataloader_num_workers > 0 else None,  # Prefetch batches
    )

    # Validation data loader (same as train but no shuffling)
    val_loader = DataLoader(
        val_set,
        batch_size=train_args.batch_size,
        shuffle=False,                              # Don't shuffle validation
        num_workers=train_args.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=True,                            # Speed up CPU->GPU transfer
        persistent_workers=True if train_args.dataloader_num_workers > 0 else False,  # Reuse workers
        prefetch_factor=2 if train_args.dataloader_num_workers > 0 else None,  # Prefetch batches
    )

    # ===== 7. MODEL INITIALIZATION =====
    model = LITProbeModel(
        vis_dim=model_args.vis_dim,                # Vision token dimension
        llm_dim=model_args.llm_dim,                # LLM embedding dimension
        perceiver_num=model_args.perceiver_num,    # Compressed token count
        decoder_layers=model_args.decoder_layers,  # Decoder depth
        decoder_heads=model_args.decoder_heads,    # Decoder attention heads
        decoder_ff_mult=model_args.decoder_ff_mult,  # Decoder FFN multiplier
        decode_mode=model_args.decode_mode,        # "pre_proj" or "post_proj"
        adapter_depth=model_args.adapter_depth,    # Adapter depth (NEW for Exp5)
        separate_adapters=model_args.separate_adapters,  # Exp9: separate global/local
    )

    # ===== 8. LOAD PRETRAINED VISION ENCODER =====
    if model_args.pretrained_visual_encoder:
        # torch.load: load saved model checkpoint from disk
        # map_location="cpu": load to CPU first (avoid GPU memory issues)
        ckpt = torch.load(model_args.pretrained_visual_encoder, map_location="cpu")

        # load_state_dict: copy weights into model
        # strict=True: require exact match of parameter names
        model.vision_encoder.load_state_dict(ckpt, strict=True)

    # ===== 9. LOAD PRETRAINED ADAPTER AND PROJECTION =====
    if model_args.pretrained_adapter and model_args.use_pretrained_adapter:

        # Case 1: Exp5b - Random initialization (skip pretrained weights)
        if model_args.random_init_adapter:
            print("[INFO] ❄️  Adapter randomly initialized (no pretrained weights)")
            print("[INFO] 🧪 Exp5b: Training fresh 1-layer adapter from scratch")

        # Case 2: Exp5a - Extract first layer from 6-layer checkpoint
        elif model_args.adapter_depth == 1 and model_args.load_first_layer_from_pretrained:
            print("[INFO] 🧪 Exp5a: Loading 1st layer from 6-layer checkpoint")
            load_first_layer_from_6layer_checkpoint(
                model.adapter,
                model_args.pretrained_adapter
            )
            # Also load projection layer if needed
            if model_args.decode_mode == "post_proj":
                adapter_ckpt = torch.load(model_args.pretrained_adapter, map_location="cpu")
                if "fc" in adapter_ckpt:
                    model.fc.load_state_dict(adapter_ckpt["fc"])
                    print("[INFO] ✓ Loaded projection layer (fc)")

        # Case 3: Normal - Load full adapter checkpoint (6-layer or matching depth)
        else:
            adapter_ckpt = torch.load(model_args.pretrained_adapter, map_location="cpu")

            # Load adapter weights
            if "perceiver" in adapter_ckpt:
                if model_args.adapter_depth == 6:
                    print("[INFO] Loading full 6-layer Perceiver checkpoint")
                    model.adapter.load_state_dict(adapter_ckpt["perceiver"])
                else:
                    print(f"[WARN] adapter_depth={model_args.adapter_depth} but loading 6-layer checkpoint")
                    print("[WARN] Weight shapes may not match. Consider using --random_init_adapter")
                    try:
                        model.adapter.load_state_dict(adapter_ckpt["perceiver"], strict=False)
                    except RuntimeError as e:
                        print(f"[ERROR] Failed to load adapter weights: {e}")
                        print("[INFO] Continuing with random initialization")

            # Load projection layer (same for all cases)
            if model_args.decode_mode == "post_proj" and "fc" in adapter_ckpt:
                model.fc.load_state_dict(adapter_ckpt["fc"])
                print("[INFO] ✓ Loaded projection layer (fc)")

    # ===== 10. MOVE MODEL TO GPU =====
    # .to(device): transfer all model parameters to specified device (CPU/GPU)
    model.to(device)

    # ===== 11. FREEZE PRETRAINED COMPONENTS =====
    # Exp1 (baseline): Train decoder only, freeze everything else
    # Exp2 (joint training): Train adapter + decoder, freeze encoder + fc
    # Exp9 (separate adapters): Train both global/local adapters + decoders
    set_requires_grad(model.vision_encoder, False)           # Always freeze ViT ❄️
    set_requires_grad(model.fc, False)                       # Always freeze projection ❄️

    if model_args.separate_adapters:
        # Exp9: Handle both adapters and both decoders
        set_requires_grad(model.global_adapter, train_args.train_adapter)
        set_requires_grad(model.local_adapter, train_args.train_adapter)
        set_requires_grad(model.global_decoder, True)
        set_requires_grad(model.local_decoder, True)
    else:
        set_requires_grad(model.adapter, train_args.train_adapter)  # Unlock if Exp2 🔥/❄️
        set_requires_grad(model.decoder, True)                   # Always train decoder 🔥

    if model_args.separate_adapters:
        print("[INFO] 🔬 Exp9 Mode: Separate Global/Local Adapters + Decoders")
        if train_args.train_adapter:
            print("[INFO] 🔥 Training BOTH adapters + BOTH decoders")
        else:
            print("[INFO] ❄️ Adapters frozen, training BOTH decoders only")
    elif train_args.train_adapter:
        print("[INFO] 🔥 Exp2 Mode: Training BOTH adapter + decoder (joint training)")
    else:
        print("[INFO] ❄️ Exp1 Mode: Training decoder ONLY (adapter frozen)")


    # ===== 12. OPTIMIZER SETUP =====
    # AdamW: Adam optimizer with weight decay (L2 regularization)
    # Exp1: Only decoder parameters
    # Exp2: Adapter + decoder parameters (joint training)
    # Exp9: Both adapters + both decoders
    if model_args.separate_adapters:
        # Exp9: Collect parameters from both adapters and both decoders
        decoder_params = (
            list(model.global_decoder.parameters()) +
            list(model.local_decoder.parameters())
        )
        num_global_decoder = sum(p.numel() for p in model.global_decoder.parameters())
        num_local_decoder = sum(p.numel() for p in model.local_decoder.parameters())

        if train_args.train_adapter:
            adapter_params = (
                list(model.global_adapter.parameters()) +
                list(model.local_adapter.parameters())
            )
            trainable_params = adapter_params + decoder_params
            num_global_adapter = sum(p.numel() for p in model.global_adapter.parameters())
            num_local_adapter = sum(p.numel() for p in model.local_adapter.parameters())
            print(f"[INFO] 📊 Exp9 Parameter Breakdown:")
            print(f"[INFO]    Global Adapter: {num_global_adapter:,} params")
            print(f"[INFO]    Local Adapter:  {num_local_adapter:,} params")
            print(f"[INFO]    Global Decoder: {num_global_decoder:,} params")
            print(f"[INFO]    Local Decoder:  {num_local_decoder:,} params")
            total_params = num_global_adapter + num_local_adapter + num_global_decoder + num_local_decoder
            print(f"[INFO]    Total Trainable: {total_params:,} params")
        else:
            trainable_params = decoder_params
            print(f"[INFO] 📊 Exp9 Parameter Breakdown (adapters frozen):")
            print(f"[INFO]    Global Decoder: {num_global_decoder:,} params")
            print(f"[INFO]    Local Decoder:  {num_local_decoder:,} params")
            print(f"[INFO]    Total Trainable: {num_global_decoder + num_local_decoder:,} params")
    elif train_args.train_adapter:
        # Joint training: optimize both adapter and decoder
        trainable_params = list(model.adapter.parameters()) + list(model.decoder.parameters())
        num_adapter_params = sum(p.numel() for p in model.adapter.parameters())
        num_decoder_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"[INFO] Training {num_adapter_params:,} adapter params + {num_decoder_params:,} decoder params")
    else:
        # Baseline: only decoder
        trainable_params = list(model.decoder.parameters())
        num_decoder_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"[INFO] Training {num_decoder_params:,} decoder params only")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_args.learning_rate,         # Learning rate
        weight_decay=train_args.weight_decay  # L2 regularization strength
    )

    # ===== 13. LAYER NORM FOR LOSS CALCULATION =====
    # Used in recon_loss to normalize features before MSE
    ln = nn.LayerNorm(model.vis_dim).to(device)

    # Global step counter for logging (increments per training batch)
    global_step = 0
    current_epoch = 0
    stop_requested = False
    stop_reason = "interrupt"

    # ===== 13.5. RESUME FROM CHECKPOINT (IF SPECIFIED) =====
    start_epoch = 1  # Default: start from epoch 1
    if train_args.resume_from_checkpoint:
        ckpt_path = train_args.resume_from_checkpoint
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[resume] Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Restore decoder weights
        if model_args.separate_adapters:
            # Exp9: Load separate decoders
            if "global_decoder_state_dict" in checkpoint:
                model.global_decoder.load_state_dict(checkpoint["global_decoder_state_dict"])
                print(f"[resume] Restored global decoder weights")
            if "local_decoder_state_dict" in checkpoint:
                model.local_decoder.load_state_dict(checkpoint["local_decoder_state_dict"])
                print(f"[resume] Restored local decoder weights")
        elif "decoder_state_dict" in checkpoint:
            model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
            print(f"[resume] Restored decoder weights")
        elif "model_state_dict" in checkpoint:
            # Handle top-k checkpoints (full model)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[resume] Restored full model weights")

        # Restore adapter weights if training adapter (Exp2/Exp9)
        if train_args.train_adapter:
            if model_args.separate_adapters:
                # Exp9: Load separate adapters
                if "global_adapter_state_dict" in checkpoint:
                    model.global_adapter.load_state_dict(checkpoint["global_adapter_state_dict"])
                    print(f"[resume] Restored global adapter weights (Exp9)")
                if "local_adapter_state_dict" in checkpoint:
                    model.local_adapter.load_state_dict(checkpoint["local_adapter_state_dict"])
                    print(f"[resume] Restored local adapter weights (Exp9)")
            elif "adapter_state_dict" in checkpoint:
                model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
                print(f"[resume] Restored adapter weights (Exp2 joint training)")

        # Restore optimizer state
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[resume] Restored optimizer state")

        # Restore training progress
        if "epoch" in checkpoint:
            current_epoch = checkpoint["epoch"]

            # Check if this is an interrupt checkpoint (epoch not completed)
            # or a regular checkpoint (epoch completed)
            is_interrupt = "reason" in checkpoint or "interrupt" in ckpt_path.lower()

            if is_interrupt:
                # Interrupt checkpoint: continue current epoch
                start_epoch = current_epoch
                print(f"[resume] Interrupt detected. Continuing epoch {start_epoch} from step {checkpoint.get('global_step', 0)}")
            else:
                # Regular checkpoint: start next epoch
                start_epoch = current_epoch + 1
                print(f"[resume] Resuming from epoch {start_epoch} (completed epoch {current_epoch})")

        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
            print(f"[resume] Restored global_step: {global_step}")

        print(f"[resume] Resume complete. Starting training from epoch {start_epoch}")

    # Ensure checkpoint directory exists (used for both top-k and interrupt saves).
    ckpt_dir = os.path.join(train_args.output_dir, train_args.checkpoint_subdir)

    def save_interrupt_checkpoint(reason: str) -> str:
        """
        Save a lightweight checkpoint when the run is interrupted.

        Saves decoder + optimizer (decoder-only) so training can be resumed.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        ts = int(time.time())
        ckpt_path = os.path.join(
            ckpt_dir, f"interrupt_epoch={current_epoch:03d}_step={global_step}_{reason}_{ts}.pt"
        )

        # Build checkpoint dict
        ckpt_dict = {
            "epoch": current_epoch,
            "global_step": global_step,
            "reason": reason,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_args": vars(model_args),
            "data_args": vars(data_args),
            "train_args": vars(train_args),
        }

        # Save decoder weights (handle separate adapters for Exp9)
        if model_args.separate_adapters:
            ckpt_dict["global_decoder_state_dict"] = model.global_decoder.state_dict()
            ckpt_dict["local_decoder_state_dict"] = model.local_decoder.state_dict()
        else:
            ckpt_dict["decoder_state_dict"] = model.decoder.state_dict()

        # Save adapter weights if we're training it (Exp2/Exp9)
        if train_args.train_adapter:
            if model_args.separate_adapters:
                ckpt_dict["global_adapter_state_dict"] = model.global_adapter.state_dict()
                ckpt_dict["local_adapter_state_dict"] = model.local_adapter.state_dict()
            else:
                ckpt_dict["adapter_state_dict"] = model.adapter.state_dict()

        torch.save(ckpt_dict, ckpt_path)
        print(f"[ckpt] interrupt checkpoint saved: {ckpt_path}")
        return ckpt_path

    def _request_stop(signum, _frame) -> None:
        """
        Request a graceful stop on SIGINT/SIGTERM.

        We avoid heavy I/O directly inside the signal handler; the training loop
        will save and exit at the next safe point.
        """
        nonlocal stop_requested, stop_reason
        stop_requested = True
        try:
            stop_reason = signal.Signals(signum).name.lower()
        except ValueError:
            stop_reason = f"signal_{signum}"
        print(f"\n[signal] received {stop_reason}; saving checkpoint and exiting...")

        # Trigger an exception so we exit promptly even if waiting on DataLoader I/O.
        if signum == signal.SIGINT:
            signal.default_int_handler(signum, _frame)  # raises KeyboardInterrupt
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # ===== 14. TRAINING/VALIDATION FUNCTION =====
    def run_epoch(
        loader: DataLoader,
        train: bool,
        split: str,
        epoch_idx: int,
    ) -> Dict[str, float]:
        """
        Run one epoch of training or validation.

        Args:
            loader: DataLoader (train_loader or val_loader)
            train: True for training mode, False for evaluation mode
            split: Split name for logging ("train" or "val")
            epoch_idx: Epoch index (1-based)

        Returns:
            Dictionary of aggregated metrics
        """

        nonlocal global_step

        # Set model mode
        if train:
            model.train()  # Enable dropout, batch norm updates, etc.
        else:
            model.eval()   # Disable dropout, batch norm in eval mode

        # Initialize metric accumulators
        agg = {
            "loss": 0.0,       # Total reconstruction loss
            "mse": 0.0,        # Mean squared error
            "cos": 0.0,        # Cosine similarity
            "top1": 0.0,       # Top-1% worst L2 error
            "reg_cos": 0.0,    # Region-specific cosine similarity
            "reg_top1": 0.0,   # Region-specific top-1% error
            "count": 0,        # Number of global image samples
            "reg_count": 0,    # Number of region samples
        }

        # ===== REGION-LEVEL STATISTICS (for size-cos correlation analysis) =====
        # Track per-region metrics to analyze relationship between region size and reconstruction quality
        region_stats = {
            region_name: {
                "cos_values": [],      # List of cosine similarity values per step
                "mse_values": [],      # List of MSE values per step
                "top1_values": [],     # List of top-1% error values per step
                "volume_sizes": [],    # List of mask-based volume sizes (in voxels)
                "compression_ratios": [],  # List of compression ratios (original_bbox / target)
                "cos_sum": 0.0,        # Running sum for epoch average
                "mse_sum": 0.0,
                "top1_sum": 0.0,
                "size_sum": 0,         # Running sum of volume sizes
                "compression_ratio_sum": 0.0,  # Running sum of compression ratios
                "count": 0,            # Number of samples for this region
            }
            for region_name in REGIONS
        }

        # Gradient accumulation counter
        step_in_accum = 0

        # Zero gradients at start of epoch
        if train:
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True: more memory efficient

        # ===== BATCH LOOP =====
        it = loader
        if train_args.show_progress and tqdm is not None:
            it = tqdm(
                loader,
                total=len(loader),
                desc=f"[{split}] epoch {epoch_idx}/{train_args.num_train_epochs}",
                dynamic_ncols=True,
                leave=False,
            )

        for step, batch in enumerate(it):
            if stop_requested:
                raise KeyboardInterrupt
            if train:
                global_step += 1

            # Move batch to GPU
            # Dict comprehension: transfer each tensor to device
            vision_x = {k: v.to(device) for k, v in batch["vision_x"].items()}
            # Also move masks to GPU for accurate region size calculation
            mask_x = {k: v.to(device) for k, v in batch["mask_x"].items()}
            # Extract bbox_sizes for compression ratio analysis (list of dicts, one per sample)
            bbox_sizes_batch = batch.get("bbox_sizes", [{}] * len(batch["vision_x"]["image"]))

            # Initialize batch loss
            loss_total = 0.0

            # ===== GLOBAL IMAGE RECONSTRUCTION =====
            # Process full CT image (not region-specific)

            # Encode: CT volume → tokens
            # torch.no_grad(): disable gradient computation (encoder is frozen)
            with torch.no_grad():
                x_g, grid = model.encode_tokens(vision_x["image"])

            # Compress: tokens → compressed representation
            # Gradients computed here in training mode (though adapter is frozen)
            z_g = model.compress_tokens(x_g, is_global=True)

            # Decode: compressed → reconstructed tokens
            # Gradients computed here (decoder is trainable)
            x_hat_g = model.decode_tokens(z_g, grid, is_global=True)

            # Calculate reconstruction loss
            loss_g, mse_g, cos_g, top1_g = recon_loss(x_hat_g, x_g, ln)

            # Add to total loss
            loss_total = loss_g

            # Accumulate metrics
            agg["loss"] += loss_g.item()  # .item(): convert to Python float
            agg["mse"] += mse_g
            agg["cos"] += cos_g
            agg["top1"] += top1_g
            agg["count"] += 1

            # ===== REGION-SPECIFIC RECONSTRUCTION =====
            # Process each anatomical region separately

            reg_losses = []   # Store region losses
            reg_cos = []      # Store region cosine similarities
            reg_top1 = []     # Store region top-1% errors

            # Iterate over regions in batch
            for key, vol in vision_x.items():
                # Skip full image (already processed above)
                if key == "image":
                    continue

                # Check which samples in batch contain this region
                # .abs().sum(): sum of absolute values across all spatial dimensions
                # > 0: True if region is present (non-zero), False if zero-padded
                present = vol.abs().sum(dim=(1, 2, 3, 4, 5)) > 0

                # Skip if no samples in batch have this region
                if not present.any():  # .any(): True if at least one True
                    continue

                # Process only samples that have this region
                # vol[present]: boolean indexing to select present samples
                with torch.no_grad():
                    x_r, reg_grid = model.encode_tokens(vol[present])

                # Exp9: Use local adapter/decoder for regions (is_global=False)
                z_r = model.compress_tokens(x_r, is_global=False)
                x_hat_r = model.decode_tokens(z_r, reg_grid, is_global=False)

                # Calculate reconstruction loss for this region
                loss_r, mse_r, cos_r, top1_r = recon_loss(x_hat_r, x_r, ln)

                # ===== MASK-BASED SIZE CALCULATION =====
                # Use binary mask to get accurate anatomical region size (in voxels)
                # This is more accurate than bounding box size which includes padding
                if key in mask_x:
                    region_mask = mask_x[key][present]
                    # Sum of non-zero voxels across all spatial dimensions
                    # shape: (B, 1, H, W, D) -> scalar after sum
                    mask_size = int((region_mask > 0).sum().item())
                else:
                    # Fallback: use non-zero volume size if mask unavailable
                    mask_size = int((vol[present].abs() > 1e-6).sum().item())

                # ===== COMPRESSION RATIO FROM BBOX_SIZES =====
                # Extract average compression ratio for this region from batch
                # compression_ratio > 1 means original was larger (info compressed/lost)
                # compression_ratio < 1 means original was smaller (info expanded)
                compression_ratios_for_region = []
                present_indices = present.nonzero(as_tuple=True)[0].tolist()
                for idx in present_indices:
                    if idx < len(bbox_sizes_batch) and key in bbox_sizes_batch[idx]:
                        compression_ratios_for_region.append(
                            bbox_sizes_batch[idx][key].get("compression_ratio", 1.0)
                        )
                avg_compression_ratio = (
                    sum(compression_ratios_for_region) / len(compression_ratios_for_region)
                    if compression_ratios_for_region else 1.0
                )

                # ===== RECORD PER-REGION STATISTICS =====
                # Store metrics for this region (for correlation analysis)
                if key in region_stats:
                    region_stats[key]["cos_values"].append(cos_r)
                    region_stats[key]["mse_values"].append(mse_r)
                    region_stats[key]["top1_values"].append(top1_r)
                    region_stats[key]["volume_sizes"].append(mask_size)
                    region_stats[key]["compression_ratios"].append(avg_compression_ratio)
                    region_stats[key]["cos_sum"] += cos_r
                    region_stats[key]["mse_sum"] += mse_r
                    region_stats[key]["top1_sum"] += top1_r
                    region_stats[key]["size_sum"] += mask_size
                    region_stats[key]["compression_ratio_sum"] += avg_compression_ratio
                    region_stats[key]["count"] += 1

                # Accumulate region metrics
                reg_losses.append(loss_r)
                reg_cos.append(cos_r)
                reg_top1.append(top1_r)

            # ===== COMBINE REGION LOSSES =====
            if reg_losses:
                # Stack list of tensors into single tensor
                # torch.stack: concatenate along new dimension
                # .mean(): average across regions
                loss_r = torch.stack(reg_losses).mean()

                # Add weighted region loss to total
                # lambda_region: controls importance of region reconstruction
                loss_total = loss_total + train_args.lambda_region * loss_r

                # Accumulate region metrics
                agg["reg_cos"] += sum(reg_cos) / len(reg_cos)
                agg["reg_top1"] += sum(reg_top1) / len(reg_top1)
                agg["reg_count"] += 1

            # ===== OPTIMIZATION STEP =====
            if train:
                # Scale loss for gradient accumulation
                # When accumulating, divide loss by number of accumulation steps
                # This keeps effective learning rate consistent
                loss_total = loss_total / train_args.gradient_accumulation_steps

                # Backward pass: compute gradients
                # .backward(): compute ∂loss/∂params for all trainable parameters
                loss_total.backward()

                # Increment accumulation counter
                step_in_accum += 1

                # Update weights if we've accumulated enough gradients
                if step_in_accum >= train_args.gradient_accumulation_steps:
                    # optimizer.step(): update parameters using computed gradients
                    # θ_new = θ_old - lr * gradient
                    optimizer.step()

                    # Zero gradients for next accumulation cycle
                    optimizer.zero_grad(set_to_none=True)

                    # Reset counter
                    step_in_accum = 0

                    # ===== STEP-LEVEL VALIDATION CHECK =====
                    # Run validation every N steps if val_check_interval is set
                    if train_args.val_check_interval > 0:
                        if global_step - last_val_check_step >= train_args.val_check_interval:
                            # Temporarily save model state
                            model.train()  # Ensure we're in train mode before validation

                            # Run validation
                            run_validation_and_checkpoint(epoch_idx, force_check=False)

                            # Check early stopping
                            if train_args.early_stopping_patience > 0:
                                if no_improvement_count >= train_args.early_stopping_patience:
                                    print(f"[early_stop] 🛑 Stopping training at step {global_step}")
                                    raise StopIteration  # Break out of nested loops

                            # Return to training mode
                            model.train()

            # ===== LOGGING =====
            # Print progress every N steps during training
            if train and (step + 1) % train_args.log_every == 0:
                print(
                    f"[train] step={step+1} loss={loss_total.item():.4f} "
                    f"cos={cos_g:.4f} reg_cos={agg['reg_cos'] / max(1, agg['reg_count']):.4f}"
                )

                # ===== STEP-LEVEL REGION DETAILS =====
                # Print per-region metrics for debugging and analysis
                region_detail_parts = []
                for rname in REGIONS:
                    rs = region_stats[rname]
                    if rs["count"] > 0:
                        avg_cos = rs["cos_sum"] / rs["count"]
                        avg_size_k = (rs["size_sum"] / rs["count"]) / 1000  # Convert to K voxels
                        region_detail_parts.append(f"{rname[:4]}:{avg_cos:.2f}@{avg_size_k:.0f}K")
                if region_detail_parts:
                    print(f"    [regions] {' | '.join(region_detail_parts)}")

                if wandb_run is not None:
                    # Calculate current batch reg_cos (instant value)
                    current_reg_cos = sum(reg_cos) / len(reg_cos) if reg_cos else 0.0

                    wandb_log_dict = {
                        # Instant values (current batch) - for observing training dynamics
                        "train/loss_step": float(loss_total.item()),
                        "train/mse_step": float(mse_g),
                        "train/cos_step": float(cos_g),
                        "train/top1_step": float(top1_g),
                        "train/reg_cos_step": float(current_reg_cos),
                        "train/reg_top1_step": float(sum(reg_top1) / len(reg_top1) if reg_top1 else 0.0),

                        # Cumulative averages (from epoch start) - for smooth trends
                        "train/loss_running": float(agg["loss"] / max(1, agg["count"])),
                        "train/mse_running": float(agg["mse"] / max(1, agg["count"])),
                        "train/cos_running": float(agg["cos"] / max(1, agg["count"])),
                        "train/top1_running": float(agg["top1"] / max(1, agg["count"])),
                        "train/reg_cos_running": float(agg["reg_cos"] / max(1, agg["reg_count"])),
                        "train/reg_top1_running": float(agg["reg_top1"] / max(1, agg["reg_count"])),

                        # Learning rate
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    }

                    # ===== PER-REGION WANDB LOGGING =====
                    # Log running average for each region (for size-cos analysis)
                    for rname in REGIONS:
                        rs = region_stats[rname]
                        if rs["count"] > 0:
                            wandb_log_dict[f"train/region_{rname}_cos"] = float(rs["cos_sum"] / rs["count"])
                            wandb_log_dict[f"train/region_{rname}_size"] = float(rs["size_sum"] / rs["count"])

                    wandb_run.log(wandb_log_dict, step=global_step)

            if train_args.show_progress and tqdm is not None:
                running = {
                    "loss": agg["loss"] / max(1, agg["count"]),
                    "cos": agg["cos"] / max(1, agg["count"]),
                    "reg_cos": agg["reg_cos"] / max(1, agg["reg_count"]),
                }
                try:
                    it.set_postfix(
                        loss=f"{running['loss']:.4f}",
                        cos=f"{running['cos']:.4f}",
                        reg_cos=f"{running['reg_cos']:.4f}",
                    )
                except Exception:
                    pass

        # ===== FINAL GRADIENT UPDATE =====
        # If there are leftover accumulated gradients at epoch end, update now
        if train and step_in_accum > 0:
            optimizer.step()

        # ===== COMPUTE EPOCH AVERAGES =====
        # Divide accumulated metrics by counts to get averages
        denom = max(1, agg["count"])         # Prevent division by zero
        reg_denom = max(1, agg["reg_count"])

        # ===== EPOCH-LEVEL REGION STATISTICS SUMMARY =====
        # Compute and print per-region epoch averages for size-cos analysis
        print(f"\n{'='*80}")
        print(f"[{split}] EPOCH {epoch_idx} REGION STATISTICS")
        print(f"{'='*80}")
        print(f"{'Region':<20} {'Cos':>8} {'MSE':>8} {'Top1':>8} {'Size(K)':>10} {'Comp.Ratio':>12} {'Count':>6}")
        print("-" * 80)

        # Collect region-level epoch averages for correlation analysis
        region_epoch_summary = {}
        sizes_for_corr = []
        cos_for_corr = []
        compression_ratios_for_corr = []

        for rname in REGIONS:
            rs = region_stats[rname]
            if rs["count"] > 0:
                avg_cos = rs["cos_sum"] / rs["count"]
                avg_mse = rs["mse_sum"] / rs["count"]
                avg_top1 = rs["top1_sum"] / rs["count"]
                avg_size = rs["size_sum"] / rs["count"]
                avg_size_k = avg_size / 1000  # Convert to K voxels
                avg_compression_ratio = rs["compression_ratio_sum"] / rs["count"]

                region_epoch_summary[rname] = {
                    "cos": avg_cos,
                    "mse": avg_mse,
                    "top1": avg_top1,
                    "avg_size": avg_size,
                    "avg_compression_ratio": avg_compression_ratio,
                    "count": rs["count"],
                }

                # Collect for correlation analysis
                sizes_for_corr.append(avg_size)
                cos_for_corr.append(avg_cos)
                compression_ratios_for_corr.append(avg_compression_ratio)

                # Print with compression ratio indicator
                # >1 = compressed (🔻 info loss), <1 = expanded (🔺 no loss)
                ratio_indicator = "🔻" if avg_compression_ratio > 1.0 else "🔺"
                print(f"{rname:<20} {avg_cos:>8.4f} {avg_mse:>8.4f} {avg_top1:>8.4f} {avg_size_k:>10.1f} {avg_compression_ratio:>10.2f}x{ratio_indicator} {rs['count']:>6}")
            else:
                print(f"{rname:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>12} {0:>6}")

        # ===== COMPRESSION RATIO-COS CORRELATION ANALYSIS =====
        # This is the key analysis: does compression ratio affect reconstruction quality?
        if len(compression_ratios_for_corr) >= 3:  # Need at least 3 points
            import numpy as np
            ratios_arr = np.array(compression_ratios_for_corr)
            cos_arr = np.array(cos_for_corr)
            sizes_arr = np.array(sizes_for_corr)

            # Helper function for Pearson correlation
            def pearson_corr(x, y):
                mean_x, mean_y = x.mean(), y.mean()
                numerator = ((x - mean_x) * (y - mean_y)).sum()
                denom_x = np.sqrt(((x - mean_x) ** 2).sum())
                denom_y = np.sqrt(((y - mean_y) ** 2).sum())
                if denom_x > 0 and denom_y > 0:
                    return numerator / (denom_x * denom_y)
                return 0.0

            # Correlation 1: Compression Ratio vs Cosine
            r_ratio_cos = pearson_corr(ratios_arr, cos_arr)

            # Correlation 2: Size vs Cosine (original analysis)
            r_size_cos = pearson_corr(sizes_arr, cos_arr)

            print("-" * 80)
            print(f"📊 CORRELATION ANALYSIS:")
            print(f"   Compression Ratio vs Cos (Pearson r): {r_ratio_cos:+.4f}")
            if r_ratio_cos < -0.3:
                print(f"      → ⚠️  Higher compression = WORSE reconstruction (info loss confirmed!)")
            elif r_ratio_cos > 0.3:
                print(f"      → Higher compression = better reconstruction (unexpected)")
            else:
                print(f"      → No strong correlation")

            print(f"   Size vs Cos (Pearson r): {r_size_cos:+.4f}")
            if r_size_cos > 0.3:
                print(f"      → Larger regions easier to reconstruct")
            elif r_size_cos < -0.3:
                print(f"      → Smaller regions easier to reconstruct (denser info)")
            else:
                print(f"      → No strong correlation")

        print(f"{'='*80}\n")

        return {
            "loss": agg["loss"] / denom,
            "mse": agg["mse"] / denom,
            "cos": agg["cos"] / denom,
            "top1": agg["top1"] / denom,
            "reg_cos": agg["reg_cos"] / reg_denom,
            "reg_top1": agg["reg_top1"] / reg_denom,
            "region_stats": region_epoch_summary,  # Add per-region statistics
        }

    # ===== 15. METRICS FILE INITIALIZATION =====
    # Create CSV file for logging metrics
    metrics_path = os.path.join(train_args.output_dir, "lit_metrics.csv")

    # Open file in write mode and write header
    # encoding="utf-8": handle special characters
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,split,loss,mse,cos,top1,reg_cos,reg_top1,decode_mode\n"
        )

    # ===== 15.1. REGION-LEVEL METRICS CSV =====
    # Create CSV file for per-region metrics (for size-cos correlation analysis)
    region_metrics_path = os.path.join(train_args.output_dir, "region_metrics.csv")
    with open(region_metrics_path, "w", encoding="utf-8") as f:
        f.write("epoch,split,region,cos,mse,top1,avg_size,compression_ratio,count\n")

    # ===== 15.5. CHECKPOINT MANAGER (TOP-K BEST) =====
    best_checkpoints: List[Tuple[float, int, str]] = []
    save_top_k = max(0, int(train_args.save_top_k))
    monitor_metric = train_args.monitor_metric
    monitor_mode = train_args.monitor_mode.lower().strip()
    if monitor_mode not in {"max", "min"}:
        raise ValueError(f"Invalid monitor_mode: {train_args.monitor_mode}. Use 'max' or 'min'.")

    if save_top_k > 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    def _is_better(a: float, b: float) -> bool:
        return a > b if monitor_mode == "max" else a < b

    def maybe_save_topk(epoch: int, train_m: Dict[str, float], val_m: Dict[str, float]) -> None:
        nonlocal best_checkpoints
        if save_top_k <= 0:
            return
        if monitor_metric not in val_m:
            raise KeyError(
                f"monitor_metric='{monitor_metric}' not found in val metrics. "
                f"Available: {sorted(val_m.keys())}"
            )

        score = float(val_m[monitor_metric])
        if best_checkpoints and len(best_checkpoints) >= save_top_k:
            worst_score = best_checkpoints[-1][0]
            if not _is_better(score, worst_score):
                return

        ckpt_name = f"epoch={epoch:03d}_val_{monitor_metric}={score:.6f}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        torch.save(
            {
                "epoch": epoch,
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
                "monitor_score": score,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_args": vars(model_args),
                "data_args": vars(data_args),
                "train_args": vars(train_args),
                "train_metrics": train_m,
                "val_metrics": val_m,
            },
            ckpt_path,
        )
        best_checkpoints.append((score, epoch, ckpt_path))
        best_checkpoints.sort(key=lambda x: x[0], reverse=(monitor_mode == "max"))

        for _, _, path in best_checkpoints[save_top_k:]:
            try:
                os.remove(path)
            except OSError:
                pass
        best_checkpoints = best_checkpoints[:save_top_k]
        print(f"[ckpt] saved {ckpt_path}")

    # ===== 15.6. EARLY STOPPING TRACKER =====
    best_val_score: Optional[float] = None
    no_improvement_count = 0
    last_val_check_step = 0  # Track when we last ran validation
    if train_args.early_stopping_patience > 0:
        print(f"[early_stop] Enabled with patience={train_args.early_stopping_patience}, "
              f"monitoring {monitor_metric} ({monitor_mode})")

    def run_validation_and_checkpoint(epoch: int, force_check: bool = False) -> Dict[str, float]:
        """
        Run validation and handle checkpointing and early stopping.

        Args:
            epoch: Current epoch number
            force_check: Force validation even if val_check_interval not reached

        Returns:
            Validation metrics dictionary
        """
        nonlocal best_val_score, no_improvement_count, last_val_check_step

        # Check if we should run validation
        should_validate = force_check
        if train_args.val_check_interval > 0 and not force_check:
            if global_step - last_val_check_step >= train_args.val_check_interval:
                should_validate = True

        if not should_validate:
            return {}

        # Run validation
        print(f"[validation] Running validation at step {global_step} (epoch {epoch})")
        val_metrics = run_epoch(val_loader, train=False, split="val", epoch_idx=epoch)
        last_val_check_step = global_step

        # Log to W&B
        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/loss_check": float(val_metrics["loss"]),
                    "val/mse_check": float(val_metrics["mse"]),
                    "val/cos_check": float(val_metrics["cos"]),
                    "val/top1_check": float(val_metrics["top1"]),
                    "val/reg_cos_check": float(val_metrics["reg_cos"]),
                    "val/reg_top1_check": float(val_metrics["reg_top1"]),
                    f"val/{monitor_metric}_check": float(val_metrics.get(monitor_metric, 0.0)),
                },
                step=global_step,
            )

        # Save checkpoint if best
        train_metrics_dummy = {}  # For checkpointing
        maybe_save_topk(epoch, train_metrics_dummy, val_metrics)

        # Early stopping check
        if train_args.early_stopping_patience > 0:
            current_val_score = float(val_metrics[monitor_metric])

            # Check if this is an improvement
            is_improvement = False
            if best_val_score is None:
                is_improvement = True
            else:
                is_improvement = _is_better(current_val_score, best_val_score)

            if is_improvement:
                best_val_score = current_val_score
                no_improvement_count = 0
                print(f"[early_stop] ✅ New best {monitor_metric}={best_val_score:.6f} at step {global_step}")
            else:
                no_improvement_count += 1
                print(f"[early_stop] ⚠️  No improvement for {no_improvement_count} check(s) "
                      f"(best={best_val_score:.6f}, current={current_val_score:.6f})")

        return val_metrics

    # ===== 16. MAIN TRAINING LOOP =====
    # Iterate over epochs (full passes through training data)
    training_interrupted = False
    try:
        for epoch in range(start_epoch, train_args.num_train_epochs + 1):
            current_epoch = epoch

            # Run one epoch of training
            try:
                train_metrics = run_epoch(train_loader, train=True, split="train", epoch_idx=epoch)
            except StopIteration:
                # Early stopping triggered during training
                print(f"[early_stop] Training stopped by early stopping at epoch {epoch}")
                training_interrupted = True
                break

            # Run epoch-end validation (always do this, regardless of val_check_interval)
            val_metrics = run_validation_and_checkpoint(epoch, force_check=True)

            # Print epoch summary
            print(
                f"[epoch {epoch}] train_cos={train_metrics['cos']:.4f} "
                f"train_reg_cos={train_metrics['reg_cos']:.4f} "
                f"val_cos={val_metrics['cos']:.4f} "
                f"val_reg_cos={val_metrics['reg_cos']:.4f}"
            )

            # W&B epoch-level logging
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": float(train_metrics["loss"]),
                        "train/mse": float(train_metrics["mse"]),
                        "train/cos": float(train_metrics["cos"]),
                        "train/top1": float(train_metrics["top1"]),
                        "train/reg_cos": float(train_metrics["reg_cos"]),
                        "train/reg_top1": float(train_metrics["reg_top1"]),
                        "val/loss": float(val_metrics["loss"]),
                        "val/mse": float(val_metrics["mse"]),
                        "val/cos": float(val_metrics["cos"]),
                        "val/top1": float(val_metrics["top1"]),
                        "val/reg_cos": float(val_metrics["reg_cos"]),
                        "val/reg_top1": float(val_metrics["reg_top1"]),
                        f"val/{monitor_metric}": float(val_metrics.get(monitor_metric, 0.0)),
                    },
                    step=global_step,
                )

            # Check early stopping (epoch-level)
            if train_args.early_stopping_patience > 0:
                if no_improvement_count >= train_args.early_stopping_patience:
                    print(f"[early_stop] 🛑 Stopping training: no improvement for {train_args.early_stopping_patience} validation(s)")
                    training_interrupted = True
                    break

            # ===== 17. LOG METRICS TO CSV =====
            # Append mode: add to end of file without overwriting
            with open(metrics_path, "a", encoding="utf-8") as f:
                # Write training metrics
                f.write(
                    f"{epoch},train,{train_metrics['loss']:.6f},"
                    f"{train_metrics['mse']:.6f},{train_metrics['cos']:.6f},"
                    f"{train_metrics['top1']:.6f},{train_metrics['reg_cos']:.6f},"
                    f"{train_metrics['reg_top1']:.6f},{model_args.decode_mode}\n"
                )

                # Write validation metrics
                f.write(
                    f"{epoch},val,{val_metrics['loss']:.6f},"
                    f"{val_metrics['mse']:.6f},{val_metrics['cos']:.6f},"
                    f"{val_metrics['top1']:.6f},{val_metrics['reg_cos']:.6f},"
                    f"{val_metrics['reg_top1']:.6f},{model_args.decode_mode}\n"
                )

            # ===== 17.1. LOG REGION-LEVEL METRICS TO CSV =====
            # Write per-region statistics for size-cos correlation analysis
            with open(region_metrics_path, "a", encoding="utf-8") as f:
                # Write training region metrics
                if "region_stats" in train_metrics:
                    for rname, rstats in train_metrics["region_stats"].items():
                        f.write(
                            f"{epoch},train,{rname},"
                            f"{rstats['cos']:.6f},{rstats['mse']:.6f},"
                            f"{rstats['top1']:.6f},{rstats['avg_size']:.1f},"
                            f"{rstats.get('avg_compression_ratio', 1.0):.4f},"
                            f"{rstats['count']}\n"
                        )

                # Write validation region metrics
                if "region_stats" in val_metrics:
                    for rname, rstats in val_metrics["region_stats"].items():
                        f.write(
                            f"{epoch},val,{rname},"
                            f"{rstats['cos']:.6f},{rstats['mse']:.6f},"
                            f"{rstats['top1']:.6f},{rstats['avg_size']:.1f},"
                            f"{rstats.get('avg_compression_ratio', 1.0):.4f},"
                            f"{rstats['count']}\n"
                        )

    except (KeyboardInterrupt, SystemExit):
        save_interrupt_checkpoint(stop_reason)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    """
    CLI entry point: parse arguments and run training.

    This function parses command-line arguments using HfArgumentParser,
    then calls run_training() with the parsed arguments.

    Usage:
        python lit_recon_probe.py --adapter_depth 1 --separate_adapters True ...

    For experiment-based running, use run_experiment.py instead:
        python run_experiment.py exp9
    """
    # HfArgumentParser: Hugging Face utility for parsing dataclass arguments
    # Parses command-line args or JSON config files into dataclass instances
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # Run training with parsed arguments
    run_training(model_args, data_args, train_args)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Script entry point.

    When this file is run directly (not imported), execute main().
    Standard Python idiom for executable scripts.
    """
    main()
