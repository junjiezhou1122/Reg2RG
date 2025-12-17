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
from dataclasses import dataclass, field  # For creating configuration classes
from typing import Dict, List, Optional, Tuple  # Type hints for better code documentation

import torch  # PyTorch main library for tensor operations and neural networks
import torch.nn.functional as F  # Functional interface for operations like MSE, normalize
from torch import nn  # Neural network modules (Linear, LayerNorm, etc.)
from torch.utils.data import DataLoader, random_split  # Data loading and splitting utilities
import transformers  # Hugging Face library for argument parsing

# Custom dataset for loading RadGenome CT scans and region masks
from Dataset.radgenome_dataset_train import RadGenomeDataset_Train
# Perceiver Resampler: compresses variable-length sequences to fixed-length
from Model.helpers import PerceiverResampler
# 3D positional encodings for volumetric data
from Model.position_encoding import PositionEmbeddingLearned3d
# 3D Vision Transformer for processing CT volumes
from Model.vit_3d import ViT


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


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass  # Decorator that auto-generates __init__, __repr__, etc.
class ModelArguments:
    """
    Model architecture and pretrained weights configuration.

    This dataclass holds all hyperparameters that define the model structure
    and paths to pretrained checkpoints.
    """

    # Path to tokenizer (used for text preprocessing in the full pipeline)
    # field() allows setting default values and metadata
    tokenizer_path: str = field(
        default="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf",
        metadata={"help": "Path to the tokenizer (for dataset preprocessing)."},
    )

    # Path to pretrained 3D ViT encoder weights
    # These weights are frozen during training to test information preservation
    pretrained_visual_encoder: str = field(
        default="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth",
        metadata={"help": "Path to pretrained ViT3D weights."},
    )

    # Path to pretrained Perceiver adapter and projection layer weights
    pretrained_adapter: str = field(
        default="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth",
        metadata={"help": "Path to pretrained Perceiver adapter weights."},
    )

    # Whether to load pretrained adapter weights
    use_pretrained_adapter: bool = field(
        default=True,
        metadata={"help": "Load pretrained Perceiver weights if available."},
    )

    # Decode mode: where to decode from in the pipeline
    # "pre_proj": decode from Perceiver output (before projection to LLM space)
    # "post_proj": decode from projected representation (after FC layer)
    decode_mode: str = field(
        default="pre_proj",
        metadata={"help": "Decode from 'pre_proj' (Z) or 'post_proj' (fc(Z))."},
    )

    # Number of latent tokens in Perceiver output (compressed representation size)
    perceiver_num: int = field(default=32, metadata={"help": "Latent token count."})

    # Dimension of ViT token embeddings (input to Perceiver)
    vis_dim: int = field(default=768, metadata={"help": "ViT token dimension."})

    # Dimension of LLM embeddings (output of projection layer)
    llm_dim: int = field(default=4096, metadata={"help": "Projection output dimension."})

    # Number of transformer layers in the reconstruction decoder
    decoder_layers: int = field(default=2, metadata={"help": "Decoder depth."})

    # Number of attention heads in decoder layers
    decoder_heads: int = field(default=8, metadata={"help": "Decoder heads."})

    # Feed-forward network expansion factor in decoder
    # FFN hidden dimension = decoder_ff_mult * vis_dim
    decoder_ff_mult: int = field(default=4, metadata={"help": "Decoder FFN multiplier."})


@dataclass
class DataArguments:
    """
    Dataset paths and data loading configuration.

    Specifies where to find CT scans, region masks, and reports.
    """

    # Directory containing preprocessed CT scan volumes
    data_folder: str = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed"
    )

    # Directory containing binary masks for anatomical regions
    mask_folder: str = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask"
    )

    # CSV file with region-specific radiology reports
    report_file: str = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv"
    )

    # Optional: dedicated validation set paths (preferred over splitting train)
    val_data_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Validation CT volume directory. If set, do not split train."},
    )
    val_mask_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Validation region mask directory. If set, do not split train."},
    )
    val_report_file: Optional[str] = field(
        default=None,
        metadata={"help": "Validation report CSV. If set, do not split train."},
    )

    # Directory for MONAI dataset caching (speeds up data loading)
    monai_cache_dir: str = field(default="/mnt2/ct/RadGenome-ChestCT/cache")

    # Fraction of train data to use for validation (used only if val_* not set)
    val_split: float = field(
        default=0.05,
        metadata={"help": "Fraction of data used for validation."},
    )


@dataclass
class TrainArguments:
    """
    Training hyperparameters and logging configuration.

    Controls the training process: learning rate, epochs, batch size, etc.
    """

    # Directory to save training metrics and model checkpoints
    output_dir: str = field(
        default="/mnt/home/zhoujunjie/outputs/LIT",
        metadata={"help": "Where to write metrics and checkpoints."},
    )

    # Total number of training epochs (full passes through dataset)
    num_train_epochs: int = field(default=15)

    # Learning rate for AdamW optimizer
    learning_rate: float = field(default=1e-4)

    # Weight decay for L2 regularization in optimizer
    weight_decay: float = field(default=0.01)

    # Number of samples processed together in each training step
    batch_size: int = field(default=1)

    # Number of parallel workers for data loading
    dataloader_num_workers: int = field(default=4)

    # Number of steps to accumulate gradients before updating weights
    # Effective batch size = batch_size * gradient_accumulation_steps
    gradient_accumulation_steps: int = field(default=8)

    # Log training metrics every N steps
    log_every: int = field(default=10)

    # Random seed for reproducibility
    seed: int = field(default=42)

    # Weight for region-specific reconstruction loss
    # Total loss = global_loss + lambda_region * region_loss
    lambda_region: float = field(default=1.0)

    # ===== Experiment Tracking (Weights & Biases) =====
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Enable Weights & Biases logging (requires wandb installed)."},
    )
    wandb_project: str = field(
        default="Reg2RG-LIT",
        metadata={"help": "W&B project name."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "W&B entity/team (optional)."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "W&B run name (optional)."},
    )
    wandb_mode: str = field(
        default="online",
        metadata={"help": "W&B mode: online | offline | disabled."},
    )

    # ===== Checkpointing =====
    monitor_metric: str = field(
        default="reg_cos",
        metadata={"help": "Validation metric key to monitor for best checkpoints."},
    )
    monitor_mode: str = field(
        default="max",
        metadata={"help": "max to maximize metric, min to minimize metric."},
    )
    save_top_k: int = field(
        default=3,
        metadata={"help": "Keep top-k best checkpoints by monitor_metric (0 disables)."},
    )
    checkpoint_subdir: str = field(
        default="checkpoints",
        metadata={"help": "Subdirectory under output_dir to store checkpoints."},
    )


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
                - region2area: Metadata about regions (not used in this function)

        Returns:
            Batched dictionary with keys:
                - vision_x: Dict of batched region volumes
                - mask_x: Dict of batched region masks
                - region2area: List of metadata
        """

        # Extract the three components from each instance
        # tuple() is used to unpack a generator into separate lists
        vision_xs, mask_xs, region2areas = tuple(
            [instance[key] for instance in instances]  # List comprehension
            for key in ("vision_x", "mask_x", "region2area")  # For each key
        )

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
        return dict(vision_x=vision_xs, mask_x=mask_xs, region2area=region2areas)


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

        # ===== 2. ADAPTER (Perceiver Resampler) =====
        # Compresses variable-length token sequence to fixed number of latents
        self.adapter = PerceiverResampler(
            dim=vis_dim,                # Input/output dimension
            num_latents=perceiver_num   # Number of latent tokens (compression target)
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

    def compress_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress token sequence using Perceiver adapter.

        Args:
            tokens: Token sequence, shape (batch, num_tokens, dim)

        Returns:
            Compressed representation, shape (batch, perceiver_num, dim or llm_dim)
        """

        # Perceiver expects shape: (batch, num_media, num_frames, num_tokens, dim)
        # unsqueeze(1).unsqueeze(1): add two dimensions
        # (b, n, d) → (b, 1, 1, n, d)
        x = tokens.unsqueeze(1).unsqueeze(1)

        # Perceiver Resampler: compress to fixed number of latent tokens
        # Output: (batch, 1, perceiver_num, dim)
        # squeeze(1): remove media dimension → (batch, perceiver_num, dim)
        z = self.adapter(x).squeeze(1)

        # If decode mode is "post_proj", project to LLM space
        if self.decode_mode == "post_proj":
            z = self.fc(z)  # (batch, perceiver_num, llm_dim)

        return z

    def decode_tokens(
        self, memory: torch.Tensor, grid: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Reconstruct original tokens from compressed representation.

        Args:
            memory: Compressed representation, shape (batch, perceiver_num, dim)
            grid: Tuple of patch counts for reconstruction

        Returns:
            Reconstructed tokens, shape (batch, h*w*d, vis_dim)
        """

        # Project memory back to vision space if needed
        # pre_proj: memory is already in vis_dim space (Identity does nothing)
        # post_proj: memory is in llm_dim space, project back to vis_dim
        memory = self.mem_proj(memory)

        # Use decoder to reconstruct tokens via cross-attention
        return self.decoder(memory, grid, memory.shape[0])


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


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main() -> None:
    """
    Main training function: setup, training loop, and evaluation.

    Overall flow:
    1. Parse command-line arguments
    2. Set random seeds for reproducibility
    3. Load dataset and split into train/val
    4. Initialize model and load pretrained weights
    5. Freeze encoder/adapter, train only decoder
    6. Training loop: forward pass, compute loss, backward pass, optimize
    7. Log metrics and save results
    """

    # ===== 1. ARGUMENT PARSING =====
    # HfArgumentParser: Hugging Face utility for parsing dataclass arguments
    # Parses command-line args or JSON config files into dataclass instances
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # ===== 2. SET RANDOM SEEDS =====
    # Ensures reproducibility: same seed → same random numbers → same results
    torch.manual_seed(train_args.seed)              # CPU random number generator
    torch.cuda.manual_seed_all(train_args.seed)     # All GPU random number generators

    # ===== 3. DEVICE SETUP =====
    # torch.cuda.is_available(): check if CUDA-capable GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory if it doesn't exist
    # exist_ok=True: don't raise error if directory already exists
    os.makedirs(train_args.output_dir, exist_ok=True)

    # ===== 3.5. EXPERIMENT TRACKING (W&B) =====
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
    )

    # Validation data loader (same as train but no shuffling)
    val_loader = DataLoader(
        val_set,
        batch_size=train_args.batch_size,
        shuffle=False,                              # Don't shuffle validation
        num_workers=train_args.dataloader_num_workers,
        collate_fn=collator,
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
        adapter_ckpt = torch.load(model_args.pretrained_adapter, map_location="cpu")

        # Load Perceiver adapter weights
        if "perceiver" in adapter_ckpt:
            model.adapter.load_state_dict(adapter_ckpt["perceiver"])

        # Load projection layer weights (only for post_proj mode)
        if model_args.decode_mode == "post_proj" and "fc" in adapter_ckpt:
            model.fc.load_state_dict(adapter_ckpt["fc"])

    # ===== 10. MOVE MODEL TO GPU =====
    # .to(device): transfer all model parameters to specified device (CPU/GPU)
    model.to(device)

    # ===== 11. FREEZE PRETRAINED COMPONENTS =====
    # Only train the decoder; freeze encoder, adapter, and projection
    # This tests if information is linearly accessible (probe can't modify features)
    set_requires_grad(model.vision_encoder, False)  # Freeze ViT
    set_requires_grad(model.adapter, False)         # Freeze Perceiver
    set_requires_grad(model.fc, False)              # Freeze projection
    set_requires_grad(model.decoder, True)          # Train decoder only

    # ===== 12. OPTIMIZER SETUP =====
    # AdamW: Adam optimizer with weight decay (L2 regularization)
    # Only optimize decoder parameters (others are frozen)
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),          # Only decoder parameters
        lr=train_args.learning_rate,         # Learning rate
        weight_decay=train_args.weight_decay  # L2 regularization strength
    )

    # ===== 13. LAYER NORM FOR LOSS CALCULATION =====
    # Used in recon_loss to normalize features before MSE
    ln = nn.LayerNorm(model.vis_dim).to(device)

    # Global step counter for logging (increments per training batch)
    global_step = 0

    # ===== 14. TRAINING/VALIDATION FUNCTION =====
    def run_epoch(loader: DataLoader, train: bool) -> Dict[str, float]:
        """
        Run one epoch of training or validation.

        Args:
            loader: DataLoader (train_loader or val_loader)
            train: True for training mode, False for evaluation mode

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

        # Gradient accumulation counter
        step_in_accum = 0

        # Zero gradients at start of epoch
        if train:
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True: more memory efficient

        # ===== BATCH LOOP =====
        for step, batch in enumerate(loader):
            if train:
                global_step += 1

            # Move batch to GPU
            # Dict comprehension: transfer each tensor to device
            vision_x = {k: v.to(device) for k, v in batch["vision_x"].items()}

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
            z_g = model.compress_tokens(x_g)

            # Decode: compressed → reconstructed tokens
            # Gradients computed here (decoder is trainable)
            x_hat_g = model.decode_tokens(z_g, grid)

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

                z_r = model.compress_tokens(x_r)
                x_hat_r = model.decode_tokens(z_r, reg_grid)

                # Calculate reconstruction loss for this region
                loss_r, _, cos_r, top1_r = recon_loss(x_hat_r, x_r, ln)

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

            # ===== LOGGING =====
            # Print progress every N steps during training
            if train and (step + 1) % train_args.log_every == 0:
                print(
                    f"[train] step={step+1} loss={loss_total.item():.4f} "
                    f"cos={cos_g:.4f} reg_cos={agg['reg_cos'] / max(1, agg['reg_count']):.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss_step": float(loss_total.item()),
                            "train/cos_step": float(cos_g),
                            "train/reg_cos_running": float(
                                agg["reg_cos"] / max(1, agg["reg_count"])
                            ),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=global_step,
                    )

        # ===== FINAL GRADIENT UPDATE =====
        # If there are leftover accumulated gradients at epoch end, update now
        if train and step_in_accum > 0:
            optimizer.step()

        # ===== COMPUTE EPOCH AVERAGES =====
        # Divide accumulated metrics by counts to get averages
        denom = max(1, agg["count"])         # Prevent division by zero
        reg_denom = max(1, agg["reg_count"])

        return {
            "loss": agg["loss"] / denom,
            "mse": agg["mse"] / denom,
            "cos": agg["cos"] / denom,
            "top1": agg["top1"] / denom,
            "reg_cos": agg["reg_cos"] / reg_denom,
            "reg_top1": agg["reg_top1"] / reg_denom,
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

    # ===== 15.5. CHECKPOINT MANAGER (TOP-K BEST) =====
    best_checkpoints: List[Tuple[float, int, str]] = []
    save_top_k = max(0, int(train_args.save_top_k))
    monitor_metric = train_args.monitor_metric
    monitor_mode = train_args.monitor_mode.lower().strip()
    if monitor_mode not in {"max", "min"}:
        raise ValueError(f"Invalid monitor_mode: {train_args.monitor_mode}. Use 'max' or 'min'.")

    ckpt_dir = os.path.join(train_args.output_dir, train_args.checkpoint_subdir)
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

    # ===== 16. MAIN TRAINING LOOP =====
    # Iterate over epochs (full passes through training data)
    for epoch in range(1, train_args.num_train_epochs + 1):
        # Run one epoch of training
        train_metrics = run_epoch(train_loader, train=True)

        # Run one epoch of validation (no gradient updates)
        val_metrics = run_epoch(val_loader, train=False)

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

        # Save best checkpoints (top-k by chosen validation metric)
        maybe_save_topk(epoch, train_metrics, val_metrics)

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

    if wandb_run is not None:
        wandb_run.finish()


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
