"""
Server-specific paths for Reg2RG project.

Usage:
    from config.paths import Paths

    # Access paths
    model_path = Paths.MODELS / "Llama-2-7b-chat-hf"
    data_path = Paths.DATA / "train_preprocessed"

To switch servers:
    1. Create a new paths file (e.g., paths_local.py)
    2. Change the import in __init__.py
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerPaths:
    """Immutable server path configuration."""

    # Base directories
    MODELS: Path
    DATA: Path
    OUTPUTS: Path
    CACHE: Path

    # Model weights
    @property
    def LLAMA_7B(self) -> Path:
        return self.MODELS / "Llama-2-7b-chat-hf"

    @property
    def VIT_3D(self) -> Path:
        return self.MODELS / "Reg2RG" / "RadFM_vit3d.pth"

    @property
    def PERCEIVER_FC(self) -> Path:
        return self.MODELS / "Reg2RG" / "RadFM_perceiver_fc.pth"

    # RadGenome dataset
    @property
    def RADGENOME(self) -> Path:
        return self.DATA

    @property
    def TRAIN_DATA(self) -> Path:
        return self.RADGENOME / "train_preprocessed"

    @property
    def TRAIN_MASK(self) -> Path:
        return self.RADGENOME / "train_region_mask"

    @property
    def TRAIN_REPORT(self) -> Path:
        return self.RADGENOME / "radgenome_files" / "train_region_report.csv"

    @property
    def VAL_DATA(self) -> Path:
        return self.RADGENOME / "valid_preprocessed"

    @property
    def VAL_MASK(self) -> Path:
        return self.RADGENOME / "valid_region_mask"

    @property
    def VAL_REPORT(self) -> Path:
        return self.RADGENOME / "radgenome_files" / "validation_region_report.csv"

    @property
    def MONAI_CACHE(self) -> Path:
        return self.CACHE / "cache_lit"


# ============================================================================
# SERVER CONFIGURATION: jhcpu7 (Zhou Junjie's server)
# ============================================================================
Paths = ServerPaths(
    MODELS=Path("/mnt/home/zhoujunjie/models"),
    DATA=Path("/mnt2/ct/RadGenome-ChestCT/dataset"),
    OUTPUTS=Path("/mnt/home/zhoujunjie/outputs"),
    CACHE=Path("/mnt2/ct/RadGenome-ChestCT"),
)


# ============================================================================
# Alternative: Local development paths (uncomment to use)
# ============================================================================
# Paths = ServerPaths(
#     MODELS=Path("/Users/junjie/models"),
#     DATA=Path("/Users/junjie/data/RadGenome"),
#     OUTPUTS=Path("/Users/junjie/outputs"),
#     CACHE=Path("/Users/junjie/cache"),
# )
