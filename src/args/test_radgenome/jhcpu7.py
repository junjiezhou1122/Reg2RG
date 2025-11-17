import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Sequence

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf"
    )
    tokenizer_path: str = field(
        default="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf",
        metadata={"help": "Path to the tokenizer data."},
    )
    pretrained_visual_encoder: Optional[str] = field(
        default="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth"
    )
    pretrained_adapter: Optional[str] = field(
        default="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth"
    )
    ckpt_path: Optional[str] = field(   
        default="/mnt/home/zhoujunjie/Reg2RG/outputs/Reg2RG_radgenome/pytorch_model.bin"
    )

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed"
    )
    mask_folder: Optional[str] = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask"
    )
    report_file: Optional[str] = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv"
    )
    monai_cache_dir: Optional[str] = field(
        default="/mnt2/ct/RadGenome-ChestCT/cache"
    )
    result_path: Optional[str] = field(
        default="/mnt/home/zhoujunjie/Reg2RG/results/radgenome_combined_reports.csv"
    )
