import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import transformers

from Dataset.radgenome_dataset_train import RadGenomeDataset_Train
from Model.helpers import PerceiverResampler
from Model.position_encoding import PositionEmbeddingLearned3d
from Model.vit_3d import ViT

REGIONS = [
    "abdomen",
    "bone",
    "breast",
    "esophagus",
    "heart",
    "lung",
    "mediastinum",
    "pleura",
    "thyroid",
    "trachea and bronchie",
]


@dataclass
class ModelArguments:
    tokenizer_path: str = field(
        default="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf",
        metadata={"help": "Path to the tokenizer (for dataset preprocessing)."},
    )
    pretrained_visual_encoder: str = field(
        default="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth",
        metadata={"help": "Path to pretrained ViT3D weights."},
    )
    pretrained_adapter: str = field(
        default="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth",
        metadata={"help": "Path to pretrained Perceiver adapter weights."},
    )
    use_pretrained_adapter: bool = field(
        default=True,
        metadata={"help": "Load pretrained Perceiver weights if available."},
    )
    decode_mode: str = field(
        default="pre_proj",
        metadata={"help": "Decode from 'pre_proj' (Z) or 'post_proj' (fc(Z))."},
    )
    perceiver_num: int = field(default=32, metadata={"help": "Latent token count."})
    vis_dim: int = field(default=768, metadata={"help": "ViT token dimension."})
    llm_dim: int = field(default=4096, metadata={"help": "Projection output dimension."})
    decoder_layers: int = field(default=2, metadata={"help": "Decoder depth."})
    decoder_heads: int = field(default=8, metadata={"help": "Decoder heads."})
    decoder_ff_mult: int = field(default=4, metadata={"help": "Decoder FFN multiplier."})


@dataclass
class DataArguments:
    data_folder: str = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed"
    )
    mask_folder: str = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask"
    )
    report_file: str = field(
        default="/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv"
    )
    monai_cache_dir: str = field(default="/mnt2/ct/RadGenome-ChestCT/cache")
    val_split: float = field(
        default=0.05,
        metadata={"help": "Fraction of data used for validation."},
    )


@dataclass
class TrainArguments:
    output_dir: str = field(
        default="/mnt/home/zhoujunjie/Reg2RG/results/LIT",
        metadata={"help": "Where to write metrics and checkpoints."},
    )
    num_train_epochs: int = field(default=10)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.0)
    batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    log_every: int = field(default=10)
    seed: int = field(default=42)
    lambda_region: float = field(default=1.0)


class LITDataCollator:
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        vision_xs, mask_xs, region2areas = tuple(
            [instance[key] for instance in instances]
            for key in ("vision_x", "mask_x", "region2area")
        )
        vision_temp = {area: [] for area in REGIONS}
        mask_temp = {area: [] for area in REGIONS}

        vision_shape = next(iter(vision_xs[0].values())).shape
        mask_shape = next(iter(mask_xs[0].values())).shape

        useless_regions = []
        for area in REGIONS:
            flag = False
            for i in range(len(vision_xs)):
                if area in vision_xs[i]:
                    vision_temp[area].append(vision_xs[i][area])
                    mask_temp[area].append(mask_xs[i][area])
                    flag = True
                else:
                    vision_temp[area].append(torch.zeros(vision_shape))
                    mask_temp[area].append(torch.zeros(mask_shape))
            if not flag:
                useless_regions.append(area)

        images = torch.cat([vision["image"].unsqueeze(0) for vision in vision_xs], dim=0)
        for area in useless_regions:
            vision_temp.pop(area)
            mask_temp.pop(area)

        useful_regions = list(vision_temp.keys())
        vision_xs = {
            area: torch.cat([_.unsqueeze(0) for _ in vision_temp[area]], dim=0)
            for area in useful_regions
        }
        vision_xs["image"] = images
        mask_xs = {
            area: torch.cat([_.unsqueeze(0) for _ in mask_temp[area]], dim=0)
            for area in useful_regions
        }
        return dict(vision_x=vision_xs, mask_x=mask_xs, region2area=region2areas)


class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, x, x, need_weights=False)[0])
        x = self.norm2(x + self.cross_attn(x, memory, memory, need_weights=False)[0])
        x = self.norm3(x + self.ff(x))
        return x


class ProbeDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int,
        ff_mult: int,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.pos_embed = PositionEmbeddingLearned3d(
            dim // 3, h_patch_num=16, w_patch_num=16, d_patch_num=128
        )
        self.layers = nn.ModuleList(
            [CrossAttentionDecoderLayer(dim, heads, ff_mult) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, memory: torch.Tensor, grid: Tuple[int, int, int], batch_size: int
    ) -> torch.Tensor:
        h, w, d = grid
        n_tokens = h * w * d
        if n_tokens != self.num_tokens:
            raise ValueError(
                f"Query tokens mismatch: expected {self.num_tokens}, got {n_tokens}"
            )
        pos = self.pos_embed(batch_size, h, w, d, memory)
        q = self.query[:, :n_tokens, :].expand(batch_size, -1, -1) + pos
        for layer in self.layers:
            q = layer(q, memory)
        return self.norm(q)


class LITProbeModel(nn.Module):
    def __init__(
        self,
        vis_dim: int,
        llm_dim: int,
        perceiver_num: int,
        decoder_layers: int,
        decoder_heads: int,
        decoder_ff_mult: int,
        decode_mode: str,
    ):
        super().__init__()
        self.patch_size = 32
        self.frame_patch_size = 4
        self.vis_dim = vis_dim
        self.llm_dim = llm_dim
        self.decode_mode = decode_mode

        self.vision_encoder = ViT(
            image_size=512,
            frames=512,
            image_patch_size=self.patch_size,
            frame_patch_size=self.frame_patch_size,
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.adapter = PerceiverResampler(dim=vis_dim, num_latents=perceiver_num)
        self.fc = nn.Linear(vis_dim, llm_dim)
        self.mem_proj = (
            nn.Identity() if decode_mode == "pre_proj" else nn.Linear(llm_dim, vis_dim)
        )

        grid = self._patch_grid(256, 256, 64)
        num_tokens = grid[0] * grid[1] * grid[2]
        self.decoder = ProbeDecoder(
            num_tokens=num_tokens,
            dim=vis_dim,
            depth=decoder_layers,
            heads=decoder_heads,
            ff_mult=decoder_ff_mult,
        )

    @staticmethod
    def _patch_grid(h: int, w: int, d: int) -> Tuple[int, int, int]:
        return h // 32, w // 32, d // 4

    def encode_tokens(
        self, volume: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        b, s, c, h, w, d = volume.shape
        volume = volume.reshape(b * s, c, h, w, d)
        tokens, _ = self.vision_encoder(volume)
        grid = self._patch_grid(h, w, d)
        return tokens, grid

    def compress_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.unsqueeze(1).unsqueeze(1)
        z = self.adapter(x).squeeze(1)
        if self.decode_mode == "post_proj":
            z = self.fc(z)
        return z

    def decode_tokens(
        self, memory: torch.Tensor, grid: Tuple[int, int, int]
    ) -> torch.Tensor:
        memory = self.mem_proj(memory)
        return self.decoder(memory, grid, memory.shape[0])


def l2_error_topk(x_hat: torch.Tensor, x: torch.Tensor, ratio: float = 0.01) -> float:
    err = torch.norm(x_hat - x, dim=-1).reshape(-1)
    k = max(1, int(err.numel() * ratio))
    return err.topk(k).values.mean().item()


def recon_loss(
    x_hat: torch.Tensor, x: torch.Tensor, ln: nn.LayerNorm
) -> Tuple[torch.Tensor, float, float, float]:
    x_ln = ln(x)
    x_hat_ln = ln(x_hat)
    mse = F.mse_loss(x_hat_ln, x_ln)
    x_norm = F.normalize(x, dim=-1)
    x_hat_norm = F.normalize(x_hat, dim=-1)
    cos = (x_norm * x_hat_norm).sum(dim=-1).mean()
    loss = mse + (1 - cos)
    top1 = l2_error_topk(x_hat, x, ratio=0.01)
    return loss, mse.item(), cos.item(), top1


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_args.output_dir, exist_ok=True)

    dataset = RadGenomeDataset_Train(
        text_tokenizer=model_args.tokenizer_path,
        data_folder=data_args.data_folder,
        mask_folder=data_args.mask_folder,
        csv_file=data_args.report_file,
        cache_dir=data_args.monai_cache_dir,
    )
    val_size = int(len(dataset) * data_args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    collator = LITDataCollator()
    train_loader = DataLoader(
        train_set,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.dataloader_num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.dataloader_num_workers,
        collate_fn=collator,
    )

    model = LITProbeModel(
        vis_dim=model_args.vis_dim,
        llm_dim=model_args.llm_dim,
        perceiver_num=model_args.perceiver_num,
        decoder_layers=model_args.decoder_layers,
        decoder_heads=model_args.decoder_heads,
        decoder_ff_mult=model_args.decoder_ff_mult,
        decode_mode=model_args.decode_mode,
    )
    if model_args.pretrained_visual_encoder:
        ckpt = torch.load(model_args.pretrained_visual_encoder, map_location="cpu")
        model.vision_encoder.load_state_dict(ckpt, strict=True)

    if model_args.pretrained_adapter and model_args.use_pretrained_adapter:
        adapter_ckpt = torch.load(model_args.pretrained_adapter, map_location="cpu")
        if "perceiver" in adapter_ckpt:
            model.adapter.load_state_dict(adapter_ckpt["perceiver"])
        if model_args.decode_mode == "post_proj" and "fc" in adapter_ckpt:
            model.fc.load_state_dict(adapter_ckpt["fc"])

    model.to(device)

    set_requires_grad(model.vision_encoder, False)
    set_requires_grad(model.adapter, False)
    set_requires_grad(model.fc, False)
    set_requires_grad(model.decoder, True)

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
    )
    ln = nn.LayerNorm(model.vis_dim).to(device)

    def run_epoch(loader: DataLoader, train: bool) -> Dict[str, float]:
        if train:
            model.train()
        else:
            model.eval()
        agg = {
            "loss": 0.0,
            "mse": 0.0,
            "cos": 0.0,
            "top1": 0.0,
            "reg_cos": 0.0,
            "reg_top1": 0.0,
            "count": 0,
            "reg_count": 0,
        }
        for step, batch in enumerate(loader):
            vision_x = {k: v.to(device) for k, v in batch["vision_x"].items()}
            loss_total = 0.0
            if train:
                optimizer.zero_grad()

            with torch.no_grad():
                x_g, grid = model.encode_tokens(vision_x["image"])
            z_g = model.compress_tokens(x_g)
            x_hat_g = model.decode_tokens(z_g, grid)
            loss_g, mse_g, cos_g, top1_g = recon_loss(x_hat_g, x_g, ln)

            loss_total = loss_g
            agg["loss"] += loss_g.item()
            agg["mse"] += mse_g
            agg["cos"] += cos_g
            agg["top1"] += top1_g
            agg["count"] += 1

            reg_losses = []
            reg_cos = []
            reg_top1 = []
            for key, vol in vision_x.items():
                if key == "image":
                    continue
                present = vol.abs().sum(dim=(1, 2, 3, 4, 5)) > 0
                if not present.any():
                    continue
                with torch.no_grad():
                    x_r, _ = model.encode_tokens(vol[present])
                z_r = model.compress_tokens(x_r)
                x_hat_r = model.decode_tokens(z_r, grid)
                loss_r, _, cos_r, top1_r = recon_loss(x_hat_r, x_r, ln)
                reg_losses.append(loss_r)
                reg_cos.append(cos_r)
                reg_top1.append(top1_r)

            if reg_losses:
                loss_r = torch.stack(reg_losses).mean()
                loss_total = loss_total + train_args.lambda_region * loss_r
                agg["reg_cos"] += sum(reg_cos) / len(reg_cos)
                agg["reg_top1"] += sum(reg_top1) / len(reg_top1)
                agg["reg_count"] += 1

            if train:
                loss_total.backward()
                optimizer.step()

            if train and (step + 1) % train_args.log_every == 0:
                print(
                    f"[train] step={step+1} loss={loss_total.item():.4f} "
                    f"cos={cos_g:.4f} reg_cos={agg['reg_cos'] / max(1, agg['reg_count']):.4f}"
                )

        denom = max(1, agg["count"])
        reg_denom = max(1, agg["reg_count"])
        return {
            "loss": agg["loss"] / denom,
            "mse": agg["mse"] / denom,
            "cos": agg["cos"] / denom,
            "top1": agg["top1"] / denom,
            "reg_cos": agg["reg_cos"] / reg_denom,
            "reg_top1": agg["reg_top1"] / reg_denom,
        }

    metrics_path = os.path.join(train_args.output_dir, "lit_metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,split,loss,mse,cos,top1,reg_cos,reg_top1,decode_mode\n"
        )

    for epoch in range(1, train_args.num_train_epochs + 1):
        train_metrics = run_epoch(train_loader, train=True)
        val_metrics = run_epoch(val_loader, train=False)
        print(
            f"[epoch {epoch}] train_cos={train_metrics['cos']:.4f} "
            f"train_reg_cos={train_metrics['reg_cos']:.4f} "
            f"val_cos={val_metrics['cos']:.4f} "
            f"val_reg_cos={val_metrics['reg_cos']:.4f}"
        )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},train,{train_metrics['loss']:.6f},"
                f"{train_metrics['mse']:.6f},{train_metrics['cos']:.6f},"
                f"{train_metrics['top1']:.6f},{train_metrics['reg_cos']:.6f},"
                f"{train_metrics['reg_top1']:.6f},{model_args.decode_mode}\n"
            )
            f.write(
                f"{epoch},val,{val_metrics['loss']:.6f},"
                f"{val_metrics['mse']:.6f},{val_metrics['cos']:.6f},"
                f"{val_metrics['top1']:.6f},{val_metrics['reg_cos']:.6f},"
                f"{val_metrics['reg_top1']:.6f},{model_args.decode_mode}\n"
            )


if __name__ == "__main__":
    main()
