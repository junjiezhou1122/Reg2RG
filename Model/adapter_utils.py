"""
Adapter Weight Loading Utilities

Utilities for extracting and loading weights from pretrained adapters.
Particularly useful for Experiment 5a: extracting first layer from 6-layer Perceiver.

Author: Junjie Zhou
Date: 2025-12-25
"""

import torch
import torch.nn as nn


def load_first_layer_from_6layer_checkpoint(adapter_1layer, checkpoint_path):
    """
    Extract first layer weights from 6-layer Perceiver checkpoint and load into 1-layer adapter.

    This function handles the weight mapping between:
    - Source: 6-layer PerceiverResampler with layers.0.0.xxx, layers.0.1.xxx structure
    - Target: 1-layer OneLayerAdapter with direct attributes (no layers list)

    Weight mapping strategy:
        1. Latents: Direct copy (shared across all layers)
        2. Layer 0 attention: layers.0.0.xxx → xxx
        3. Layer 0 FFN: layers.0.1.xxx → ff.xxx
        4. Final norm: Direct copy

    Args:
        adapter_1layer: OneLayerAdapter instance (target model)
        checkpoint_path: Path to 6-layer Perceiver checkpoint (.pth file)

    Returns:
        adapter_1layer with loaded weights from first layer

    Raises:
        KeyError: If checkpoint doesn't contain 'perceiver' key
        RuntimeError: If weight shapes don't match

    Example:
        >>> from Model.one_layer_adapter import OneLayerAdapter
        >>> adapter = OneLayerAdapter(dim=768, num_latents=32, heads=8)
        >>> load_first_layer_from_6layer_checkpoint(
        ...     adapter,
        ...     "/path/to/RadFM_perceiver_fc.pth"
        ... )
        [INFO] Loading 1st layer from 6-layer checkpoint: /path/to/RadFM_perceiver_fc.pth
        ✓ Copied latents: torch.Size([32, 768])
        ✓ layers.0.0.norm_media.weight → norm_media.weight
        ...
        [INFO] Loaded 15 parameters
    """

    print(f"[INFO] Loading 1st layer from 6-layer checkpoint: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Validate checkpoint structure
    if "perceiver" not in ckpt:
        available_keys = list(ckpt.keys())
        raise KeyError(
            f"Checkpoint missing 'perceiver' key. Available keys: {available_keys}. "
            f"Expected structure: {{'perceiver': state_dict, 'fc': state_dict}}"
        )

    old_state = ckpt["perceiver"]
    new_state = {}

    # ===== 1. COPY LATENTS (SHARED) =====
    if "latents" in old_state:
        new_state["latents"] = old_state["latents"]
        print(f"   ✓ Copied latents: {old_state['latents'].shape}")
    else:
        print("   ⚠️  Warning: No latents found in checkpoint")

    # ===== 2. EXTRACT LAYER 0 ATTENTION WEIGHTS =====
    # Old structure: layers.0.0.xxx (layers[0][0] = first PerceiverAttention)
    # New structure: xxx (direct attributes in OneLayerAdapter)

    attention_mapping = {
        "layers.0.0.norm_media": "norm_media",
        "layers.0.0.norm_latents": "norm_latents",
        "layers.0.0.to_q": "to_q",
        "layers.0.0.to_kv": "to_kv",
        "layers.0.0.to_out": "to_out",
    }

    for old_prefix, new_prefix in attention_mapping.items():
        # Find all keys starting with this prefix (e.g., "layers.0.0.norm_media.weight")
        matched_keys = [k for k in old_state.keys() if k.startswith(old_prefix)]

        if not matched_keys:
            print(f"   ⚠️  Warning: No weights found for {old_prefix}")
            continue

        for old_key in matched_keys:
            # Replace prefix: layers.0.0.norm_media.weight → norm_media.weight
            new_key = old_key.replace(old_prefix, new_prefix)
            new_state[new_key] = old_state[old_key]
            print(f"   ✓ {old_key} → {new_key}")

    # ===== 3. EXTRACT LAYER 0 FFN WEIGHTS =====
    # Old structure: layers.0.1.0.xxx, layers.0.1.1.xxx, ... (layers[0][1] = FeedForward)
    # New structure: ff.0.xxx, ff.1.xxx, ... (Sequential module)

    ffn_prefix = "layers.0.1"
    ffn_keys = [k for k in old_state.keys() if k.startswith(ffn_prefix)]

    if not ffn_keys:
        print(f"   ⚠️  Warning: No FFN weights found for {ffn_prefix}")
    else:
        for old_key in ffn_keys:
            # Replace prefix: layers.0.1.X.yyy → ff.X.yyy
            new_key = old_key.replace(ffn_prefix, "ff")
            new_state[new_key] = old_state[old_key]
            print(f"   ✓ {old_key} → {new_key}")

    # ===== 4. COPY FINAL NORM (SHARED) =====
    if "norm.weight" in old_state:
        new_state["norm.weight"] = old_state["norm.weight"]
        new_state["norm.bias"] = old_state["norm.bias"]
        print(f"   ✓ Copied final norm")
    else:
        print("   ⚠️  Warning: No final norm found in checkpoint")

    # ===== 5. LOAD INTO ADAPTER =====
    missing_keys, unexpected_keys = adapter_1layer.load_state_dict(new_state, strict=False)

    print(f"[INFO] Loaded {len(new_state)} parameters")

    # Report missing keys (parameters in target model not in checkpoint)
    if missing_keys:
        # Truncate long lists for readability
        if len(missing_keys) > 3:
            print(f"   ⚠️  Missing keys ({len(missing_keys)} total): {missing_keys[:3]} ...")
        else:
            print(f"   ⚠️  Missing keys: {missing_keys}")

    # Report unexpected keys (parameters in checkpoint not in target model)
    if unexpected_keys:
        if len(unexpected_keys) > 3:
            print(f"   ⚠️  Unexpected keys ({len(unexpected_keys)} total): {unexpected_keys[:3]} ...")
        else:
            print(f"   ⚠️  Unexpected keys: {unexpected_keys}")

    # Validate critical components were loaded
    critical_params = ["latents", "to_q.weight", "to_kv.weight", "ff.1.weight"]
    loaded_critical = [p for p in critical_params if any(p in k for k in new_state.keys())]

    if len(loaded_critical) < len(critical_params):
        missing_critical = set(critical_params) - set(loaded_critical)
        print(f"   ⚠️  Warning: Some critical parameters may not have loaded: {missing_critical}")
    else:
        print(f"   ✅ All critical parameters loaded successfully")

    return adapter_1layer


def print_adapter_summary(adapter, name="Adapter"):
    """
    Print summary statistics of adapter parameters.

    Useful for debugging and comparing different adapter configurations.

    Args:
        adapter: nn.Module (OneLayerAdapter or PerceiverResampler)
        name: Display name for the adapter

    Example:
        >>> adapter = OneLayerAdapter(dim=768, num_latents=32)
        >>> print_adapter_summary(adapter, "1-Layer Adapter")
        ========== 1-Layer Adapter ==========
        Total parameters: 1,234,567
        Trainable parameters: 1,234,567
        Parameter breakdown:
          latents: 24,576 (2.0%)
          attention: 1,000,000 (81.0%)
          ffn: 210,000 (17.0%)
    """

    print(f"\n{'=' * 10} {name} {'=' * 10}")

    # Count total parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Parameter breakdown by component
    param_groups = {
        "latents": 0,
        "attention": 0,
        "ffn": 0,
        "norm": 0,
        "other": 0,
    }

    for name, param in adapter.named_parameters():
        if "latents" in name:
            param_groups["latents"] += param.numel()
        elif any(k in name for k in ["to_q", "to_kv", "to_out", "norm_media", "norm_latents"]):
            param_groups["attention"] += param.numel()
        elif "ff" in name:
            param_groups["ffn"] += param.numel()
        elif "norm" in name:
            param_groups["norm"] += param.numel()
        else:
            param_groups["other"] += param.numel()

    print("\nParameter breakdown:")
    for group, count in param_groups.items():
        if count > 0:
            percentage = (count / total_params) * 100
            print(f"  {group}: {count:,} ({percentage:.1f}%)")

    print("=" * (20 + len(name)))


def compare_adapter_outputs(adapter1, adapter2, sample_input, name1="Adapter 1", name2="Adapter 2"):
    """
    Compare outputs of two adapters on the same input.

    Useful for validating that extracted weights produce similar outputs.

    Args:
        adapter1: First adapter (e.g., 6-layer Perceiver)
        adapter2: Second adapter (e.g., extracted 1-layer)
        sample_input: Input tensor (B, T, F, v, D)
        name1: Display name for adapter1
        name2: Display name for adapter2

    Returns:
        dict with comparison metrics

    Example:
        >>> perceiver_6layer = PerceiverResampler(dim=768, num_latents=32)
        >>> adapter_1layer = OneLayerAdapter(dim=768, num_latents=32)
        >>> load_first_layer_from_6layer_checkpoint(adapter_1layer, "checkpoint.pth")
        >>>
        >>> sample = torch.randn(1, 1, 1, 1024, 768)
        >>> metrics = compare_adapter_outputs(perceiver_6layer, adapter_1layer, sample)
        >>> print(f"Output similarity: {metrics['cosine_similarity']:.4f}")
    """

    with torch.no_grad():
        out1 = adapter1(sample_input)
        out2 = adapter2(sample_input)

    # Compute comparison metrics
    mse = torch.nn.functional.mse_loss(out1, out2).item()

    # Cosine similarity
    out1_flat = out1.reshape(-1)
    out2_flat = out2.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        out1_flat.unsqueeze(0),
        out2_flat.unsqueeze(0),
        dim=1
    ).item()

    # L2 distance
    l2_dist = torch.norm(out1 - out2).item()

    # Relative error
    rel_error = (l2_dist / torch.norm(out1).item()) * 100

    print(f"\n{'=' * 10} Output Comparison {'=' * 10}")
    print(f"{name1} shape: {out1.shape}")
    print(f"{name2} shape: {out2.shape}")
    print(f"MSE: {mse:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"L2 distance: {l2_dist:.6f}")
    print(f"Relative error: {rel_error:.2f}%")
    print("=" * 34)

    return {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "l2_distance": l2_dist,
        "relative_error": rel_error,
    }
