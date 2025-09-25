"""
Model utilities for transformer models used in disentangle-sycophancy project.
"""
import torch
import os
import re
from typing import Optional


def _get_decoder_layers(model):
    """Try to locate decoder layers for different model architectures."""
    # Try common architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # Gemma: sometimes under model.model.decoder.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    # Some Gemma builds put it at top-level "layers"
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not locate decoder layers for steering.")


def _hidden_to_block_index(hidden_idx: int, model) -> int:
    """
    Convert a hidden_states index (0..num_layers) to a decoder block index (0..num_layers-1).
    If input already looks like a block index (within range), return it unchanged.
    """
    if hidden_idx is None:
        raise ValueError(
            "Resolved steer layer is None. Provide --steer_layer or --layer, or run training to select a layer."
        )
    layers = _get_decoder_layers(model)
    num_blocks = len(layers)
    
    # Try to read from config, else fallback
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(model.config, "n_layer", None)
    if num_layers is None:
        num_layers = num_blocks

    # If it's clearly a hidden index (embeddings at 0, last at num_blocks), map to block
    if hidden_idx == 0:
        raise ValueError(
            "Cannot steer at embeddings (hidden index 0). Choose a layer >= 1."
        )
    if hidden_idx == num_layers:  # last hidden index maps to last block
        return num_blocks - 1
    # If within block range, assume already a block index
    if 0 <= hidden_idx < num_blocks:
        return hidden_idx
    # If within hidden range (1..num_blocks-1), map by -1
    if 1 <= hidden_idx <= num_blocks - 1:
        return hidden_idx - 1
    # Fallback clamp
    return max(0, min(num_blocks - 1, hidden_idx - 1))


def build_special_mask(tokenizer, device):
    """Build a mask for special tokens."""
    max_id = max(tokenizer.all_special_ids)
    mask_size = max(tokenizer.vocab_size, max_id + 1)
    special_mask = torch.zeros(mask_size, device=device, dtype=torch.bool)
    for sid in tokenizer.all_special_ids:
        special_mask[sid] = True
    return special_mask


def _resolve_layer_index(idx, num_layers):
    """Resolve layer index, handling negative indices."""
    if idx is None:
        return None
    return idx if idx >= 0 else (num_layers + 1 + idx)


def _default_w_path(output_dir: str, primary_label: str, hidden_layer: Optional[int]) -> Optional[str]:
    """Helper to resolve a default w path per hidden layer."""
    if hidden_layer is None:
        return None
    return os.path.join(
        output_dir, f"wDiffMean_raw_{primary_label}_L{hidden_layer}.npy"
    )


def _infer_label_from_w_path(path_or_name: str) -> Optional[str]:
    """Try to parse the behavior label from a file path like
    .../wDiffMean_raw_<label>_L30.npy. Returns None if not found."""
    base = os.path.basename(path_or_name)
    m = re.search(r"wDiffMean_raw_(.+?)_L\d+\.npy$", base)
    if m:
        return m.group(1)
    return None


def _select_dtype(name):
    """Select appropriate torch dtype."""
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    # auto
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32