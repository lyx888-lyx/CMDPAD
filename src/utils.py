"""Small helpers used by CMDPAD training and dialogue feature extraction.

This file reconstructs helpers missing from the released upstream repository.
It intentionally does not modify the author's model architecture.
"""

import random
from pathlib import Path

import numpy as np
import torch

from model import MultimodalAttentionModel


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Avoid enforcing deterministic kernels: some operations/platforms do not support them.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_dims_from_feature(df, modalities, prefix=""):
    """Return feature sizes under base modality names, e.g. 'bert-base-chinese'."""
    if df.empty:
        raise ValueError("Cannot infer dimensions from an empty feature table")
    dims = {}
    for modality in modalities:
        column = f"{prefix}{modality}"
        if column not in df.columns:
            raise KeyError(f"Missing feature column: {column}")
        value = next((v for v in df[column] if v is not None), None)
        if value is None:
            raise ValueError(f"All values are missing in {column}")
        arr = np.asarray(value)
        if arr.size == 0:
            raise ValueError(f"Empty feature vector in {column}")
        dims[modality] = int(arr.size)
    return dims


def _load_state_dict(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions without weights_only support
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return checkpoint


def build_model_and_load(model_path, modality_dims, args):
    """Restore a trained single-turn MAT for extracting frozen CLS vectors.

    The official single-turn trainer saves a raw state_dict. Output dimension
    (1 for affect, 5 for personality) is inferred from the final head weight.
    Strict key/shape checking prevents silent modality mix-ups.
    """
    state = _load_state_dict(model_path)
    checkpoint_modalities = [
        key.split(".")[1]
        for key in state
        if key.startswith("projectors.") and key.endswith(".net.0.weight")
    ]
    if checkpoint_modalities != list(modality_dims):
        raise ValueError(
            "Modality/order mismatch for checkpoint " + str(model_path)
            + f"; expected {checkpoint_modalities}, got {list(modality_dims)}"
        )
    head_indices = [
        int(key.split(".")[2])
        for key in state
        if key.startswith("head.net.") and key.endswith(".weight")
    ]
    if not head_indices:
        raise ValueError(f"Cannot find MAT regression head in {model_path}")
    output_dim = int(state[f"head.net.{max(head_indices)}.weight"].shape[0])
    model = MultimodalAttentionModel(
        modality_dims=modality_dims,
        proj_dim=args.proj_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        head_hidden=tuple(args.hidden),
        output_dim=output_dim,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not restore {model_path}. Check original single-turn "
            "model hyperparameters and modality dimensions.\n" + str(exc)
        ) from exc
    model.eval()
    return model
