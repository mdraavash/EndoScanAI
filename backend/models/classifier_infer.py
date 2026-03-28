"""
classifier_infer.py
-------------------
Binary malignancy classifier for EndoScan AI.

Expects a NIfTI CT (or CT-PET) file.  Extracts the centre axial slice,
resizes to the model's input size, and runs a forward pass through a
pretrained ResNet-18 (or whatever checkpoint is configured).

Returns: (probability: float, label: str, confidence: float, error: str|None)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import nibabel as nib
import config


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_centre_slice(nifti_path: Path) -> np.ndarray:
    """Load the centre axial slice from a NIfTI file and return as float32 HxW."""
    img  = nib.load(str(nifti_path))
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim < 3:
        raise ValueError(f'Expected ≥3D volume, got {data.ndim}D')
    z = data.shape[2] // 2
    return data[:, :, z].astype(np.float32)


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Robust percentile normalisation → [0, 1]."""
    lo, hi = np.percentile(arr, (1, 99))
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-8)


def _preprocess(nifti_path: Path, img_size: int = 224) -> "torch.Tensor":
    """Load, normalise and convert a centre slice to a 1×3×H×W tensor."""
    import torch
    from PIL import Image

    arr   = _load_centre_slice(nifti_path)
    norm  = _normalise(arr)
    uint8 = (norm * 255).astype(np.uint8)

    pil   = Image.fromarray(uint8).convert('RGB').resize(
        (img_size, img_size), Image.BILINEAR
    )
    tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0

    # ImageNet-style normalisation (matches most pretrained backbones)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)   # 1×3×H×W


def _build_model():
    """Build or load the classifier model from config.CLASSIFIER_MODEL_PATH."""
    import torch
    import torch.nn as nn

    try:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)   # binary: benign / malignant
    except Exception as e:
        raise RuntimeError(f'Could not build ResNet-18: {e}')

    if config.CLASSIFIER_MODEL_PATH.exists():
        try:
            ckpt = torch.load(str(config.CLASSIFIER_MODEL_PATH),
                              map_location='cpu')
            # Support both raw state_dict and {'state_dict': …} checkpoints
            sd = ckpt.get('state_dict', ckpt)
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            raise RuntimeError(f'Failed to load classifier weights: {e}')
    else:
        raise FileNotFoundError(
            f'Classifier model not found: {config.CLASSIFIER_MODEL_PATH}'
        )

    return model


# ── Public API ─────────────────────────────────────────────────────────────────

def run_classifier_inference(
    input_paths: list,
    modality: str
) -> tuple[float | None, str | None, float | None, str | None]:
    """
    Run binary malignancy classification on the first input file.

    Returns
    -------
    probability : float  — P(malignant), 0–1
    label       : str    — 'Malignant' or 'Benign'
    confidence  : float  — max softmax score (0–1)
    error       : str | None
    """
    import torch
    import torch.nn.functional as F

    if not input_paths:
        return None, None, None, 'No input files provided'

    nifti_path = Path(input_paths[0])
    if not nifti_path.exists():
        return None, None, None, f'Input file not found: {nifti_path}'

    try:
        tensor = _preprocess(nifti_path)
    except Exception as e:
        return None, None, None, f'Preprocessing failed: {e}'

    try:
        model = _build_model()
    except Exception as e:
        return None, None, None, str(e)

    model.eval()
    device = config.DEVICE
    model.to(device)
    tensor = tensor.to(device)

    try:
        with torch.no_grad():
            logits = model(tensor)               # 1×2
            probs  = F.softmax(logits, dim=1)    # 1×2
            mal_prob   = float(probs[0, 1])      # probability of class 1 = malignant
            confidence = float(probs.max())
            label = 'Malignant' if mal_prob > config.CLASSIFIER_THRESHOLD else 'Benign'
    except Exception as e:
        return None, None, None, f'Inference failed: {e}'

    return mal_prob, label, confidence, None