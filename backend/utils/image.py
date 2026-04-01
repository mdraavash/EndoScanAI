import uuid
from pathlib import Path
from PIL import Image
import numpy as np


def save_mask_preview(mask_data: np.ndarray, output_dir: Path) -> Path:
    """Save a PNG of the centre axial slice of a binary mask."""
    if mask_data.ndim < 3:
        raise ValueError('Expected 3D mask data')
    z = mask_data.shape[2] // 2
    slice_2d = mask_data[:, :, z]
    img_arr = (slice_2d > 0).astype(np.uint8) * 255
    preview = Image.fromarray(img_arr).convert('L')
    preview_name = f'mask_preview_{uuid.uuid4().hex}.png'
    preview_path = output_dir / preview_name
    preview.save(str(preview_path), format='PNG')
    return preview_path


def save_raw_preview(nifti_path: Path, output_dir: Path) -> Path:
    """Save a normalised PNG of the centre axial slice of a NIfTI volume."""
    import nibabel as nib
    img   = nib.load(str(nifti_path))
    data  = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim < 3:
        raise ValueError('Expected at least 3D NIfTI data')
    z = data.shape[2] // 2
    slice_2d = data[:, :, z]

    # Robust percentile normalisation
    low, high = np.percentile(slice_2d, (1, 99))
    clipped = np.clip(slice_2d, low, high)
    if high > low:
        scaled = (clipped - low) / (high - low) * 255.0
    else:
        scaled = np.zeros_like(clipped)

    img_arr = np.round(scaled).astype(np.uint8)
    preview = Image.fromarray(img_arr).convert('RGB')
    preview_name = f'raw_preview_{uuid.uuid4().hex}.png'
    preview_path = output_dir / preview_name
    preview.save(str(preview_path), format='PNG')
    return preview_path
