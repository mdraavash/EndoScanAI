import uuid
from pathlib import Path
from PIL import Image
import numpy as np


def save_mask_preview(mask_data: np.ndarray, output_dir: Path):
    if mask_data.ndim < 3:
        raise ValueError('Expected 3D mask data')

    z = mask_data.shape[2] // 2
    slice_2d = mask_data[:, :, z]
    img_arr = (slice_2d > 0).astype(np.uint8) * 255
    preview = Image.fromarray(img_arr)
    preview = preview.convert('L')

    preview_name = f'segmentation_preview_{uuid.uuid4().hex}.png'
    preview_path = output_dir / preview_name
    preview.save(str(preview_path), format='PNG')
    return preview_path
