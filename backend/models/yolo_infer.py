import os
import uuid
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def convert_nifti_to_2d_images(nifti_path: Path, output_dir: Path, modality_label: str = 'image'):
    """Convert 3D/4D NIfTI file to 2D PNG images for YOLO detection.
    
    Args:
        nifti_path: Path to NIfTI file
        output_dir: Directory to save PNG slices
        modality_label: Label for this modality (e.g., 'CT', 'PET')
    """
    try:
        img = nib.load(str(nifti_path))
        data = img.get_fdata()

        if data.ndim == 4:
            # Use first timepoint/channel if 4D and reshuffle to 3D
            data = data[..., 0]

        if data.ndim != 3:
            return None, f'Expected 3D image, got {data.ndim}D'

        z_slices = data.shape[2]
        if z_slices < 1:
            return None, 'No slices found in NIfTI volume'

        # pick a slice subset to limit overload; include middle plus spread
        max_slices = 20
        if z_slices <= max_slices:
            selected = list(range(z_slices))
        else:
            step = max(1, z_slices // max_slices)
            selected = list(range(0, z_slices, step))
            if selected[-1] != z_slices - 1:
                selected.append(z_slices - 1)

        output_paths = []
        for slice_idx in selected:
            slice_2d = data[:, :, slice_idx]

            # intensity normalization with robust clipping to reduce outliers
            low, high = np.percentile(slice_2d, (1, 99))
            slice_clipped = np.clip(slice_2d, low, high)

            if high > low:
                slice_scaled = ((slice_clipped - low) / (high - low) * 255.0)
            else:
                slice_scaled = np.zeros_like(slice_clipped)

            slice_uint8 = np.round(slice_scaled).astype(np.uint8)

            img_pil = Image.fromarray(slice_uint8)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            output_name = f'yolo_slice_{modality_label}_{nifti_path.stem}_s{slice_idx}_{uuid.uuid4().hex}.png'
            output_path = output_dir / output_name
            img_pil.save(str(output_path), format='PNG')
            output_paths.append((output_path, modality_label))

        if not output_paths:
            return None, 'No PNG images were created from NIfTI'

        return output_paths, None
    except Exception as e:
        return None, str(e)


def run_yolo_inference(input_paths, modality: str):
    """Run YOLO detection on a single input image.

    The YOLO model was trained on mixed CT+PET dataset but processes images independently.
    This function accepts a SINGLE file from either CT or PET modality.

    Args:
        input_paths: List with ONE file path (CT or PET image)
        modality: 'ct' or 'ctpet' (determines which modality the single file is)

    Returns:
        detections: List of detection results
        inference_images: List of annotated visualization images
        error: Error message if any
    """
    if not config.YOLO_MODEL_PATH.exists():
        return None, None, f'YOLO model not found at {config.YOLO_MODEL_PATH}'

    # Validate single input file
    if not input_paths or len(input_paths) == 0:
        return None, None, 'No input file provided for YOLO detection'
    
    if len(input_paths) > 1:
        return None, None, f'YOLO detection accepts only ONE file. Got {len(input_paths)} files.'

    input_file = input_paths[0]
    if not input_file.exists():
        return None, None, f'Input file not found: {input_file}'

    # Create working directory for YOLO processing
    work_dir = config.OUTPUT_DIR / f'yolo_input_{uuid.uuid4().hex}'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Convert NIfTI file to 2D PNG images
    # Determine modality label based on input
    if modality.lower() == 'ctpet':
        # For CT+PET, the single input could be either CT or PET
        # We'll process it as-is without assuming which one
        modality_label = 'CT+PET'
    else:
        modality_label = 'CT'

    is_nifti = input_file.suffix.lower() in ['.nii', '.gz'] or input_file.name.lower().endswith('.nii.gz')
    
    if is_nifti:
        yolo_input_paths, error = convert_nifti_to_2d_images(input_file, work_dir, modality_label=modality_label)
        if error:
            return None, None, f'Failed to convert {input_file.name}: {error}'
    else:
        dst = work_dir / input_file.name
        shutil.copy2(input_file, dst)
        yolo_input_paths = [(dst, modality_label)]

    if not yolo_input_paths:
        return None, None, 'No valid input images generated for YOLO detection'

    # Load YOLO model and run inference
    model = YOLO(str(config.YOLO_MODEL_PATH))
    detections = []
    inference_images = []

    try:
        for image_path, label in yolo_input_paths:
            results = model.predict(source=str(image_path), conf=config.YOLO_CONFIDENCE, save=False)
            annotated_path = None

            for r in results:
                # Save overlayed detection image
                if annotated_path is None:
                    plot_image = r.plot()
                    annotated_name = f'yolo_out_{label}_{image_path.stem}_{uuid.uuid4().hex}.png'
                    annotated_path = config.OUTPUT_DIR / annotated_name
                    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(plot_image).save(str(annotated_path), format='PNG')
                    inference_images.append({
                        'image': image_path.name,
                        'modality': label,
                        'url': f'/download/{annotated_name}'
                    })

                # Extract detections
                for b in r.boxes:
                    cx1, cy1, cx2, cy2 = map(float, b.xyxy[0])
                    detection = {
                        'image': image_path.name,
                        'modality': label,
                        'class': model.names[int(b.cls[0])],
                        'confidence': float(b.conf[0]),
                        'x1': cx1,
                        'y1': cy1,
                        'x2': cx2,
                        'y2': cy2
                    }
                    detections.append(detection)

    except Exception as e:
        return None, None, f'YOLO prediction failed: {str(e)}'

    return detections, inference_images, None