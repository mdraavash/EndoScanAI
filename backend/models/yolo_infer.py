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
    """Run YOLO detection on input images.
    
    For CT+PET modality:
        - First image is treated as PET (scanned first)
        - Second image is treated as CT (uses PET bounding box as reference)
    
    For CT modality:
        - Single image detection
    """
    if not config.YOLO_MODEL_PATH.exists():
        return None, None, f'YOLO model not found at {config.YOLO_MODEL_PATH}'

    # Create working directory for YOLO processing
    work_dir = config.OUTPUT_DIR / f'yolo_input_{uuid.uuid4().hex}'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Convert NIfTI files to 2D PNG images
    # For CT+PET: first is PET, second is CT
    yolo_input_paths = []
    pet_detections = []  # Store PET detections for CT reference
    
    modality_labels = {
        'ct': ['CT'],
        'ctpet': ['PET', 'CT']
    }
    labels = modality_labels.get(modality.lower(), ['Image'])
    
    for idx, p in enumerate(input_paths):
        if not p.exists():
            continue

        label = labels[idx] if idx < len(labels) else f'Image{idx}'
        is_nifti = p.suffix.lower() in ['.nii', '.gz'] or p.name.lower().endswith('.nii.gz')
        if is_nifti:
            png_results, error = convert_nifti_to_2d_images(p, work_dir, modality_label=label)
            if error:
                return None, None, f'Failed to convert {p.name}: {error}'
            yolo_input_paths.extend(png_results)
        else:
            dst = work_dir / p.name
            shutil.copy2(p, dst)
            yolo_input_paths.append((dst, label))

    if not yolo_input_paths:
        return None, None, 'No valid input files for YOLO inference'

    # Run YOLO on each image
    model = YOLO(str(config.YOLO_MODEL_PATH))
    detections = []
    inference_images = []

    # ── Process images ─────────────────────────────────────────────────────
    for image_path, modality_label in yolo_input_paths:
        try:
            results = model.predict(source=str(image_path), conf=config.YOLO_CONFIDENCE, save=False)
            annotated_path = None

            for r in results:
                # Save overlayed detection image for first result of this input path
                if annotated_path is None:
                    plot_image = r.plot()  # returns ndarray with boxes drawn
                    annotated_name = f'yolo_out_{modality_label}_{image_path.stem}_{uuid.uuid4().hex}.png'
                    annotated_path = config.OUTPUT_DIR / annotated_name

                    # Ensure output directory exists
                    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    # Convert array to RGB image and save
                    Image.fromarray(plot_image).save(str(annotated_path), format='PNG')
                    inference_images.append({
                        'image': image_path.name,
                        'modality': modality_label,
                        'url': f'/download/{annotated_name}'
                    })

                for b in r.boxes:
                    cx1, cy1, cx2, cy2 = map(float, b.xyxy[0])
                    detection = {
                        'image': image_path.name,
                        'modality': modality_label,
                        'class': model.names[int(b.cls[0])],
                        'confidence': float(b.conf[0]),
                        'x1': cx1,
                        'y1': cy1,
                        'x2': cx2,
                        'y2': cy2
                    }
                    detections.append(detection)
                    
                    # Store PET detections for potential CT reference
                    if modality_label == 'PET':
                        pet_detections.append(detection)
                        
        except Exception as e:
            return None, None, f'YOLO prediction failed on {modality_label}: {str(e)}'

    # ── For CT+PET: Add reference note to CT detections ─────────────────────
    if modality.lower() == 'ctpet' and pet_detections:
        for detection in detections:
            if detection['modality'] == 'CT':
                # Mark that this CT detection is in context of PET bounding boxes
                detection['pet_reference'] = True

    return detections, inference_images, None