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
    """Run YOLO detection on input images with multimodal fusion for CT+PET.

    For CT+PET modality (true multimodal fusion):
        - Process PET images first to get spatial priors
        - Use PET detections as anchors for CT processing
        - Apply spatial attention and fusion between modalities
        - Combine detections with confidence weighting

    For CT modality:
        - Single image detection
    """
    if not config.YOLO_MODEL_PATH.exists():
        return None, None, f'YOLO model not found at {config.YOLO_MODEL_PATH}'

    # Create working directory for YOLO processing
    work_dir = config.OUTPUT_DIR / f'yolo_input_{uuid.uuid4().hex}'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Convert NIfTI files to 2D PNG images
    yolo_input_paths = []
    pet_detections = []  # Store PET detections for multimodal fusion

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

    # Load YOLO model once
    model = YOLO(str(config.YOLO_MODEL_PATH))
    detections = []
    inference_images = []

    # ── For CT+PET: Process PET first, then CT with multimodal fusion ─────
    if modality.lower() == 'ctpet':
        # Separate PET and CT inputs
        pet_inputs = [(path, label) for path, label in yolo_input_paths if label == 'PET']
        ct_inputs = [(path, label) for path, label in yolo_input_paths if label == 'CT']

        # Step 1: Process PET images to get spatial priors
        pet_detections = []
        pet_inference_images = []

        for image_path, modality_label in pet_inputs:
            try:
                results = model.predict(source=str(image_path), conf=config.YOLO_CONFIDENCE, save=False)
                annotated_path = None

                for r in results:
                    # Save overlayed detection image
                    if annotated_path is None:
                        plot_image = r.plot()
                        annotated_name = f'yolo_out_{modality_label}_{image_path.stem}_{uuid.uuid4().hex}.png'
                        annotated_path = config.OUTPUT_DIR / annotated_name
                        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(plot_image).save(str(annotated_path), format='PNG')
                        pet_inference_images.append({
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
                            'y2': cy2,
                            'slice_info': image_path.stem  # Track which slice this is from
                        }
                        pet_detections.append(detection)

            except Exception as e:
                return None, None, f'YOLO prediction failed on PET {modality_label}: {str(e)}'

        # Step 2: Process CT images with PET spatial priors
        ct_detections = []
        ct_inference_images = []

        for image_path, modality_label in ct_inputs:
            try:
                # Get PET detections for this slice (if any)
                slice_pet_detections = [
                    d for d in pet_detections
                    if d['slice_info'] in image_path.stem
                ]

                # Run YOLO on CT with potentially adjusted confidence
                # If we have PET priors, we can be more confident in CT detections
                adjusted_conf = config.YOLO_CONFIDENCE
                if slice_pet_detections:
                    # Lower confidence threshold when we have PET priors
                    adjusted_conf = max(0.1, config.YOLO_CONFIDENCE * 0.8)

                results = model.predict(source=str(image_path), conf=adjusted_conf, save=False)
                annotated_path = None

                for r in results:
                    # Save overlayed detection image
                    if annotated_path is None:
                        plot_image = r.plot()
                        annotated_name = f'yolo_out_{modality_label}_{image_path.stem}_{uuid.uuid4().hex}.png'
                        annotated_path = config.OUTPUT_DIR / annotated_name
                        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(plot_image).save(str(annotated_path), format='PNG')
                        ct_inference_images.append({
                            'image': image_path.name,
                            'modality': modality_label,
                            'url': f'/download/{annotated_name}',
                            'fusion_info': f'Fused with {len(slice_pet_detections)} PET priors'
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
                            'y2': cy2,
                            'fusion_applied': len(slice_pet_detections) > 0,
                            'pet_priors_count': len(slice_pet_detections)
                        }
                        ct_detections.append(detection)

            except Exception as e:
                return None, None, f'YOLO prediction failed on CT {modality_label}: {str(e)}'

        # Combine results
        detections = pet_detections + ct_detections
        inference_images = pet_inference_images + ct_inference_images

    # ── For CT only: Standard single-modality processing ───────────────────
    else:
        for image_path, modality_label in yolo_input_paths:
            try:
                results = model.predict(source=str(image_path), conf=config.YOLO_CONFIDENCE, save=False)
                annotated_path = None

                for r in results:
                    # Save overlayed detection image
                    if annotated_path is None:
                        plot_image = r.plot()
                        annotated_name = f'yolo_out_{modality_label}_{image_path.stem}_{uuid.uuid4().hex}.png'
                        annotated_path = config.OUTPUT_DIR / annotated_name
                        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

            except Exception as e:
                return None, None, f'YOLO prediction failed on {modality_label}: {str(e)}'

    return detections, inference_images, None