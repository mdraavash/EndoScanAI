import uuid
import shutil
import os
from pathlib import Path
import nibabel as nib
import torch

# Relative import from parent package
# Ensure your PYTHONPATH is set to 'backend' so this works
from utils.image import save_mask_preview
import config

def run_nnunet_inference(upload_dir: Path, input_paths, modality: str):
    # 1. Path Verification
    if not config.NNUNET_MODEL_PATH.exists():
        return None, None, f'nnUNet model path not found: {config.NNUNET_MODEL_PATH}'

    # 2. Prepare input folder (Must be a directory for nnU-Net)
    model_input = upload_dir / 'imagesTs'
    model_input.mkdir(parents=True, exist_ok=True)

    case_id = 'case_001'
    modality_map = {'ct': [0], 'ctpet': [0, 1]}
    modality_indices = modality_map.get(modality.lower(), [0])

    if len(input_paths) != len(modality_indices):
        return None, None, f'Expected {len(modality_indices)} file(s) for {modality}, got {len(input_paths)}'

    # 3. Rename files to the required suffix format: case_001_0000.nii.gz
    for idx, fp in enumerate(input_paths):
        fp = Path(fp) # Ensure it's a Path object
        if fp.exists():
            modality_idx = modality_indices[idx]
            new_filename = f'{case_id}_{modality_idx:04d}.nii.gz'
            dest = model_input / new_filename
            shutil.copy2(fp, dest)

    # 4. Prepare Output Directory
    infer_out = config.OUTPUT_DIR / f'nnunet_pred_{uuid.uuid4().hex}'
    infer_out.mkdir(parents=True, exist_ok=True)

    # 5. Initialize nnU-Net Predictor
    # We import here to ensure environment variables from config are loaded first
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=config.DEVICE,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # 6. Initialize from folder - FIXING THE ARGUMENT ERROR
    try:
        # Note: 'plans_identifier' is often not needed in v2.1+ 
        # because it is read from the 'dataset.json' in the model folder.
        # If it fails, try removing plans_identifier entirely.
        predictor.initialize_from_trained_model_folder(
            str(config.NNUNET_MODEL_PATH),
            use_folds=(str(config.NNUNET_FOLD),), # Ensure fold is a tuple of strings
            checkpoint_name=config.NNUNET_CHECKPOINT,
            # Removed plans_identifier to fix your specific error
        )
    except Exception as e:
        return None, None, f'Failed to initialize nnUNet predictor: {str(e)}'

    # 7. Run Inference
    try:
        predictor.predict_from_files(
            str(model_input),
            str(infer_out),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
    except Exception as e:
        return None, None, f'nnUNet prediction failed: {str(e)}'

    # 8. Post-processing
    outputs = sorted(infer_out.glob('*.nii.gz'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not outputs:
        return None, None, 'nnUNet output mask not found'

    mask_path = outputs[0]
    
    try:
        mask_nifti = nib.load(str(mask_path))
        mask_data = mask_nifti.get_fdata()

        preview_path = None
        if mask_data.ndim >= 3:
            # Pass the data to your preview utility
            preview_path = save_mask_preview(mask_data, config.OUTPUT_DIR)

        return mask_path, preview_path, None
    except Exception as e:
        return mask_path, None, f'Mask generated but preview failed: {str(e)}'