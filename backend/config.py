import os
import torch
from pathlib import Path

# BASE_DIR should be D:\aavash\6th sem\minor project\EndoScanAI
BASE_DIR    = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / 'backend'


# ── Dynamic model path finder ──────────────────────────────────────────────────
def find_model_path(base_path, folder_name):
    for root, dirs, _ in os.walk(base_path):
        if folder_name in dirs:
            return Path(root) / folder_name
    return None


# ── nnU-Net — Select based on modality ─────────────────────────────────────────
MODELS_ROOT = BACKEND_DIR / 'models' / 'trained_weights' / 'nnunet'

# CT-only model
CT_ONLY_RESULTS = MODELS_ROOT / 'For_CT_only' / 'nnunet_results'
CT_ONLY_MODEL_FALLBACK = CT_ONLY_RESULTS.exists()
if CT_ONLY_MODEL_FALLBACK:
    os.environ['nnUNet_raw_ct'] = str(MODELS_ROOT / 'For_CT_only' / 'nnunet_raw')
    os.environ['nnUNet_preprocessed_ct'] = str(MODELS_ROOT / 'For_CT_only' / 'nnunet_preprocessed')
    os.environ['nnUNet_results_ct'] = str(CT_ONLY_RESULTS)

# CT+PET model
CTPET_RESULTS = MODELS_ROOT / 'For_CT_PET' / 'nnunet_results_ct_pet'
CTPET_MODEL_EXISTS = CTPET_RESULTS.exists()
if CTPET_MODEL_EXISTS:
    os.environ['nnUNet_raw'] = str(MODELS_ROOT / 'For_CT_PET' / 'nnunet_raw')
    os.environ['nnUNet_preprocessed'] = str(MODELS_ROOT / 'For_CT_PET' / 'nnunet_preprocessed_ct_pet')
    os.environ['nnUNet_results'] = str(CTPET_RESULTS)

def get_nnunet_model_path(modality: str):
    """Get the appropriate nnUNet model path based on modality.
    
    Returns the path to the nnUNetTrainer folder (containing dataset.json and plans.json),
    not the results root folder.
    """
    if modality.lower() == 'ct':
        if not CT_ONLY_MODEL_FALLBACK:
            raise ValueError('CT-only model not found. Check For_CT_only folder.')
        # Find Dataset* folder
        dataset_folders = list(CT_ONLY_RESULTS.glob('Dataset*'))
        if not dataset_folders:
            raise ValueError(f'No Dataset folders found in {CT_ONLY_RESULTS}')
        dataset_path = dataset_folders[0]
        
        # Find nnUNetTrainer* folder inside dataset
        trainer_folders = list(dataset_path.glob('nnUNetTrainer*'))
        if not trainer_folders:
            raise ValueError(f'No nnUNetTrainer folders found in {dataset_path}')
        return trainer_folders[0]
        
    elif modality.lower() == 'ctpet':
        if not CTPET_MODEL_EXISTS:
            raise ValueError('CT+PET model not found. Check For_CT_PET folder.')
        # Find Dataset* folder
        dataset_folders = list(CTPET_RESULTS.glob('Dataset*'))
        if not dataset_folders:
            raise ValueError(f'No Dataset folders found in {CTPET_RESULTS}')
        dataset_path = dataset_folders[0]
        
        # Find nnUNetTrainer* folder inside dataset
        trainer_folders = list(dataset_path.glob('nnUNetTrainer*'))
        if not trainer_folders:
            raise ValueError(f'No nnUNetTrainer folders found in {dataset_path}')
        return trainer_folders[0]
        
    else:
        raise ValueError(f'Unknown modality: {modality}')

NNUNET_FOLD       = '0'
NNUNET_CHECKPOINT = 'checkpoint_best.pth'
NNUNET_PLAN       = 'nnUNetPlans'

# ── YOLO ───────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH  = BACKEND_DIR / 'models' / 'trained_weights' / 'yolo' / 'v10m_combined.pt'
YOLO_CONFIDENCE  = 0.25

# ── Classifier ─────────────────────────────────────────────────────────────────
CLASSIFIER_MODEL_PATH     = BACKEND_DIR / 'models' / 'trained_weights' / 'classifier' / 'demo_bundle.pkl'
CLASSIFIER_SELECTOR_PATH  = BACKEND_DIR / 'models' / 'trained_weights' / 'classifier' / 'save_model.py'
CLASSIFIER_THRESHOLD      = 0.50      # probability above which prediction = Malignant
CLASSIFIER_IMG_SIZE       = 224

# ── Shared ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = BACKEND_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('--- Config Loaded ---')
print(f'Device                : {DEVICE}')
print(f'CT-only model exists  : {CT_ONLY_MODEL_FALLBACK}')
print(f'CT+PET model exists   : {CTPET_MODEL_EXISTS}')
print(f'YOLO model path       : {YOLO_MODEL_PATH}  (exists: {YOLO_MODEL_PATH.exists()})')
print(f'Classifier path       : {CLASSIFIER_MODEL_PATH}  (exists: {CLASSIFIER_MODEL_PATH.exists()})')
print(f'Classifier Selector   : {CLASSIFIER_SELECTOR_PATH}  (exists: {CLASSIFIER_SELECTOR_PATH.exists()})')