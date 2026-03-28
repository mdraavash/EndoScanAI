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


# ── nnU-Net ────────────────────────────────────────────────────────────────────
RESULTS_ROOT = find_model_path(BACKEND_DIR, 'nnunet_results_ct_pet')

if RESULTS_ROOT:
    os.environ['nnUNet_raw']          = str(RESULTS_ROOT.parent / 'nnunet_raw')
    os.environ['nnUNet_preprocessed'] = str(RESULTS_ROOT.parent / 'nnunet_preprocessed_ct_pet')
    os.environ['nnUNet_results']       = str(RESULTS_ROOT)
    try:
        NNUNET_MODEL_PATH = next(RESULTS_ROOT.glob('**/nnUNetTrainer*2d'))
    except StopIteration:
        NNUNET_MODEL_PATH = RESULTS_ROOT
else:
    NNUNET_MODEL_PATH = Path('NOT_FOUND')

NNUNET_FOLD       = '0'
NNUNET_CHECKPOINT = 'checkpoint_best.pth'
NNUNET_PLAN       = 'nnUNetPlans'

# ── YOLO ───────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH  = BACKEND_DIR / 'models' / 'trained_weights' / 'yolo' / 'v10m_combined.pt'
YOLO_CONFIDENCE  = 0.25

# ── Classifier ─────────────────────────────────────────────────────────────────
CLASSIFIER_MODEL_PATH  = BACKEND_DIR / 'models' / 'trained_weights' / 'classifier' / 'resnet18_endometrial.pth'
CLASSIFIER_THRESHOLD   = 0.50     # probability above which prediction = Malignant
CLASSIFIER_IMG_SIZE    = 224

# ── Shared ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = BACKEND_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('--- Config Loaded ---')
print(f'Device             : {DEVICE}')
print(f'nnU-Net model path : {NNUNET_MODEL_PATH}  (exists: {NNUNET_MODEL_PATH.exists()})')
print(f'YOLO model path    : {YOLO_MODEL_PATH}  (exists: {YOLO_MODEL_PATH.exists()})')
print(f'Classifier path    : {CLASSIFIER_MODEL_PATH}  (exists: {CLASSIFIER_MODEL_PATH.exists()})')