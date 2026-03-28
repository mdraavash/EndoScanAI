import os
import torch
from pathlib import Path

# BASE_DIR should be D:\aavash\6th sem\minor project\EndoScanAI
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / 'backend'

# --- DYNAMIC PATH FINDER ---
# This ensures that even if there is a typo in the folder name, Python finds it.
def find_model_path(base_path, folder_name):
    for root, dirs, _ in os.walk(base_path):
        if folder_name in dirs:
            return Path(root) / folder_name
    return None

# Locate the results root
RESULTS_ROOT = find_model_path(BACKEND_DIR, 'nnunet_results_ct_pet')

if RESULTS_ROOT:
    # Set the environment variables exactly where they were found
    os.environ['nnUNet_raw'] = str(RESULTS_ROOT.parent / 'nnunet_raw')
    os.environ['nnUNet_preprocessed'] = str(RESULTS_ROOT.parent / 'nnunet_preprocessed_ct_pet')
    os.environ['nnUNet_results'] = str(RESULTS_ROOT)
    
    # Drill down to the specific 2D model folder
    # Note: Using glob('*2d') handles cases where the name might have extra underscores
    try:
        NNUNET_MODEL_PATH = next(RESULTS_ROOT.glob('**/nnUNetTrainer*2d'))
    except StopIteration:
        NNUNET_MODEL_PATH = RESULTS_ROOT # Fallback
else:
    NNUNET_MODEL_PATH = Path("NOT_FOUND")

# --- REST OF CONFIG ---
YOLO_MODEL_PATH = BACKEND_DIR / 'models' / 'trained_weights' / 'yolo' / 'v10m_combined.pt'
OUTPUT_DIR = BACKEND_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NNUNET_FOLD = '0'
NNUNET_CHECKPOINT = 'checkpoint_best.pth'

print(f"--- Config Loaded ---")
print(f"Device: {DEVICE}")
print(f"Final Model Path: {NNUNET_MODEL_PATH}")
print(f"Model Path Exists: {NNUNET_MODEL_PATH.exists()}")