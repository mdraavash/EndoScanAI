# EndoScanAI - YOLO and Classification Configuration Update
## April 2, 2026

### Changes Made

#### 1. **YOLO Detection - Single File Input** 
**File:** `backend/models/yolo_infer.py`

**Changes:**
- Simplified `run_yolo_inference()` to accept **exactly ONE file** (single modality)
- Removed pseudo-fusion code that attempted to combine PET and CT
- Model processes CT and PET **independently** (as designed - mixed dataset training approach)
- Single input can be either CT or PET image from the mixed-trained YOLO model
- Returns detections with modality label and visualization

**API Behavior:**
- Input: 1 file (CT or PET)
- Modality: Can be 'ct' or 'ctpet' (both process single input independently)
- Output: Detections + annotated visualization images

#### 2. **Backend Validation** 
**File:** `backend/app.py`

**Changes:**
- Detection endpoint now validates: `len(files) == 1` 
- Classification endpoint now validates:
  - `modality == 'ct'` (rejects CT+PET)
  - `len(files) == 1`
- Returns helpful error messages if constraints violated

#### 3. **Frontend Model Selection**
**File:** `frontend/js/app.js`

**Changes:**
- Added `updateFileHint()` function to show max files per model type:
  - **Detection:** max 1 file (CT or PET)
  - **Classification:** max 1 file (CT only)
  - **Segmentation:** 1 file (CT) or 2 files (CT+PET)

- Updated toggle button handler to:
  - Disable Classification button when CT+PET selected
  - Switch to Segmentation if Classification was active and user selects CT+PET
  - Show tooltip: "Classification only available for CT modality"
  - Re-validate file selection when model type changes

- Updated `handleFiles()` to enforce per-model file limits

### Summary of Input Requirements

| Model Type | CT Modality | CT+PET Modality | Files Required |
|-----------|-----------|-----------------|-----------------|
| **Segmentation** | ✓ 1 file | ✓ 2 files | 1-2 |
| **Detection** | ✓ 1 file | ✓ 1 file | 1 (ONLY) |
| **Classification** | ✓ 1 file | ✗ NOT AVAILABLE | 1 (CT only) |

### User Workflow

1. **For Detection (YOLO):**
   - Select either CT or CT+PET modality
   - Drop ONE image file (CT or PET)
   - Click Detection model
   - Results show detections from that single modality

2. **For Classification (Radiomics):**
   - Select CT modality only
   - Drop ONE CT image file
   - Click Classification model
   - Results show malignancy classification

3. **For Segmentation (nnUNet):**
   - Select CT modality → 1 file
   - Select CT+PET modality → 2 files (CT + PET)
   - Click Segmentation model
   - Results show segmentation masks and volumes

### Technical Details

**YOLO Model Characteristics:**
- Trained on mixed CT+PET dataset
- Processes images independently (no true multimodal fusion)
- Can detect objects in either CT or PET modality
- Takes one file at a time for inference

**Classification (Radiomics):**
- Based on radiomics features from CT images
- Not available for CT+PET (requires CT-specific features)
- Uses pre-trained scikit-learn pipeline

**Segmentation (nnUNet):**
- CT-only model: processes CT images
- CT+PET model: processes CT+PET image pairs
- Both models available and working correctly
