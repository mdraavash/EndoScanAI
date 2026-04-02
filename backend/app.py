import mimetypes
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('application/javascript', '.js')

import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import nibabel as nib

from config import OUTPUT_DIR
from models.nnunet_infer import run_nnunet_inference
from models.yolo_infer import run_yolo_inference
from models.classifier_infer import run_classifier_inference
from utils.fileops import save_uploads, cleanup
from utils.image import save_raw_preview

app = FastAPI(title='EndoScan AI Backend')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['Content-Disposition'],
)

# ── Static files ───────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
print(f"Frontend dir: {frontend_dir}, exists: {frontend_dir.exists()}")
if frontend_dir.exists():
    print("Mounting static files")
    app.mount('/static', StaticFiles(directory='../frontend'), name='static')
else:
    print("Frontend dir not found")


@app.get('/')
async def root():
    index_file = frontend_dir / 'index.html'
    if index_file.exists():
        return FileResponse(str(index_file), media_type='text/html')
    return JSONResponse(status_code=404, content={'detail': 'index.html not found'})


@app.get('/app')
async def app_page():
    app_file = frontend_dir / 'app.html'
    if app_file.exists():
        return FileResponse(str(app_file), media_type='text/html')
    return JSONResponse(status_code=404, content={'detail': 'app.html not found'})


# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.post('/predict')
async def predict(
    modality:   str = Form(...),
    model_type: str = Form(...),
    files: list[UploadFile] = File(...)
):
    if modality not in ['ct', 'ctpet']:
        raise HTTPException(status_code=400, detail='Unsupported modality. Use ct or ctpet.')
    if model_type not in ['segmentation', 'detection', 'classification']:
        raise HTTPException(status_code=400, detail='Unsupported model_type.')

    # ── Validate file count based on model type ─────────────────────────────
    if model_type == 'detection':
        if len(files) != 1:
            raise HTTPException(status_code=400, detail='Detection model requires exactly ONE file.')
    
    if model_type == 'classification':
        if modality != 'ct':
            raise HTTPException(status_code=400, detail='Classification is only available for CT modality, not CT+PET.')
        if len(files) != 1:
            raise HTTPException(status_code=400, detail='Classification model requires exactly ONE file.')

    tmp_dir = None
    try:
        tmp_dir, input_paths = save_uploads(files)

        # ── Segmentation ─────────────────────────────────────────────────────
        if model_type == 'segmentation':
            output_path, preview_path, err = run_nnunet_inference(tmp_dir, input_paths, modality)
            if err:
                return JSONResponse(status_code=200, content={
                    'status': 'error', 'detail': err,
                    'modality': modality, 'model': model_type
                })

            mask_data = nib.load(str(output_path)).get_fdata()
            stats = {
                'volume_cc':   float((mask_data > 0).sum()),
                'voxel_count': int((mask_data > 0).sum()),
            }

            # Save a raw CT slice preview (first input file)
            raw_preview_path = None
            try:
                raw_preview_path = save_raw_preview(input_paths[0], OUTPUT_DIR)
            except Exception:
                pass  # non-fatal

            result = {
                'status':   'done',
                'job_id':   uuid.uuid4().hex,
                'modality': modality,
                'model':    model_type,
                'stats':    stats,
                'download': f'/download/{output_path.name}',
            }
            if preview_path:
                result['preview'] = f'/download/{preview_path.name}'
            if raw_preview_path:
                result['raw_preview'] = f'/download/{raw_preview_path.name}'
            return result

        # ── Detection ─────────────────────────────────────────────────────────
        if model_type == 'detection':
            detections, inference_images, err = run_yolo_inference(input_paths, modality)
            if err:
                return JSONResponse(status_code=200, content={
                    'status': 'error', 'detail': err,
                    'modality': modality, 'model': model_type
                })
            return {
                'status':          'done',
                'job_id':          uuid.uuid4().hex,
                'modality':        modality,
                'model':           model_type,
                'stats':           {'volume_cc': None, 'voxel_count': None},
                'detections':      detections,
                'inference_images': inference_images,
                'download':        None,
            }

        # ── Classification ────────────────────────────────────────────────────
        probability, label, confidence, err = run_classifier_inference(input_paths, modality)
        if err:
            return JSONResponse(status_code=200, content={
                'status': 'error', 'detail': err,
                'modality': modality, 'model': model_type
            })
        return {
            'status':      'done',
            'job_id':      uuid.uuid4().hex,
            'modality':    modality,
            'model':       model_type,
            'probability': probability,
            'label':       label,
            'confidence':  confidence,
        }

    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(ex))

    finally:
        cleanup(tmp_dir)


# ── Download endpoint ──────────────────────────────────────────────────────────
@app.get('/download/{fname}')
async def download(fname: str):
    # Safety: no path traversal
    if any(c in fname for c in ('/', '\\', '..', '<', '>')):
        raise HTTPException(status_code=400, detail='Invalid filename')

    out_file = OUTPUT_DIR / fname
    if not out_file.exists():
        raise HTTPException(status_code=404, detail='File not found')

    # Determine media type — handle double extension .nii.gz explicitly
    name_lower = fname.lower()
    if name_lower.endswith('.nii.gz') or name_lower.endswith('.gz') or name_lower.endswith('.nii'):
        media_type = 'application/octet-stream'
    elif name_lower.endswith('.png'):
        media_type = 'image/png'
    else:
        media_type = 'application/octet-stream'

    # FileResponse handles Content-Disposition itself via filename=
    # Do NOT also pass a headers dict with Content-Disposition — that causes duplicates
    return FileResponse(
        path=str(out_file),
        media_type=media_type,
        filename=fname,          # sets Content-Disposition: attachment; filename="..."
    )