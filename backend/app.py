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
from utils.fileops import save_uploads, cleanup

app = FastAPI(title='EndoScan AI Backend')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Serve frontend static files
frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
if frontend_dir.exists():
    app.mount('/static', StaticFiles(directory=str(frontend_dir)), name='static')


@app.get('/')
async def root():
    """Serve the default homepage (index.html)"""
    index_file = frontend_dir / 'index.html'
    if index_file.exists():
        return FileResponse(str(index_file), media_type='text/html')
    return JSONResponse(status_code=404, content={'detail': 'index.html not found'})


@app.post('/predict')
async def predict(
    modality: str = Form(...),
    model_type: str = Form(...),
    files: list[UploadFile] = File(...)
):
    if modality not in ['ct', 'ctpet']:
        raise HTTPException(status_code=400, detail='Unsupported modality')
    if model_type not in ['segmentation', 'detection']:
        raise HTTPException(status_code=400, detail='Unsupported model_type')

    tmp_dir = None
    try:
        tmp_dir, input_paths = save_uploads(files)

        if model_type == 'segmentation':
            output_path, preview_path, err = run_nnunet_inference(tmp_dir, input_paths, modality)
            if err:
                return JSONResponse(status_code=200, content={'status': 'error', 'detail': err, 'modality': modality, 'model': model_type})

            mask_data = nib.load(str(output_path)).get_fdata()
            stats = {
                'volume_cc': float((mask_data > 0).sum()),
                'voxel_count': int((mask_data > 0).sum())
            }

            result = {
                'status': 'done',
                'job_id': uuid.uuid4().hex,
                'modality': modality,
                'model': model_type,
                'stats': stats,
                'download': f'/download/{output_path.name}'
            }
            if preview_path is not None:
                result['preview'] = f'/download/{preview_path.name}'
            return result

        detections, inference_images, err = run_yolo_inference(input_paths, modality)
        if err:
            return JSONResponse(status_code=200, content={'status': 'error', 'detail': err, 'modality': modality, 'model': model_type})

        return {
            'status': 'done',
            'job_id': uuid.uuid4().hex,
            'modality': modality,
            'model': model_type,
            'stats': {'volume_cc': None, 'voxel_count': None},
            'detections': detections,
            'inference_images': inference_images,
            'download': None
        }

    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(ex))

    finally:
        cleanup(tmp_dir)


@app.get('/download/{fname}')
async def download(fname: str):
    out_file = OUTPUT_DIR / fname
    if not out_file.exists():
        raise HTTPException(status_code=404, detail='File not found')
    media_type = 'image/png' if out_file.suffix == '.png' else 'application/gzip'
    return FileResponse(str(out_file), media_type=media_type, filename=fname)
