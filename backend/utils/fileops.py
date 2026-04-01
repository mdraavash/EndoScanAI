import shutil
import tempfile
from pathlib import Path
from fastapi import HTTPException


def save_uploads(files):
    tmp_dir = Path(tempfile.mkdtemp(prefix='endoscan_input_'))
    input_paths = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if (ext in ['.gz', '.nii', '.dcm']) or f.filename.lower().endswith('.nii.gz'):
            pass
        else:
            raise HTTPException(status_code=400, detail=f'Unsupported extension {ext} in {f.filename}')

        file_path = tmp_dir / f.filename
        with file_path.open('wb') as out:
            shutil.copyfileobj(f.file, out)
        input_paths.append(file_path)
    return tmp_dir, input_paths


def cleanup(tmp_dir):
    if tmp_dir and tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)