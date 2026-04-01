"""
classifier_infer.py
-------------------
Uterine cancer grade classifier for EndoScan AI.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import config

warnings.filterwarnings("ignore")

LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("classifier_infer")

_BUNDLE    = None
_PIPELINE  = None
_FEAT_COLS = None
_THRESHOLD = None
_EXTRACTOR = None


def _register_selector_shim() -> type:
    """Injects CorrelationLassoSelector so joblib can unpickle the model safely."""
    p = Path(config.CLASSIFIER_SELECTOR_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Selector shim not found: {p}")
    spec = importlib.util.spec_from_file_location("classifier_selector_shim", str(p))
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    cls = getattr(shim, "CorrelationLassoSelector", None)
    if cls is None:
        raise AttributeError("save_model.py must define CorrelationLassoSelector.")

    for mod_name in ("__main__", "__mp_main__", "save_model", "classifier_selector_shim"):
        mod = sys.modules.setdefault(mod_name, shim)
        setattr(mod, "CorrelationLassoSelector", cls)

    return cls


def _load_bundle() -> None:
    """Loads the model bundle using the absolute path in config.py."""
    global _BUNDLE, _PIPELINE, _FEAT_COLS, _THRESHOLD
    import joblib
    import torch

    _register_selector_shim()
    
    # PyTorch 2.6+ changed weights_only default to True; we need False for compatibility
    # with pickled objects from older PyTorch versions
    original_torch_load = torch.load
    def torch_load_compat(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    torch.load = torch_load_compat
    try:
        raw = joblib.load(config.CLASSIFIER_MODEL_PATH)
    finally:
        torch.load = original_torch_load

    if isinstance(raw, dict):
        _BUNDLE    = raw
        _PIPELINE  = raw["pipeline"]
        _FEAT_COLS = raw.get("feature_cols")
        bundle_thr = float(raw.get("threshold", 0.5))
    else:
        log.warning("Bundle is a bare pipeline. Re-run save_model.py.")
        _BUNDLE    = {}
        _PIPELINE  = raw
        _FEAT_COLS = None
        bundle_thr = 0.5

    cfg_thr    = getattr(config, "CLASSIFIER_THRESHOLD", None)
    _THRESHOLD = cfg_thr if (cfg_thr is not None) else bundle_thr

    log.info(f"Bundle loaded | threshold={_THRESHOLD:.4f} | feature_cols={'Present' if _FEAT_COLS else 'Missing'}")


def _ensure_model() -> None:
    if _PIPELINE is not None:
        return
    p = Path(config.CLASSIFIER_MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found: {p}")
    _load_bundle()


def _read_nifti(nifti_path: Path) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nifti_path)))
    if arr.ndim == 2: arr = arr[None, ...]
    elif arr.ndim == 4:
        sq  = np.squeeze(arr)
        arr = sq if sq.ndim == 3 else arr[..., 0]

    z_ax = int(np.argmin(arr.shape))
    if z_ax != 0: arr = np.moveaxis(arr, z_ax, 0)

    if arr.dtype == np.uint8: return arr
    arr  = arr.astype(np.float32)
    fin  = np.isfinite(arr)
    arr  = np.where(fin, arr, 0.0)
    vmin = float(arr[fin].min()) if fin.any() else 0.0
    vmax = float(arr[fin].max()) if fin.any() else 1.0

    if vmax <= vmin: return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)


def _get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is not None: return _EXTRACTOR
    
    try:
        from radiomics import featureextractor
    except ImportError:
        try:
            from radiomics.featureextractor import RadiomicsFeatureExtractor as RFE
            # Create a module-like object
            import types
            featureextractor = types.ModuleType('featureextractor')
            featureextractor.RadiomicsFeatureExtractor = RFE
        except ImportError:
            # Use our stub when real radiomics isn't available
            from . import radiomics_stub
            featureextractor = radiomics_stub
    
    settings = {
        "binWidth": 25,
        "interpolator": sitk.sitkBSpline,
        "resampledPixelSpacing": None,
        "verbose": False,
        "label": 255,
        "force2D": True,
        "force2Ddimension": 0,
    }
    _EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(**settings)
    _EXTRACTOR.enableAllFeatures()
    _EXTRACTOR.enableFeatureClassByName("shape2D")
    return _EXTRACTOR


def _make_mask_255(arr_2d: np.ndarray) -> sitk.Image:
    fg = ((arr_2d > 127).astype(np.uint8)) * 255
    if fg.max() == 0: fg = np.full_like(arr_2d, 255, dtype=np.uint8)
    mask = sitk.GetImageFromArray(fg)
    mask.SetSpacing((1.0, 1.0))
    return sitk.Cast(mask, sitk.sitkUInt8)


def _extract_slice(arr_2d: np.ndarray, extractor) -> dict:
    img = sitk.GetImageFromArray(arr_2d.astype(np.float32))
    img.SetSpacing((1.0, 1.0))
    result = extractor.execute(img, _make_mask_255(arr_2d))
    return {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}


def _process_volume(volume: np.ndarray) -> pd.Series:
    extractor = _get_extractor()
    Z = volume.shape[0]
    slice_features = []

    for z in range(Z):
        try:
            feats = _extract_slice(volume[z], extractor)
            slice_features.append(feats)
        except Exception as exc:
            log.warning(f"Slice {z + 1}/{Z} skipped: {exc}")

    if not slice_features:
        raise ValueError("No slices could be processed from this NIfTI volume.")

    df_sl = pd.DataFrame(slice_features)
    profile = pd.concat([
        df_sl.mean().add_suffix("_Mean"),
        df_sl.max().add_suffix("_Max"),
        df_sl.std().fillna(0).add_suffix("_Std"),
    ])
    return profile


def _align_columns(profile: pd.Series) -> np.ndarray:
    if _FEAT_COLS is None:
        return profile.values.astype(np.float32).reshape(1, -1)
    df = pd.DataFrame([profile])
    df = df.reindex(columns=_FEAT_COLS, fill_value=0.0)
    return df.values.astype(np.float32)


def run_classifier_inference(input_paths: list, modality: str = "CT") -> tuple[float | None, str | None, float | None, str | None]:
    if not input_paths: return None, None, None, "No input files provided."
    nifti_path = Path(input_paths[0])
    if not nifti_path.exists(): return None, None, None, f"Input file not found: {nifti_path}"

    try: _ensure_model()
    except Exception as exc: return None, None, None, f"Model load failed: {exc}"

    try: volume = _read_nifti(nifti_path)
    except Exception as exc: return None, None, None, f"NIfTI read failed: {exc}"

    if volume.max() == 0: return None, None, None, "NIfTI volume is entirely zero."

    try: profile = _process_volume(volume)
    except Exception as exc: return None, None, None, f"Feature extraction failed: {exc}"

    try:
        X = _align_columns(profile)
        if int((X != 0).sum()) == 0:
            return None, None, None, "All input features are zero."
    except Exception as exc: return None, None, None, f"Column alignment failed: {exc}"

    try:
        proba = _PIPELINE.predict_proba(X)[0]
        prob_high = float(proba[1])
        confidence = float(max(proba))
        label = "High Grade" if prob_high >= _THRESHOLD else "Low Grade"
    except Exception as exc: return None, None, None, f"Prediction failed: {exc}"

    return round(prob_high, 4), label, round(confidence, 4), None