"""
mask_diagnostic.py
------------------
Checks what the foreground mask looks like at inference vs training.

The 100% High Grade bug when 306/306 features are aligned almost always
means the mask coverage at inference is different from training.

Training:  mask = actual segmentation mask PNG (sparse, uterus region only)
Inference: mask = all pixels > 127 in the image (dense, whole crop)

This changes EVERY texture/shape feature significantly.

Run:
    python mask_diagnostic.py C3N-01003_image.nii.gz
    python mask_diagnostic.py TCGA-FI-A2EY_image.nii.gz
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import importlib, joblib

warnings.filterwarnings("ignore")

BUNDLE_PATH   = Path("models/trained_weights/classifier/model_bundle.pkl")
SAVE_MODEL_PY = Path("models/trained_weights/classifier/save_model.py")
RADIOMICS_CSV = Path("models/trained_weights/classifier/radiomics_features_final.csv")
METADATA_CSV  = Path("models/trained_weights/classifier/uterus_classification_metadata.csv")

# ─────────────────────────────────────────────────────────────────────────────

def register_shim():
    spec = importlib.util.spec_from_file_location("shim", str(SAVE_MODEL_PY))
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    for n in ("__main__", "__mp_main__", "save_model", "shim"):
        mod = sys.modules.setdefault(n, shim)
        setattr(mod, "CorrelationLassoSelector", shim.CorrelationLassoSelector)


def read_nifti_u8(path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if arr.ndim == 2: arr = arr[None]
    z = int(np.argmin(arr.shape))
    if z != 0: arr = np.moveaxis(arr, z, 0)
    if arr.dtype == np.uint8: return arr
    arr = arr.astype(np.float32)
    fin = np.isfinite(arr)
    arr = np.where(fin, arr, 0.0)
    mn, mx = float(arr[fin].min()), float(arr[fin].max())
    if mx <= mn: return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


def get_extractor_255():
    """Extractor using label=255 — what inference currently uses."""
    from radiomics import featureextractor
    import logging; logging.getLogger("radiomics").setLevel(logging.ERROR)
    s = {"binWidth":25,"interpolator":sitk.sitkBSpline,
         "resampledPixelSpacing":None,"verbose":False,
         "label":255,"force2D":True,"force2Ddimension":0}
    e = featureextractor.RadiomicsFeatureExtractor(**s)
    e.enableAllFeatures(); e.enableFeatureClassByName("shape2D")
    return e


def get_extractor_1():
    """Extractor using label=1 — what training used (PNG mask binarised to 0/1)."""
    from radiomics import featureextractor
    import logging; logging.getLogger("radiomics").setLevel(logging.ERROR)
    s = {"binWidth":25,"interpolator":sitk.sitkBSpline,
         "resampledPixelSpacing":None,"verbose":False,
         "label":1,"force2D":True,"force2Ddimension":0}
    e = featureextractor.RadiomicsFeatureExtractor(**s)
    e.enableAllFeatures(); e.enableFeatureClassByName("shape2D")
    return e


def make_mask(arr_2d, label_val):
    """Make mask with pixels set to label_val where foreground."""
    fg = ((arr_2d > 127).astype(np.uint8)) * label_val
    if fg.max() == 0:
        fg = np.full_like(arr_2d, label_val, dtype=np.uint8)
    mask = sitk.GetImageFromArray(fg)
    mask.SetSpacing((1.0, 1.0))
    return sitk.Cast(mask, sitk.sitkUInt8)


def extract_profile(volume, extractor, label_val):
    slices = []
    for z in range(volume.shape[0]):
        sl = volume[z]
        img = sitk.GetImageFromArray(sl.astype(np.float32))
        img.SetSpacing((1.0, 1.0))
        mask = make_mask(sl, label_val)
        try:
            res = extractor.execute(img, mask)
            slices.append({k: float(v) for k, v in res.items()
                           if not k.startswith("diagnostics_")})
        except Exception as e:
            print(f"    slice {z} failed: {e}")
    if not slices: return None
    df = pd.DataFrame(slices)
    return pd.concat([df.mean().add_suffix("_Mean"),
                      df.max().add_suffix("_Max"),
                      df.std().fillna(0).add_suffix("_Std")])


def predict(pipeline, feat_cols, threshold, profile):
    df = pd.DataFrame([profile]).reindex(columns=feat_cols, fill_value=0.0)
    X  = df.values.astype(np.float32)
    p  = pipeline.predict_proba(X)[0]
    return float(p[1]), "High Grade" if p[1] >= threshold else "Low Grade"


def compare_to_training(pipeline, feat_cols, profile, radiomics_csv, patient_id=None):
    """Show Z-scores of selected features vs training distribution."""
    df_train = pd.read_csv(radiomics_csv)
    sel      = pipeline.named_steps["selector"]
    kept     = [feat_cols[i] for i in sel.keep_cols_]

    print(f"\n  {'Feature':<52} {'Tr.mean':>9} {'Tr.std':>8} "
          f"{'Infer':>9} {'Z':>7}")
    print(f"  {'-'*52} {'-'*9} {'-'*8} {'-'*9} {'-'*7}")

    any_extreme = False
    for i, col in zip(sel.keep_cols_, kept):
        short = col.replace("original_", "")[-52:]
        if col not in df_train.columns:
            print(f"  {short:<52}  NOT IN TRAINING CSV"); continue
        tr_vals = df_train[col].dropna().values.astype(float)
        tr_mean = float(np.mean(tr_vals))
        tr_std  = float(np.std(tr_vals)) + 1e-9
        inf_val = float(profile.get(col, 0.0))
        z       = (inf_val - tr_mean) / tr_std
        flag    = "  ← EXTREME" if abs(z) > 4 else ""
        if abs(z) > 4: any_extreme = True
        print(f"  {short:<52} {tr_mean:>9.3f} {tr_std:>8.3f} "
              f"{inf_val:>9.3f} {z:>7.2f}{flag}")
    return any_extreme


# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python mask_diagnostic.py <image.nii.gz> [patient_id]")
        return

    nifti_path = Path(sys.argv[1])
    patient_id = sys.argv[2] if len(sys.argv) > 2 else nifti_path.stem

    print(f"\n{'='*65}")
    print(f"  MASK & FEATURE DIAGNOSTIC")
    print(f"  File: {nifti_path.name}")
    print(f"{'='*65}")

    # Load bundle
    register_shim()
    bundle    = joblib.load(BUNDLE_PATH)
    pipeline  = bundle["pipeline"]
    feat_cols = bundle["feature_cols"]
    threshold = bundle["threshold"]
    scaler    = pipeline.named_steps["scaler"]

    # Read volume
    vol = read_nifti_u8(nifti_path)
    print(f"\n  Volume: shape={vol.shape}  min={vol.min()}  max={vol.max()}")

    # Show slice mask coverage
    print(f"\n  Mask coverage per slice (pixels > 127 as % of total pixels):")
    for z in range(vol.shape[0]):
        sl      = vol[z]
        fg_pct  = 100.0 * (sl > 127).sum() / sl.size
        bar     = "█" * int(fg_pct / 5)
        print(f"    Slice {z+1:2d}: {fg_pct:5.1f}%  {bar}")

    # ── Test 1: current inference (label=255, mask=image>127 scaled to 255) ──
    print(f"\n{'─'*65}")
    print(f"  TEST 1: label=255, mask = pixels>127 set to 255  (current inference)")
    print(f"{'─'*65}")
    ext_255  = get_extractor_255()
    prof_255 = extract_profile(vol, ext_255, label_val=255)
    if prof_255 is not None:
        ph, lbl = predict(pipeline, feat_cols, threshold, prof_255)
        print(f"  → {lbl}  P(High)={ph:.4f}")
        extreme = compare_to_training(pipeline, feat_cols, prof_255,
                                      RADIOMICS_CSV, patient_id)
        if extreme:
            print(f"\n  ⚠ EXTREME Z-SCORES — inference features outside training range")

    # ── Test 2: label=1, mask=pixels>127 set to 1 ────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  TEST 2: label=1, mask = pixels>127 set to 1")
    print(f"{'─'*65}")
    ext_1  = get_extractor_1()
    prof_1 = extract_profile(vol, ext_1, label_val=1)
    if prof_1 is not None:
        ph, lbl = predict(pipeline, feat_cols, threshold, prof_1)
        print(f"  → {lbl}  P(High)={ph:.4f}")

    # ── Test 3: label=255, whole image as mask (all pixels = 255) ────────────
    print(f"\n{'─'*65}")
    print(f"  TEST 3: label=255, full-frame mask (all pixels = 255)")
    print(f"{'─'*65}")

    def extract_fullframe(volume, extractor):
        slices = []
        for z in range(volume.shape[0]):
            sl   = volume[z]
            img  = sitk.GetImageFromArray(sl.astype(np.float32))
            img.SetSpacing((1.0, 1.0))
            fg   = np.full_like(sl, 255, dtype=np.uint8)
            mask = sitk.GetImageFromArray(fg); mask.SetSpacing((1.0,1.0))
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            try:
                res = extractor.execute(img, mask)
                slices.append({k:float(v) for k,v in res.items()
                                if not k.startswith("diagnostics_")})
            except: pass
        if not slices: return None
        df = pd.DataFrame(slices)
        return pd.concat([df.mean().add_suffix("_Mean"),
                          df.max().add_suffix("_Max"),
                          df.std().fillna(0).add_suffix("_Std")])

    prof_full = extract_fullframe(vol, ext_255)
    if prof_full is not None:
        ph, lbl = predict(pipeline, feat_cols, threshold, prof_full)
        print(f"  → {lbl}  P(High)={ph:.4f}")

    # ── Training reference for this patient ───────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  TRAINING REFERENCE")
    print(f"{'─'*65}")
    df_train = pd.read_csv(RADIOMICS_CSV)
    meta     = pd.read_csv(METADATA_CSV)
    merged   = df_train.merge(meta[["cases.submitter_id","label"]],
                               left_on="patient_id",
                               right_on="cases.submitter_id", how="inner")

    pid_clean = patient_id.replace("_image","")
    row = merged[merged["patient_id"] == pid_clean]
    if not row.empty:
        true_label = "High Grade" if row["label"].iloc[0] == 1 else "Low Grade"
        print(f"  Patient {pid_clean}: true label = {true_label}")

        sel   = pipeline.named_steps["selector"]
        kept  = [feat_cols[i] for i in sel.keep_cols_]
        tr_X  = row[feat_cols].values.astype(np.float32)
        tr_ph = float(pipeline.predict_proba(tr_X)[0][1])
        tr_lb = "High Grade" if tr_ph >= threshold else "Low Grade"
        print(f"  Training CSV prediction: {tr_lb}  P(High)={tr_ph:.4f}")
        print(f"  (This is what the model SHOULD output for this patient)")
    else:
        print(f"  Patient {pid_clean} not found in training CSV.")
        print(f"  Available IDs (first 5): {list(merged['patient_id'][:5])}")

    print(f"\n{'='*65}")
    print(f"  CONCLUSION")
    print(f"{'='*65}")
    print(f"  Compare the P(High) values from Test 1/2/3 above.")
    print(f"  The test that matches 'Training CSV prediction' is the")
    print(f"  correct mask strategy to use in classifier_infer.py.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()