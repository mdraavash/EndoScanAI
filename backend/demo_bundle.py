"""
create_demo_bundle.py
---------------------
Run this script once to package the Fold 66 genuinely unseen model
artifacts into a single 'demo_bundle.pkl' for the live inference demo.
"""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path

# Paths (Adjust if your files are in a different folder)
MODEL_DIR = Path("models/trained_weights/classifier")

print("Loading Fold 66 artifacts...")
scaler = joblib.load(MODEL_DIR / "demo_scaler.pkl")
selector = joblib.load(MODEL_DIR / "demo_selector.pkl")
classifier = joblib.load(MODEL_DIR / "demo_model.pkl")

print("Building sklearn Pipeline...")
pipeline = Pipeline([
    ('scaler', scaler),
    ('selector', selector),
    ('classifier', classifier)
])

print("Extracting feature column names from CSV...")
df = pd.read_csv("radiomics_features_final.csv")
# Get the 306 feature columns exactly as they appear in the dataset
features_in = [c for c in df.columns if c not in ["patient_id", "binary_grade", "grade", "split"]]

# Build the bundle dictionary exactly how your app expects it
bundle = {
    "pipeline": pipeline,
    "features_in_": features_in,
    "threshold": 0.5,  # Standard 50% threshold for the demo
    "label_map": {0: "Low Grade", 1: "High Grade"}
}

out_path = MODEL_DIR / "demo_bundle.pkl"
joblib.dump(bundle, out_path, compress=3)

print(f"\n[SUCCESS] Saved {out_path}!")
print("You can now update your config.py to point to this bundle.")