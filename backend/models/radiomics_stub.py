"""
Minimal radiomics stub for when PyRadiomics is not installed.
This allows the classifier module to import successfully but will raise
a proper error when feature extraction is attempted.
"""

class RadiomicsFeatureExtractor:
    """Stub class that raises an error when instantiated."""
    
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "PyRadiomics is not installed. "
            "To use classification, install with:\n"
            "  pip install numpy scipy scikit-learn && pip install --no-build-isolation pyradiomics\n"
            "Or use a pre-built environment with PyRadiomics already compiled."
        )
    
    def enableAllFeatures(self):
        pass
    
    def enableFeatureClassByName(self, name):
        pass


class featureextractor:
    """Stub module that provides RadiomicsFeatureExtractor."""
    RadiomicsFeatureExtractor = RadiomicsFeatureExtractor
