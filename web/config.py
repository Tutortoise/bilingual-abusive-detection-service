import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Model paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "abussive_detection.keras")
VECTORIZER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer_config.pkl")

# Abusive words paths
EN_ABUSIVE_PATH = os.path.join(PROJECT_ROOT, "data", "en_abusive.csv")
ID_ABUSIVE_PATH = os.path.join(PROJECT_ROOT, "data", "id_abusive.csv")
