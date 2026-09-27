"""
Central configuration for the Pneumonia Chest X-Ray Classification project.
All paths, image settings, and shared constants live here so every module
(and the Streamlit app) stays in sync.
"""
import os

# --- Image / training settings ---
IMG_SIZE = (150, 150)
INPUT_SHAPE = IMG_SIZE + (3,)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
BATCH_SIZE = 32
SEED = 42

# --- Project paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
