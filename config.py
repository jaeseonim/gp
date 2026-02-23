# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/cleaned_features.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data/processed/cleaned_features_test.csv")
EXP_DIR = os.path.join(BASE_DIR, "experiments")

# 9개 중 2개를 Validation으로 사용 (2/9 ≈ 0.222)
VAL_SIZE = 0.222 
RANDOM_STATE = 42
TARGET_COL = 'Classification'
GROUP_COL = 'Source_File'