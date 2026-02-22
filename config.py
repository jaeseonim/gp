import os

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/cleaned_features.csv")
EXP_DIR = os.path.join(BASE_DIR, "experiments")

# 2. 7:1:1 분할 비율 설정 (9개 파일 기준)
# 전체 9개 중 1개를 Test로 빼기 위해: 1/9 ≈ 0.111
TEST_SIZE = 0.111   

# Test가 빠진 남은 8개 중 1개를 Val로 빼기 위해: 1/8 = 0.125
VAL_SIZE = 0.125    

# 3. 모델 및 데이터 메타데이터
RANDOM_STATE = 42
TARGET_COL = 'Classification'
GROUP_COL = 'Source_File' # 이미지 단위 분할을 위한 핵심 키