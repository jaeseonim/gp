# test.py
# pkl 파일 불러오기. 머신러닝에서는 하이퍼파라미터, 파라미터 둘 다 저장되어 있어.
import joblib
import config
from src.data_loader import load_test_data, load_and_split_data
from src.model_utils import get_label_encoder
from src.trainer import train_and_evaluate
import os

def run_final_test():
    # 1. 데이터 로드
    print("🧪 테스트 데이터를 로드합니다...")
    X_test, y_test = load_test_data(config)
    
    # 2. Label Encoder 복원 (학습 데이터의 클래스 순서를 그대로 가져와야 함)
    # 기존 data_loader를 통해 y_train을 잠깐 불러와서 인코더를 다시 만듭니다.
    X_train, y_train, _, _ = load_and_split_data(config)
    le = get_label_encoder(y_train)

    # 3. 저장된 최적 모델 불러오기
    # 05_xgb_tuned 폴더 안에 저장된 모델 파일 이름을 확인하세요.
    model_path = os.path.join(config.EXP_DIR, "05_xgb_tuned", "05_xgb_tuned_model.pkl")
    
    if not os.path.exists(model_path):
        print("❌ 모델 파일을 찾을 수 없습니다. 경로를 확인하세요!")
        return

    print(f"📦 모델을 불러오는 중: {model_path}")
    best_model = joblib.load(model_path)

    # 4. 최종 평가 (trainer의 함수 재사용)
    print("\n🏁 [FINAL TEST] 외부 데이터셋 평가 시작...")
    train_and_evaluate(
        model=best_model,
        model_name="xgboost",
        X_train=X_train, y_train=y_train, # trainer 구조상 필요하지만 fit은 하지 않음
        X_val=X_test, y_val=y_test,       # 핵심: 검증셋 자리에 테스트셋을 넣음
        exp_name="FINAL_TEST_RESULT",
        le=le
    )
    print("\n✨ 최종 결과가 'experiments/FINAL_TEST_RESULT'에 저장되었습니다!")

if __name__ == "__main__":
    run_final_test()