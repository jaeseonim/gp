import config
from src.data_loader import load_and_split_data
from src.model_utils import (
    get_label_encoder, 
    get_baseline_model, 
    get_balanced_model, 
    calculate_sample_weights
)
from src.trainer import train_and_evaluate
from src.tuner import tune_model

def main():
    # 1. 데이터 로드 (7:2 분할)
    print("📥 데이터를 로드하고 7:2로 분할합니다...")
    X_train, y_train, X_val, y_val = load_and_split_data(config)

    # 2. Label Encoder 준비
    le = get_label_encoder(y_train)
    # 가중치 미리 계산 (04번, 05번 실험에서 공통 사용)
    s_weights = calculate_sample_weights(y_train, le)

    # --- [Experiment 01: RF Baseline] ---
    print("\n--- [Experiment 01: RF Baseline] ---")
    rf_model = get_baseline_model("rf", config)
    train_and_evaluate(rf_model, "rf", X_train, y_train, X_val, y_val, "01_rf_baseline", le)

    # --- [Experiment 02: XGBoost Baseline] ---
    print("\n--- [Experiment 02: XGBoost Baseline] ---")
    xgb_model = get_baseline_model("xgboost", config)
    train_and_evaluate(xgb_model, "xgboost", X_train, y_train, X_val, y_val, "02_xgb_baseline", le)

    # --- [Experiment 03: RF Balanced] ---
    print("\n--- [Experiment 03: RF Balanced] ---")
    rf_balanced = get_balanced_model("rf", config)
    train_and_evaluate(rf_balanced, "rf", X_train, y_train, X_val, y_val, "03_rf_balanced", le)

    # --- [Experiment 04: XGBoost Balanced] ---
    print("\n--- [Experiment 04: XGBoost Balanced] ---")
    xgb_balanced = get_balanced_model("xgboost", config)
    train_and_evaluate(
        xgb_balanced, "xgboost", 
        X_train, y_train, X_val, y_val, 
        "04_xgb_balanced", le, 
        sample_weight=s_weights
    )

    # --- [Experiment 05: XGBoost Tuned] ---
    print("\n--- [Experiment 05: XGBoost Tuned] ---")
    # n_iter는 시간 관계상 20 정도로 시작하고, 나중에 성능을 더 올리고 싶으면 50~100으로 늘리세요.
    best_xgb, best_params = tune_model("xgboost", X_train, y_train, X_val, y_val, config, le, n_iter=100)

    # ⭐ 중요: 튜닝된 모델을 평가할 때도 반드시 sample_weight=s_weights를 넣어줘야 합니다!
    train_and_evaluate(
        best_xgb, "xgboost", 
        X_train, y_train, X_val, y_val, 
        "05_xgb_tuned", le, 
        sample_weight=s_weights
    )

    print("\n✨ 모든 실험이 완료되었습니다. experiments 폴더를 확인하세요!")

if __name__ == "__main__":
    main()