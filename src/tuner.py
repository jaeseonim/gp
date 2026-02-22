# src/tuner.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, f1_score
from src.model_utils import get_search_space, calculate_sample_weights, get_balanced_model

def tune_model(model_name, X_train, y_train, X_val, y_val, config, le, n_iter=10):
    """
    고정된 Validation set을 사용하여 하이퍼파라미터 튜닝을 수행 (IndexError 수정 버전)
    """
    # 1. Balanced 모델 객체 생성
    base_model = get_balanced_model(model_name, config)
    
    # 2. 데이터 합치기 (PredefinedSplit은 합쳐진 데이터를 입력받아야 함)
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    
    # 3. PredefinedSplit 설정 (Train: -1, Val: 0)
    test_fold = np.concatenate([
        np.full(len(X_train), -1), 
        np.full(len(X_val), 0)
    ])
    ps = PredefinedSplit(test_fold)

    # 4. 최적화 기준 설정
    scorer = make_scorer(f1_score, average='macro')
    param_dist = get_search_space(model_name)
    
    # 5. XGBoost용 인코딩 및 가중치 처리
    y_combined_fit = le.transform(y_combined) if model_name == "xgboost" else y_combined
    
    fit_params = {}
    if model_name == "xgboost":
        # 가중치도 Train과 Val 합친 길이에 맞춰야 함 (Val 구간은 0이나 1로 채움)
        s_weights_train = calculate_sample_weights(y_train, le)
        s_weights_val = np.ones(len(y_val)) # 검증셋 가중치는 1로 설정
        s_weights_combined = np.concatenate([s_weights_train, s_weights_val])
        fit_params = {'sample_weight': s_weights_combined}

    # 6. Randomized Search 설정
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=ps, # 여기서 ps가 제대로 작동하려면 fit에 combined 데이터를 넣어야 함
        verbose=1,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

    # 7. 실행 (X_combined를 넣어주는 것이 핵심!)
    print(f"🔎 {model_name} 하이퍼파라미터 튜닝 시작 (총 {n_iter}회 시도)...")
    search.fit(X_combined, y_combined_fit, **fit_params)

    print(f"🏆 최적 파라미터: {search.best_params_}")
    print(f"📈 최고 Macro F1 점수: {search.best_score_:.4f}")
    
    # 최적의 모델만 반환 (이 모델은 이미 내부적으로 X_train에 대해 학습된 상태)
    return search.best_estimator_, search.best_params_