import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import config

MODEL_NAMES = {
    '01_rf_baseline':    'Random Forest (Baseline)',
    '02_xgb_baseline':   'XGBoost (Baseline)',
    '03_rf_balanced':    'Random Forest (Class-Weighted)',
    '04_xgb_balanced':   'XGBoost (Class-Weighted)',
    '05_xgb_tuned':      'XGBoost (Tuned)',
    'FINAL_TEST_RESULT': 'XGBoost (Final Test)',
}

# 1. 인자에 sample_weight=None 추가
def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val, exp_name, le, sample_weight=None):
    exp_path = os.path.join(config.EXP_DIR, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    
    print(f"\n🚀 [{exp_name}] 학습 시작...")

    # 라벨 인코딩 적용 (XGBoost인 경우)
    y_train_fit = le.transform(y_train) if model_name == "xgboost" else y_train
    
    # 2. sample_weight 처리 로직 (XGBoost뿐만 아니라 일반적인 대응이 가능하도록)
    if sample_weight is not None:
        model.fit(X_train, y_train_fit, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train_fit)

    # 예측 및 복원
    y_pred = model.predict(X_val)
    if model_name == "xgboost":
        y_pred = le.inverse_transform(y_pred)

    # --- [결과물 저장] ---
    
    # 1. 지표 테이블
    report_dict = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(exp_path, f"{exp_name}_report.csv"))

    # 2. 혼동 행렬
    # plt.figure(figsize=(10, 8))
    # cm = confusion_matrix(y_val, y_pred, labels=le.classes_)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    # plt.title(f"Confusion Matrix - {exp_name}")
    # plt.tight_layout()
    # plt.savefig(os.path.join(exp_path, f"{exp_name}_confusion_matrix.png"))
    # plt.close()

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val, y_pred, labels=le.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                annot_kws={'size': 12})
    plt.title(f"Confusion Matrix: {MODEL_NAMES.get(exp_name, exp_name)}",
            fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=13)
    plt.ylabel('True Class', fontsize=13)
    plt.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f"{exp_name}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 피처 중요도
    # importances = model.feature_importances_
    # feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    # plt.figure(figsize=(10, 6))
    # # 3. FutureWarning 방지를 위해 hue 설정 추가
    # sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette='viridis', legend=False)
    # plt.title(f"Feature Importance - {exp_name}")
    # plt.tight_layout()
    # plt.savefig(os.path.join(exp_path, f"{exp_name}_feature_importance.png"))
    # plt.close()

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(x=feat_imp.values, y=feat_imp.index,
                hue=feat_imp.index, palette='viridis', legend=False)
    plt.title(f"Feature Importance: {MODEL_NAMES.get(exp_name, exp_name)}",
            fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Importance Score', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    plt.tick_params(axis='both', labelsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f"{exp_name}_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 모델 저장
    joblib.dump(model, os.path.join(exp_path, f"{exp_name}_model.pkl"))
    print(f"✅ 모든 결과물이 '{exp_path}' 폴더 내에 실험명 접두사와 함께 저장되었습니다.")