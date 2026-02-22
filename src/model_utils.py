import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight  

def get_label_encoder(y_train):
    """
    세포 클래스 명칭(문자열)을 숫자(0, 1, 2...)로 변환하는 도구
    XGBoost 학습을 위해 반드시 필요해.
    """
    le = LabelEncoder()
    le.fit(y_train)
    return le

def get_baseline_model(model_name, config):
    """
    가장 기본적인 베이스라인 모델을 반환 (가중치 미적용 상태)
    """
    if model_name == "rf":
        # Random Forest: 아무 옵션 없이 기본 설정으로 시작
        return RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
    
    elif model_name == "xgboost":
        # XGBoost: 다중 분류(multi:softmax) 기본 설정
        return XGBClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            # 최신 XGBoost는 hist 방식이 기본이지만 명시적으로 지정
            tree_method='hist' 
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_balanced_model(model_name, config):
    """
    클래스 불균형을 고려한 가중치 설정 모델 반환
    """
    if model_name == "rf":
        # Random Forest는 내부에 'balanced' 옵션이 있어서 아주 쉬움!
        return RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced' # 핵심: 적은 클래스에 더 큰 가중치를 줌
        )
    
    elif model_name == "xgboost":
        # XGBoost는 모델 생성 시가 아니라 '학습 시' 가중치를 줘야 함
        # 여기서는 모델 객체만 반환하고, 가중치 계산은 trainer나 main에서 수행
        return XGBClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist'
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def calculate_sample_weights(y, le):
    """
    XGBoost 학습을 위해 샘플별 가중치를 계산함
    """
    # 1. 클래스별 가중치 계산 (N / (n_classes * count))
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    
    # 2. 모든 샘플에 가중치 매핑
    sample_weights = np.array([weight_dict[cls] for cls in y])
    return sample_weights

def get_search_space(model_name):
    """
    각 모델별 하이퍼파라미터 탐색 범위를 반환
    """
    if model_name == "rf":
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    elif model_name == "xgboost":
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
    return {}