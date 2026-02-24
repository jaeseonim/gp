# Day 11 (노션 일지)

날짜: 2026년 2월 23일

- 문제 정의: 쥐의 long bone 조직 사진으로부터 성장판 내의 cell들을 분류하기
- growth plate 내 cell의 종류 및 라벨링 기준
    
    
    | 클래스 | 위치 | 형태적 특징 |
    | --- | --- | --- |
    | **Hypertrophic** | 성장판 하부 | 크고 둥근 형태, 기둥형 배열 |
    | **Undetermined_1** | Hypertrophic ↔ Proliferative 경계 | 두 zone 사이 transition 세포 |
    | **Proliferative** | 성장판 중부 | 납작하고 기둥형 배열 |
    | **Undetermined_2** | Proliferative ↔ Resting 경계 | Proliferative와 Resting 사이에서 분류가 애매함 |
    | **Resting** | 성장판 상부 | 작고 불규칙 배열 |
- 데이터 정보 및 전처리
    - 11장의 WSI (.svs) 파일: 각 파일의 마우스 자세한 출처 정보는 Skip
        - 모두 Tibia 조직 사진임. 그리고 Masson 염색임.
        - data split: train/val/test를 7:2:2로 나눔
            - 1~9: train & validation
                - 1, 3, 4, 5, 6, 7: train
                - 2, 8: validation
            - 10, 11: test (hold out)
        - 각 파일별로 나눈 이유(모든 파일을 합쳐서 랜덤으로 나누지 않고): data leakage 때문 (이에 대해서 보충 설명 필요)
    - 각 WSI를 Qupath라는 오픈소스 소프트웨어를 통해 열음
        - Qupath: **open source software for bioimage analysis**.
        - Site: https://qupath.github.io/
        - Citation:  Bankhead, P. et al. QuPath: Open source software for digital pathology image analysis. Scientific Reports (2017). https://doi.org/10.1038/s41598-017-17204-5
    - Growth Plate를 먼저 ROI(Regoin of Interest)를 잡음
        - ‘Qupath extension SAM’ 프로그램 이용
        - Github: https://github.com/ksugar/qupath-extension-sam
    - 이후 ROI 안의 cell들을 QuPath Cellpose extension으로 detection함
        - ‘QuPath Cellpose extension’ 프로그램 이용
        - Github:: https://github.com/BIOP/qupath-extension-cellpose
    - Detection된 cell들을 class별로 annotation(labeling)함
        - 1~9(train & validation): Junha가 annotation함
        - 10~11(test): Jaeseon이 annotation함
        - Uncertain: 5개의 Class로 분류하기 애매한 것들은 ‘Uncertain’으로 라벨링 → 나중에 제거함
    - annotation된 정보를 바탕으로 Qupath를 통해 feature를 뽑아내어 각 사진마다 csv로 저장
        - 기본 feature
            - Source_File: features_1.csv ~ features_9.csv
            - Classification: 라벨링한 class 종류
            - 위치 정보
                - Centroid X µm
                - Centroid Y µm
            - 형태 정보
                - Area µm^2
                - Length µm
                - Perimeter µm
                - Circularity
                - Solidity
                - Max diameter µm
                - Min diameter µm
            - 염색 정보
                - Hematoxylin: Mean
                - Hematoxylin: Median
                - Hematoxylin: Min
                - Hematoxylin: Max
                - Hematoxylin: Std.Dev.
                - DAB: Mean
                - DAB: Median
                - DAB: Min
                - DAB: Max
                - DAB: Std.Dev.
- feature engineering
    - 추가한 feature
        - Relative_Pos_X: 각 source file에 대해서 최대 Centroid X와 최소 Centroid X을 바탕으로 0~1 값으로 보정한 값
            - 성장판이 가로로 놓여져 있고, resting zone에서 hypertrophic zone으로 갈수록 X 좌표 값이 커짐
            - 식: (X - min) / (max - min)
        - AR (calc Max/Min)
            - 식: Max diameter µm / Min diameter µm
        - Eccentricity (Feret-approx)
            - 식: c / a
                - a = Max diameter µm/2
                - b = Min diameter µm/2
                - c = $\sqrt{(a^2 - b^2)}$
    - 제거한 feature
        - [’Centroid X µm’, ‘Centroid Y µm’]: 기존 위치 정보 제거
        - ['Max diameter µm', 'Hematoxylin: Median', 'Perimeter µm']: 피처들간의 상관계수가 0.95 이상인 것 제거
            - 사진 첨부: Feature Correlation Matrix (numeric feature 대해서만)
- csv에서 결측치 있는 행 제거
- EDA
    - Class별 Relative_Pos_X의 분포 (box plot 사진)
        - resting에서 hypertrophic으로 갈수록 커지는 걸 확인
        - resting과 Undetermined_2는 비슷
    - Class 숫자 비교 (히스토그램 사진) (실제 개수 적자 전체 데이터개수랑)
        - Resting 개수가 매우 적음 (약 1%): 모델링 때 보정 필요
        - Undetermined_1과 Undetermined_2의 개수 비슷 (각각 약 8%)
        - Proliferative(약 61%)가 가장 많고, hypertrophic(약 22%)이 그 뒤를 이음
- Modeling
    - config.py
        - 파일 경로들
        - RANDOM_STATE = 42
        - TARGET_COL = ‘Classification’
        - GROUP_COL = ‘Source_File’
    - src/data_loader.py
        - train_df → X_train, y_train 생성
        - val_df → X_val y_val 생성
        - X: data.drop(columns=[config.TARGET_COL, config.GROUP_COL])
            - 남은 피처 개수: 19 - 2 = 17개
        - y: data[config.TARGET_COL]
    - src/model_utils.py
        - 먼저 4개의 모델을 사용 → train data로부터 훈련하여 val data로 평가한 지표가 제일 좋은 모델 하나 선정해서 하이퍼파라미터 최적화 진행
        - 4개의 모델
            - 01_rf_baseline
            - 02_xgb_baseline
            - 03_rf_balanced
            - 04_xgb_balanced
        - 4개의 모델 설명
            - 일단 앙상블 트리 기반 모델인 Random Forest와 XGBoost Valina 사용
                - 두 개의 모델 설명 필요 (하지만 난 사실 제대로 모르긴 해. TABULAR DATA에서 좋고, 데이터 값의 스케일링 필요 없다는 것만 알음…’
                - 하이퍼파라미터 아무런 설정 안 함 → 기본값
            - 이후 클래스 불균형 보정 위해 가중치 보정함
                - Random Forest: class_weight='balanced’
                - XGBoost: 따로 인자가 없음. “XGBoost는 모델 생성 시가 아니라 '학습 시' 가중치를 줘야 함” → 따로 계산해서 적용
    - src/trainer.py
        - XGBoost인 경우 라벨 인코딩 적용
        - 결과물을 experiments 폴더에 각 모델별로 저장
            - classification_report
            - confusion_matrix
            - feature_importances
            - model.pkl
    - src/tuner.py
        - 7:2 분할은 유지하면서(PredefinedSplit 이용) Randomized Search 통해 최적의 하이퍼파라미터 찾기
        - **탐색 범위 설정 (Search Space):** "자, `max_depth`는 3에서 10 사이에서 찾아보고, `learning_rate`는 0.01에서 0.3 사이에서 골라봐."라고 범위를 정해줘.
            
            ```python
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
            ```
            
        - **무한 반복 (Trials):** 컴퓨터가 이 범위 안에서 무작위로 조합을 만들어 n번씩 학습을 반복해.
            - 최적화 기준 설정 (Macro F1 사용
            - 시도 1: 깊이 5, 속도 0.1 -> F1-score: 0.51
            - 시도 2: 깊이 8, 속도 0.05 -> F1-score: 0.54 (오, 더 좋네?)
        - **최고의 조합 선택 (Best Params):** 정해진 횟수 내에서 가장 성적이 좋았던 '황금 조합'을 최종 모델로 결정해.
- main.py: train 및 predict 진행
- 결과 (각 모델에 대해서 classification_report, confusion_matrix, feature_importances 데이터 다 있어. 어떻게 보여주고 비교 어떻게 해야할지 궁금. 예를 들어서 classification_report는 4개 각각 보여주는게 아니라 하나의 table로 보여줘야 하나? 일단 데이터 자체를 아직 첨부하진 않겠음)
    - 01_rf_baseline
    - 02_xgb_baseline
    - 03_rf_balanced
    - 04_xgb_balanced
- ‘Macro F1-score’ 지표 고려했을 때, 04_xgb_balanced 가 가장 적합한 모델. 이를 바탕으로 05_xgb_tuned 모델링도 하여 데이터 추출
    - n_iter = 100 (근데 얘가 뭘 의미하는지 잘 모르겠어. 탐색 범위 설정까진 이해했거든.)
    - 최적의 하이퍼파라미터
        - 최적 파라미터: {'subsample': 1.0, 'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.6}
        - 비교
        
        | **파라미터** | **기본값 (Default)** | **최적값 (Optimized)** | **의미와 변화** |
        | --- | --- | --- | --- |
        | **`n_estimators`** | **100** | **500** | 나무를 5배 더 많이 심어 학습량을 대폭 늘렸습니다. |
        | **`max_depth`** | **6** | **9** | 나무의 깊이를 더 깊게 하여 더 복잡한 패턴을 학습합니다. |
        | **`learning_rate`** | **0.3** | **0.1** | 학습 속도를 늦추어 더 정교하게 최적의 지점을 찾습니다. |
        | **`gamma`** | **0** | **0.2** | 노드 분할을 위한 최소 손실 감소 값을 높여 과적합을 방지합니다. |
        | **`colsample_bytree`** | **1.0** | **0.6** | 나무 하나당 전체 피처의 60%만 사용하여 과적합을 막습니다. |
        | **`subsample`** | **1.0** | **1.0** | 데이터 샘플링 비율은 기본값 그대로 전체를 사용하셨네요. |
        - 최고 Macro F1 점수: 0.7023
- 04_xgb_balanced와 05_xgb_tuned 비교(마찬가지로 각 데이터 비교 및 해석 넣어야 해)
- 10, 11 test
    - 10, 11 csv를 전에 했던 것처럼 전처리한 후에 05_xgb_tuned 모델(학습된 가중치 불러옴)통해서 test해봄
    - 1~9에서의 05_xgb_tuned와 10,11 test에서의 05_xgb_tuned 결과 비교
    - Qupath에서 visualization해서 비교
        - 결측치, Uncertain은 None으로 처리
        - features_10_original vs. features_10_test
        - features_11_original vs. features_11_test