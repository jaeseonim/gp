import os
import pandas as pd
import numpy as np
import joblib
import config

def run_visualization():
    model_path = os.path.join(config.EXP_DIR, "05_xgb_tuned", "05_xgb_tuned_model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ 모델을 찾을 수 없습니다.")
        return

    model = joblib.load(model_path)

    # 학습 시 사용했던 17개 피처
    FEATURES_USED = [
        'Area µm^2', 'Length µm', 'Circularity', 'Solidity', 'Min diameter µm',
        'Hematoxylin: Mean', 'Hematoxylin: Min', 'Hematoxylin: Max', 'Hematoxylin: Std.Dev.',
        'DAB: Mean', 'DAB: Median', 'DAB: Min', 'DAB: Max', 'DAB: Std.Dev.',
        'AR (calc Max/Min)', 'Eccentricity (Feret-approx)', 'Relative_Pos_X'
    ]

    classes = ['Hypertrophic', 'Proliferative', 'Resting', 'Undetermined_1', 'Undetermined_2']
    target_files = ['features_10.csv', 'features_11.csv']
    raw_dir = "data/raw/test" 
    output_dir = "data/processed/visualize"
    os.makedirs(output_dir, exist_ok=True)

    for file_name in target_files:
        file_path = os.path.join(raw_dir, file_name)
        if not os.path.exists(file_path): continue

        original_df = pd.read_csv(file_path)
        X_input = original_df.copy()

        # 1. Relative_Pos_X 계산
        x_min, x_max = X_input['Centroid X µm'].min(), X_input['Centroid X µm'].max()
        X_input['Relative_Pos_X'] = (X_input['Centroid X µm'] - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0

        # 2. ⭐ 결측치 및 'Uncertain' 핸들링 로직
        # (A) 피처 중 하나라도 NaN이 있는 행
        mask_nan = X_input[FEATURES_USED].isnull().any(axis=1)
        
        # (B) 원본 Classification이 'Uncertain'인 행
        # 만약 원본에 해당 컬럼이 없다면 무시하도록 처리
        if 'Classification' in original_df.columns:
            mask_uncertain = (original_df['Classification'] == 'Uncertain')
        else:
            mask_uncertain = pd.Series([False] * len(original_df))

        # 제외할 최종 마스크 (결측치 OR Uncertain)
        mask_exclude = mask_nan | mask_uncertain
        
        # 전체를 일단 'None'으로 채운 리스트 생성
        final_predictions = ["None"] * len(original_df)
        
        # 3. 결측치가 없고 Uncertain도 아닌 '깨끗한' 행들만 골라서 예측 수행
        X_clean = X_input.loc[~mask_exclude, FEATURES_USED]
        
        if len(X_clean) > 0:
            print(f"🧪 {file_name}: {len(X_clean)}개 세포 예측 중... (제외: 결측치 {mask_nan.sum()}개, Uncertain {mask_uncertain.sum()}개)")
            y_pred = model.predict(X_clean)
            
            # 예측된 값들을 문자열 라벨로 변환
            pred_labels = [classes[i] for i in y_pred]
            
            # 예측 성공한 행의 인덱스에만 값을 채워넣음
            for idx, label in zip(X_clean.index, pred_labels):
                final_predictions[idx] = label
        
        # 4. 결과 저장
        visualize_df = original_df.copy()
        visualize_df['Predicted_Class'] = final_predictions
        
        save_path = os.path.join(output_dir, f"visualize_{file_name}")
        visualize_df.to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path} (총 {len(visualize_df)}행 유지)\n")

if __name__ == "__main__":
    run_visualization()