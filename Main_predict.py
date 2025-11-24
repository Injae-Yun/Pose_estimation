"""
학습된 상태 분류 모델을 사용하여 새로운 포즈 데이터에 대한 예측을 수행합니다.

워크플로우:
1. 예측할 원본 데이터(CSV)를 지정합니다.
2. CSV 헤더를 분석하여 데이터 형식(COCO, Unreal 등)에 맞는 설정 파일을 동적으로 로드합니다.
3. 데이터를 정규화합니다.
4. 학습 시와 동일한 모델 구조를 생성하고, 저장된 가중치를 불러옵니다.
5. 정규화된 데이터를 모델 입력 형식에 맞게 시퀀스로 변환합니다.
6. 모델을 사용하여 각 시퀀스의 상태를 예측하고 결과를 출력합니다.
"""
import torch
import pandas as pd
import numpy as np
import os
import sys
import yaml
from typing import Any, Dict, List

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.State_classifier import PoseClassifierTransformer
from utils.Normalizer import PoseNormalizer
from utils.Preprocess import generate_sequences
from utils.config_loader import load_config


def get_config_by_csv_header(csv_path: str, config_dir: str = 'configs') -> Dict[str, Any]:
    """
    CSV 파일의 헤더를 분석하여 적절한 설정 파일을 결정하고 로드합니다.
    COCO 또는 Unreal 키포인트와의 일치율을 계산하여 가장 적합한 설정을 선택합니다.
    """
    try:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV 파일 '{csv_path}'을(를) 찾을 수 없습니다.")

    config_names = [os.path.splitext(f)[0] for f in os.listdir('configs') if f.endswith('.yaml') and f != 'models.yaml']        
    best_match = {'name': None, 'score': 0}

    for fname in config_names:
        main_config = load_config(fname)
        keypoints = main_config.get('HEADER_KEYPOINTS', [])
        if not keypoints:
            continue

        # 헤더에 설정 파일의 키포인트가 얼마나 포함되어 있는지 확인
        score = sum(1 for kp in keypoints if any(kp in col for col in header))
        if score > best_match['score']:
            best_match['name'] = fname.replace('.yaml', '')
            best_match['score'] = score
    main_config = load_config(best_match['name'])
    if not best_match['name']:
        raise ValueError("CSV 헤더와 일치하는 설정 파일을 찾을 수 없습니다.")

    print(f"감지된 데이터셋 유형: '{best_match['name']}' (일치 점수: {best_match['score']}) ")
    
    # 최종 설정 병합
    
    # 데이터셋 설정에 모델 정보가 없으면 models.yaml에서 가져와 병합
    if 'models' not in main_config:
        print("데이터셋 설정에 모델 정보가 없어 'models.yaml'에서 기본 모델 설정을 병합합니다.")
        model_config = load_config('models')
        main_config['models'] = model_config['models']

    # input_dim 동적 계산
    for model_name, model_cfg in main_config.get('models', {}).items():
        if 'data' in model_cfg and 'keypoint_indices' in model_cfg['data']:
            num_keypoints = len(model_cfg['data']['keypoint_indices'])
            model_cfg['architecture']['input_dim'] = num_keypoints * 2
            print(f"모델 '{model_name}'의 'input_dim'이 {num_keypoints * 2}(으)로 설정되었습니다.")

    return main_config

def run_prediction(PREDICT_RAW_CSV: str, MODEL_NAME: str, train_version: str = None):
    """
    학습된 모델로 상태 예측을 수행합니다.
    """
    if not os.path.exists(PREDICT_RAW_CSV):
        print(f"오류: 예측할 파일 '{PREDICT_RAW_CSV}'을(를) 찾을 수 없습니다.")
        return

    # --- 설정 로드 ---
    try:
        CONFIG = get_config_by_csv_header(PREDICT_RAW_CSV)
    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}")
        return

    model_cfg = CONFIG['models'][MODEL_NAME]
    paths_cfg = CONFIG['data_paths']
    state_map = CONFIG.get('STATE_MAP', {'stand': 0, 'sit': 1, 'lying': 2})
    if train_version:
        MODEL_PATH = os.path.join(paths_cfg['model_dir'], train_version, MODEL_NAME + '.pth')
        print(f"--- 버전 '{train_version}'의 모델을 사용합니다. ---")
    else:
        MODEL_PATH = os.path.join(paths_cfg['model_dir'], MODEL_NAME+ '.pth')
        print("--- 기본 모델을 사용합니다. ---")

    # --- 파이프라인 1: 예측할 데이터 정규화 ---
    print("="*50)
    print(f"--- 예측 데이터 정규화 시작: '{PREDICT_RAW_CSV}' ---")
    normalizer = PoseNormalizer(config=CONFIG)
    normalized_df = normalizer.process_csv_batch(PREDICT_RAW_CSV)
    print("--- 예측 데이터 정규화 완료 ---")

    # --- 파이프라인 2: 모델 로드 ---
    print("="*50)
    print(f"--- 모델 로드 시작: '{MODEL_PATH}' ---")
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 학습된 모델 파일 '{MODEL_PATH}'을(를) 찾을 수 없습니다.")
        return
    if train_version == 'v1_1':
        input_dim = model_cfg['architecture']['input_dim'] +2 # angle 2개 추가
    else:
        input_dim = model_cfg['architecture']['input_dim']

    model = PoseClassifierTransformer(
        input_dim=input_dim,
        d_model=model_cfg['architecture']['d_model'],
        nhead=model_cfg['architecture']['n_head'],
        num_encoder_layers=model_cfg['architecture']['num_layers'],
        dim_feedforward=model_cfg['architecture']['ffn_dim'],
        num_classes=model_cfg['architecture']['num_classes']
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("--- 모델 로드 완료 ---")

    # --- 파이프라인 3: 데이터 전처리 및 예측 ---
    print("="*50)
    print("--- 예측 실행 ---")
    
    sequences, _ = generate_sequences(
        normalized_df, 
        CONFIG,
        model_name=MODEL_NAME,
        train_version=train_version
    )
    if sequences.shape[0] == 0:
        print("예측을 위한 시퀀스를 생성할 수 없습니다. 데이터가 너무 짧습니다.")
        return

    X_predict = torch.tensor(sequences, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_predict)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)

    idx_to_state = {v: k for k, v in state_map.items()}
    predicted_labels = [idx_to_state[idx.item()] for idx in predicted_indices]
    
    result_df = pd.DataFrame()
    result_df['frame'] = np.arange(len(normalized_df)) + 1
    result_df['labeled_state'] = normalized_df['state'].reset_index(drop=True) if 'state' in normalized_df.columns else 'N/A'
    
    seq_length = model_cfg['data']['seq_length']
    prediction_start_index = seq_length - 1
    result_df['predict-state'] = pd.Series(dtype='object')
    end_index = prediction_start_index + len(predicted_labels)
    result_df.loc[prediction_start_index:end_index-1, 'predict-state'] = predicted_labels

    # 결과 저장
    os.makedirs(paths_cfg['results_dir'], exist_ok=True)
    result_csv_path = os.path.join(paths_cfg['results_dir'], f"predicted_{os.path.basename(PREDICT_RAW_CSV)}")
    result_df.to_csv(result_csv_path, index=False)
    print(f"정렬된 예측 결과를 '{result_csv_path}' 파일로 저장했습니다.")

    # 정확도 계산
    if 'state' in normalized_df.columns:
        valid_comparison_df = result_df.dropna(subset=['labeled_state', 'predict-state'])
        if not valid_comparison_df.empty:
            correct = (valid_comparison_df['labeled_state'] == valid_comparison_df['predict-state']).sum()
            total = len(valid_comparison_df)
            accuracy = (correct / total) * 100
            print(f"\n--- 예측 정확도: {accuracy:.2f}% (정답 {correct} / 전체 {total}) ---")

    print(f"\n총 {len(predicted_labels)}개의 시퀀스에 대한 예측이 완료되었습니다.")


if __name__ == '__main__':
    # --- 예측 실행 설정 ---
    # 예측할 원본 CSV 파일 경로. 이 파일의 헤더를 기반으로 설정을 자동으로 감지합니다.
    PREDICT_RAW_CSV = 'Test_data.csv' # 예: 'data/Raw/coco_sample.csv'
    #PREDICT_RAW_CSV = 'data/Processed/Kneeling_Down_3_modified.csv' # 예: 'data/Raw/coco_sample.csv'

    # 사용할 모델 버전 ('v1', 'v2' 등). None으로 설정 시 기본 경로의 모델 사용
    TRAIN_VERSION = 'v1_1' 
    
    # 불러올 모델 파일 이름
    MODEL_NAME = 'model1' #.pth 생략

    # --- 예측 실행 ---
    run_prediction(
        PREDICT_RAW_CSV=PREDICT_RAW_CSV, 
        MODEL_NAME=MODEL_NAME,
        train_version=TRAIN_VERSION
    )
