"""(v1) 단일 데이터셋 기반 상태 분류 모델 학습 스크립트."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob

from core.State_classifier import PoseClassifierTransformer
from core.Trainer import train_model

from utils.config_loader import load_config
from utils.Preprocess import preprocess_and_save
from utils.Dataloader import create_dataloader

def run_state_classification_training(config_name: str = 'models', train_version: str = 'v1'):
    """
    지정된 설정을 사용하여 데이터 전처리부터 모델 학습까지 전체 파이프라인을 실행합니다.
    """
    # --- 설정 로드 ---
    config = load_config(config_name)
    model_name = 'model1'
    model_cfg = config['models']['model1']
    paths_cfg = config['data_paths']
    processed_data_dir = paths_cfg['processed_dir']
    
    MODEL_SAVE_PATH = os.path.join(paths_cfg['model_dir'],train_version, f'{model_name}.pth')

    # 모델 인스턴스 생성
    model = PoseClassifierTransformer(
        input_dim=len(model_cfg['data']['keypoint_indices']) * 2,
        d_model=model_cfg['architecture']['d_model'],
        nhead=model_cfg['architecture']['n_head'],
        num_encoder_layers=model_cfg['architecture']['num_layers'],
        dim_feedforward=model_cfg['architecture']['ffn_dim'],
        num_classes=model_cfg['architecture']['num_classes']
    )

    print("--- 모델 구조 ---")
    print(model)
    print("\n" + "="*50 + "\n")

    # --- 파이프라인 1: 정규화된 데이터로 시퀀스 생성 및 H5 저장 ---
    print("--- 데이터 전처리 시작 ---")
    
    files_to_process = glob.glob(os.path.join(processed_data_dir, '**', '*.csv'), recursive=True)
    
    if not files_to_process:
        print(f"오류: '{processed_data_dir}'에 처리된 CSV 파일이 없습니다.")
        return

    H5_FILE_PATH = os.path.join(paths_cfg['h5_dir'], f'{model_name}_train_data.h5')

    preprocess_and_save(
        input_files=files_to_process,
        output_path=H5_FILE_PATH,
        config=config,
        model_name=model_name,
        train_version=train_version
    )
    print("--- 데이터 전처리 완료 ---\
")

    # --- 파이프라인 2: 학습 데이터 로드 ---
    print("--- 학습 데이터 로드 시작 ---")
    try:
        train_loader = create_dataloader(H5_FILE_PATH, model_cfg['training']['batch_size'])
        print(f"데이터 로드 완료: {len(train_loader.dataset)}개의 샘플")
    except (IOError, KeyError, ValueError) as e:
        print(f"데이터 로딩 실패: {e}")
        return
    print("--- 학습 데이터 로드 완료---\
")

    # --- 파이프라인 3: 모델 학습 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_cfg['training']['learning_rate'])

    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=model_cfg['training']['num_epochs'],
        model_save_path=MODEL_SAVE_PATH
    )

def run_state_classification_training_v1_1(config_name: str = 'models', train_version: str = 'v1'):
    """
    angle을 추가로 받습니다.
    """
    # --- 설정 로드 ---
    config = load_config(config_name)
    model_name = 'model1'
    model_cfg = config['models']['model1']
    paths_cfg = config['data_paths']
    processed_data_dir = paths_cfg['processed_dir']
    
    MODEL_SAVE_PATH = os.path.join(paths_cfg['model_dir'],train_version, f'{model_name}.pth')

    # 모델 인스턴스 생성
    model = PoseClassifierTransformer(
        input_dim=len(model_cfg['data']['keypoint_indices']) * 2 + 2, # angle 2개 추가
        d_model=model_cfg['architecture']['d_model'],
        nhead=model_cfg['architecture']['n_head'],
        num_encoder_layers=model_cfg['architecture']['num_layers'],
        dim_feedforward=model_cfg['architecture']['ffn_dim'],
        num_classes=model_cfg['architecture']['num_classes']
    )

    print("--- 모델 구조 ---")
    print(model)
    print("\n" + "="*50 + "\n")

    # --- 파이프라인 1: 정규화된 데이터로 시퀀스 생성 및 H5 저장 ---
    print("--- 데이터 전처리 시작 ---")
    
    files_to_process = glob.glob(os.path.join(processed_data_dir, '**', '*.csv'), recursive=True)
    
    if not files_to_process:
        print(f"오류: '{processed_data_dir}'에 처리된 CSV 파일이 없습니다.")
        return

    H5_FILE_PATH = os.path.join(paths_cfg['h5_dir'], f'{model_name}_train_data.h5')

    preprocess_and_save(
        input_files=files_to_process,
        output_path=H5_FILE_PATH,
        config=config,
        model_name=model_name,
        train_version=train_version
    )
    print("--- 데이터 전처리 완료 ---\
")

    # --- 파이프라인 2: 학습 데이터 로드 ---
    print("--- 학습 데이터 로드 시작 ---")
    try:
        train_loader = create_dataloader(H5_FILE_PATH, model_cfg['training']['batch_size'])
        print(f"데이터 로드 완료: {len(train_loader.dataset)}개의 샘플")
    except (IOError, KeyError, ValueError) as e:
        print(f"데이터 로딩 실패: {e}")
        return
    print("--- 학습 데이터 로드 완료---\
")

    # --- 파이프라인 3: 모델 학습 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_cfg['training']['learning_rate'])

    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=model_cfg['training']['num_epochs'],
        model_save_path=MODEL_SAVE_PATH
    )