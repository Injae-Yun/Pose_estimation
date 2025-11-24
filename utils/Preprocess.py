"""
포즈 시퀀스 데이터 전처리 및 저장을 위한 스크립트.

이 스크립트는 정규화된 CSV 파일들을 읽어, 설정 파일에 정의된 내용을 바탕으로
학습에 사용할 시퀀스 데이터를 생성하고 HDF5(.h5) 파일로 저장합니다.
"""
import pandas as pd
import numpy as np
import h5py
import os
from utils.config_loader import load_config
from utils.Dataloader import detect_dataset_type

def get_keypoint_list_from_config(config: dict) -> list | None:
    """설정 사전에서 'HEADER_KEYPOINTS' 리스트를 찾습니다."""
    return config.get('HEADER_KEYPOINTS')

def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """연속적인 데이터에서 시퀀스를 생성합니다."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def generate_sequences(
    df: pd.DataFrame, config: dict, model_name: str, train_version: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    설정 파일을 기반으로 정규화된 DataFrame에서 시퀀스 데이터와 레이블을 생성합니다.
    """
    # 1. 설정에서 필요한 정보 추출
    model_config = config['models'][model_name]
    data_config = model_config['data']
    seq_length = data_config['seq_length']
    keypoint_indices = data_config['keypoint_indices']
    
    state_map = config['state_map']
    all_keypoints = get_keypoint_list_from_config(config)
    col_format = config['column_format']
    x_suffix = config['x_suffix']
    y_suffix = config['y_suffix']

    # 2. 사용할 관절 이름과 컬럼 이름 생성
    target_keypoint_names = [all_keypoints[i] for i in keypoint_indices]

    selected_columns = [
        col_format.format(name=name, suffix=suffix)
        for name in target_keypoint_names
        for suffix in [x_suffix, y_suffix]
    ]
    if train_version == 'v1_1':
        # angle_left_deg, angle_right_deg 추가
        selected_columns.extend(['angle_left_deg', 'angle_right_deg'])
        
    # 3. 'state' 컬럼 확인 및 레이블 배열 생성
    if 'state' not in df.columns:
        if 'Unnamed: 19' in df.columns:
            df = df.rename(columns={'Unnamed: 19': 'state'}) # 이상한 데이터가 있네요
        else:   
            raise ValueError("입력 DataFrame에 'state' 컬럼이 없습니다. 레이블을 생성할 수 없습니다.")
    if 'lay' in df['state'].values:
        df['state'][df['state']=='lay'] = 'lying'  # 'lay'를 'lying'으로 변경
    if 'stand ' in df['state'].values:
        df['state'][df['state']=='stand '] = 'stand' # 'stand '를 'stand'로 변경
        
    labels_array = df['state'].map(state_map).to_numpy()

    # 4. 주요 관절 선택 및 결측치 처리
    # reindex를 사용하여 순서 보장 및 없는 컬럼은 NaN으로 채움
    pose_data = df.reindex(columns=selected_columns).fillna(0.0)

    # 5. 시퀀스 및 해당 레이블 생성
    sequences = []
    labels = []
    for i in range(len(pose_data) - seq_length + 1):
        sequences.append(pose_data.values[i:i + seq_length])
        # 시퀀스의 마지막 프레임의 레이블을 해당 시퀀스의 레이블로 사용
        labels.append(labels_array[i + seq_length - 1])

    return np.array(sequences), np.array(labels, dtype=int)

def preprocess_and_save(
    input_files: list[str], output_path: str, config: dict, model_name: str, train_version: str = None
):
    """
    여러 CSV 파일을 처리하고, 시퀀스를 생성하여 H5 파일로 저장합니다.
    """
    try:
        config_names = [os.path.splitext(f)[0] for f in os.listdir('configs') if f.endswith('.yaml') and f != 'models.yaml']        
        all_configs = {name: load_config(name) for name in config_names}
    except FileNotFoundError as e:
        print(f"오류: 설정을 불러오는 데 실패했습니다: {e}")
        return

    all_sequences = []
    all_labels = []

    for csv_path in input_files:
        print(f"처리 중: '{csv_path}'")
        # 1. 이미 정규화된 CSV 파일 읽기
        normalized_df = pd.read_csv(csv_path, index_col=0)
        header = set(pd.read_csv(csv_path, nrows=0).columns)
        config_name = detect_dataset_type(header, all_configs)
        config = load_config(config_name)
        # 2. 시퀀스 및 레이블 생성
        sequences, labels = generate_sequences(normalized_df, config, model_name,train_version = train_version)
        if sequences.shape[0] > 0:
            all_sequences.append(sequences)
            all_labels.append(labels)

    if not all_sequences:
        print("생성된 시퀀스가 없습니다. 입력 파일이나 시퀀스 길이를 확인해주세요.")
        # 빈 H5파일이라도 생성하여 이후 과정에서 에러가 나지 않도록 함
        X_data, y_data = np.array([]), np.array([])
    else:
        X_data = np.concatenate(all_sequences, axis=0)
        y_data = np.concatenate(all_labels, axis=0)

    print(f"\n총 {len(input_files)}개 파일로부터 {X_data.shape[0]}개의 시퀀스 생성 완료.")
    print(f"X_data 형태: {X_data.shape}")
    print(f"y_data 형태: {y_data.shape}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('x_data', data=X_data)
        hf.create_dataset('y_data', data=y_data)

    print(f"\n데이터를 '{output_path}' 파일로 성공적으로 저장했습니다.")