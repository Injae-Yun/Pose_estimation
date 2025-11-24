import h5py
import torch
import os
import glob
import random
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Dict

# utils.Preprocess에서 함수를 가져옵니다.
def get_keypoint_lists_from_config(config: dict) -> list[list]:
    """설정 사전에서 '_KEYPOINTS' 패턴을 가진 **모든** 키포인트 리스트를 찾습니다."""
    found_lists = []
    for key, value in config.items():
        if key.upper().endswith('_KEYPOINTS') and isinstance(value, list):
            found_lists.append(value)
    return found_lists # 리스트의 리스트 (e.g., [ [...], [...] ])

def detect_dataset_type(csv_header: set, all_configs: dict) -> str | None:
    """
    CSV 헤더와 모든 설정을 비교하여 가장 적합한 데이터셋 유형을 감지합니다.
    각 설정 내 여러 키포인트 리스트 중 가장 높은 점수를 기준으로 비교합니다.
    """
    best_match = None
    max_score = 0 # 전체 설정 중 최고 점수

    for config_name, config in all_configs.items():
        # config에서 가능한 모든 키포인트 리스트를 가져옵니다 (e.g., [HEADER_.., PROCESSED_..])
        keypoint_lists = get_keypoint_lists_from_config(config)
        
        if not keypoint_lists:
            continue
        
        config_max_score = 0 # 현 config 내의 리스트 중 최고 점수
        
        # 이 config에 포함된 여러 키포인트 리스트(e.g., HEADER, PROCESSED)를 순회
        for keypoints in keypoint_lists:
            # 헤더에 키포인트 이름이 얼마나 포함되는지 점수 계산
            score = sum(1 for kp in keypoints if any(kp in header_col for header_col in csv_header))
            
            # 이 리스트의 점수가 현 config의 최고 점수보다 높으면 갱신
            if score > config_max_score:
                config_max_score = score
        
        # 현 config의 최고 점수(config_max_score)가
        # 전체 최고 점수(max_score)보다 높으면 갱신
        if config_max_score > max_score:
            max_score = config_max_score
            best_match = config_name

    # 최소 2개 이상의 키포인트가 헤더와 일치해야 유효한 것으로 간주
    if max_score > 1:
        return best_match
    return None

def create_dataloader(h5_path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    H5 파일에서 데이터를 로드하여 PyTorch DataLoader를 생성합니다.
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            x_data = hf['x_data'][:]
            y_data = hf['y_data'][:]
    except (IOError, KeyError) as e:
        print(f"오류: H5 파일 '{h5_path}'을(를) 읽는 중 문제가 발생했습니다: {e}")
        raise

    if x_data.shape[0] == 0:
        # 데이터가 없는 경우 경고만 출력하고 빈 DataLoader를 반환합니다.
        print(f"경고: '{h5_path}'에 데이터가 없습니다. 빈 DataLoader를 반환합니다.")
        return DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size)

    X_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def create_scenario_dataloaders(
    config: Dict[str, any],
    model_name: str,
    train_scenarios: List[str],
    val_scenarios: List[str],
    test_scenarios: List[str],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    시나리오 기반으로 분할된 데이터셋에 대한 DataLoader를 생성합니다.
    모든 설정은 config 객체에서 가져옵니다.
    """
    data_paths = config['data_paths']
    raw_data_dir = data_paths['raw_dir']
    h5_dir = data_paths['h5_dir']
    
    training_params = config['models'][model_name]['training']
    batch_size = training_params['batch_size']
    
    os.makedirs(h5_dir, exist_ok=True)

    def get_dataset_for_scenarios(scenarios: List[str], set_name: str) -> TensorDataset:
        """주어진 시나리오 목록에 대해 데이터셋을 생성하거나 로드합니다."""
        h5_path = os.path.join(h5_dir, f'{config["dataset_type"]}_{model_name}_{set_name}_data.h5')
        
        all_files = []
        for scenario in scenarios:
            scenario_path = os.path.join(raw_data_dir, scenario)
            # 정규화된 CSV 파일은 Processed 디렉터리에 있습니다.
            processed_scenario_path = scenario_path.replace(raw_data_dir, data_paths['processed_dir'])
            files = glob.glob(os.path.join(processed_scenario_path, '**', '*.csv'), recursive=True)
            if not files:
                print(f"경고: '{processed_scenario_path}'에서 CSV 파일을 찾을 수 없습니다.")
            all_files.extend(files)

        if not all_files:
            print(f"경고: {set_name} 세트를 위한 CSV 파일이 없습니다. 빈 데이터셋을 생성합니다.")
            return TensorDataset(torch.empty(0), torch.empty(0))

        # 데이터 전처리 및 H5 파일로 저장
        preprocess_and_save(
            input_files=all_files,
            output_path=h5_path,
            config=config,
            model_name=model_name
        )

        # H5 파일에서 데이터 로드
        with h5py.File(h5_path, 'r') as hf:
            x_data = hf['x_data'][:] if 'x_data' in hf and hf['x_data'].shape[0] > 0 else np.array([])
            y_data = hf['y_data'][:] if 'y_data' in hf and hf['y_data'].shape[0] > 0 else np.array([])
        
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.long)
        
        return TensorDataset(x_tensor, y_tensor)

    # 각 세트에 대한 데이터셋 생성
    train_dataset = get_dataset_for_scenarios(train_scenarios, 'train')
    val_dataset = get_dataset_for_scenarios(val_scenarios, 'val')
    test_dataset = get_dataset_for_scenarios(test_scenarios, 'test')

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"데이터 로더 생성 완료:")
    print(f"  - 학습용: {len(train_dataset)} 샘플")
    print(f"  - 검증용: {len(val_dataset)} 샘플")
    print(f"  - 테스트용: {len(test_dataset)} 샘플")

    return train_loader, val_loader, test_loader

def split_scenarios(raw_data_dir: str, 
                    train_ratio: float = 0.75, 
                    val_ratio: float = 0.125, 
                    seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    주어진 디렉터리에서 시나리오 폴더를 찾아 학습, 검증, 테스트 세트로 분할합니다.
    """
    random.seed(seed)
    
    # 시나리오 폴더는 Raw 데이터 디렉터리에 있습니다.
    scenarios = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d)) and d.startswith('scenario')]
    if not scenarios:
        raise FileNotFoundError(f"'{raw_data_dir}'에서 'scenario'로 시작하는 폴더를 찾을 수 없습니다.")

    random.shuffle(scenarios)
    
    num_scenarios = len(scenarios)
    num_train = int(num_scenarios * train_ratio)
    num_val = int(num_scenarios * val_ratio)
    
    if num_scenarios == 8 and train_ratio == 0.75 and val_ratio == 0.125:
        num_train = 6
        num_val = 1

    num_test = num_scenarios - num_train - num_val
    if num_test < 1 and num_scenarios > num_train + num_val:
        num_test = 1
        if num_val > 1:
            num_val -=1
        else:
            num_train -=1

    train_scenarios = scenarios[:num_train]
    val_scenarios = scenarios[num_train:num_train + num_val]
    test_scenarios = scenarios[num_train + num_val:]

    print("--- 시나리오 분할 결과 ---")
    print(f"총 {num_scenarios}개의 시나리오")
    print(f"  - 학습용: {train_scenarios}")
    print(f"  - 검증용: {val_scenarios}")
    print(f"  - 테스트용: {test_scenarios}")
    print("-------------------------")

    return train_scenarios, val_scenarios, test_scenarios