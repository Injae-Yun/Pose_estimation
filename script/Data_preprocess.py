
"""자동으로 데이터 유형을 감지하여 포즈를 정규화하는 스크립트.

이 스크립트는 Raw 데이터 디렉터리 내의 모든 CSV 파일을 순회하며, 각 파일의 헤더를 분석하여
configs 폴더에 정의된 설정(예: 'coco-17', 'unreal-6') 중 어떤 유형인지 자동으로 판단합니다.
그 후, 해당 유형에 맞는 정규화 프로세스를 적용하고 처리된 데이터를 별도의 디렉터리에 저장합니다.

- 설정 파일의 *KEYPOINTS 리스트를 사용하여 데이터 유형을 식별합니다.
- 'except'라는 이름의 폴더는 처리에서 제외됩니다.
- 알 수 없는 데이터 형식의 파일은 오류 메시지와 함께 건너뜁니다.
"""

import pandas as pd
import os
import argparse
import sys
import glob
from utils.Normalizer import PoseNormalizer
from utils.config_loader import load_config
from utils.Dataloader import detect_dataset_type


def run_batch_normalization(raw_dir: str, processed_dir: str, exclude_folder: str = 'except'):
    """
    raw_dir 안의 모든 CSV 파일을 유형에 맞게 자동 정규화하여 processed_dir에 저장합니다.
    """
    # 1. 모든 설정과 Normalizer를 미리 로드
    try:
        config_names = [os.path.splitext(f)[0] for f in os.listdir('configs') if f.endswith('.yaml') and f != 'models.yaml']        
        all_configs = {name: load_config(name) for name in config_names}
        normalizers = {name: PoseNormalizer(config) for name, config in all_configs.items()}
    except FileNotFoundError as e:
        print(f"오류: 설정을 불러오는 데 실패했습니다: {e}")
        return

    os.makedirs(processed_dir, exist_ok=True)

    # 2. 모든 CSV 파일 찾고 제외 폴더 필터링
    all_files = glob.glob(os.path.join(raw_dir, '**', '*.csv'), recursive=True)
    exclude_path_segment = os.path.sep + exclude_folder + os.path.sep
    raw_files = [f for f in all_files if exclude_path_segment not in f]

    if not raw_files:
        print(f"경고: '{raw_dir}'에서 처리할 CSV 파일을 찾지 못했습니다 ('{exclude_folder}' 폴더 제외).")
        return

    print(f"총 {len(raw_files)}개의 파일을 처리합니다.")

    for raw_path in raw_files:
        try:
            header = set(pd.read_csv(raw_path, nrows=0).columns)
            dataset_type = detect_dataset_type(header, all_configs)
            
            if not dataset_type:
                print(f"error: 부적절한 데이터 형식 감지되었습니다 ({os.path.basename(raw_path)})")
                continue

            normalizer = normalizers[dataset_type]
            
            relative_path = os.path.relpath(raw_path, raw_dir)
            processed_path = os.path.join(processed_dir, relative_path)
            
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            
            print(f"  - 처리 중: '{raw_path}' ({dataset_type}) -> '{processed_path}'")
            
            normalized_df = normalizer.process_csv_batch(raw_path)
            normalized_df.to_csv(processed_path)

        except Exception as e:
            print(f"오류: '{raw_path}' 파일 처리 중 문제 발생: {e}")
    
    print("일괄 정규화 완료.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="디렉터리 내의 모든 CSV 파일을 자동으로 감지하고 정규화하여 처리합니다.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/Raw',
        help="원본 CSV 파일이 포함된 디렉터리."
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='data/Processed',
        help="처리된 CSV 파일을 저장할 디렉터리."
    )
    parser.add_argument(
        '--exclude',
        type=str,
        default='except',
        help="처리에서 제외할 폴더 이름."
    )

    args = parser.parse_args()

    run_batch_normalization(args.raw_dir, args.processed_dir, args.exclude)
