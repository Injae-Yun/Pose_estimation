"""
포즈 분석 및 분류 모델 학습을 위한 메인 파이프라인 스크립트.

이 스크립트는 여러 단계를 순차적으로 실행하여 모델 학습을 완료합니다.
1. 데이터 정규화 (Main_norm.py)
2. 상태 분류 모델 학습 (Main_1_state.py)
3. (미구현) 다음 단계
4. (미구현) 그 다음 단계
"""
import os
import sys

# 각 단계의 스크립트에서 실행 함수를 가져옵니다.
try:
    # 프로젝트 루트를 sys.path에 추가하여 다른 모듈을 찾을 수 있도록 합니다.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from script.Data_preprocess import run_batch_normalization
except ImportError as e:
    print(f"오류: 필요한 모듈을 임포트할 수 없습니다. ({e})")
    print("스크립트들이 올바른 위치에 있는지 확인해주세요.")
    sys.exit(1)


def run_main_2_placeholder():
    """Main_2.py에 대한 플레이스홀더 함수."""
    print("\n" + "="*50)
    print("--- 파이프라인 3: Main_2 (미구현) ---")
    print("이 단계는 아직 구현되지 않았습니다.")
    print("="*50)

def run_main_3_placeholder():
    """Main_3.py에 대한 플레이스홀더 함수."""
    print("\n" + "="*50)
    print("--- 파이프라인 4: Main_3 (미구현) ---")
    print("이 단계는 아직 구현되지 않았습니다.")
    print("="*50)


if __name__ == '__main__':
    # --- 파이프라인 설정 ---
    # 원본 데이터와 처리된 데이터 디렉터리
    RAW_DATA_DIR = os.path.join('data', 'Raw')
    PROCESSED_DATA_DIR = os.path.join('data', 'Processed')

    # --- 파이프라인 1: 데이터 정규화 ---
    # Raw 디렉터리의 모든 CSV 파일을 찾아 정규화한 후 Processed 디렉터리에 저장합니다.
    print("="*50)
    print("--- 파이프라인 1: 데이터 일괄 정규화 실행 ---")
    run_batch_normalization(raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR)


