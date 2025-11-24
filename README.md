# 포즈 기반 행동 분석 프로젝트

이 프로젝트는 2D 포즈 시퀀스 데이터를 기반으로 사람의 행동을 해석하는 파이프라인을 구축합니다. **상태 분류 → 행동 분류 → 행동 예측**의 3단계 구조를 목표로 하며, 현재는 1단계(상태 분류)가 구현되어 있습니다.

---

## 프로젝트 구조

```
Pose_estimation/
├── configs/                  # 설정 파일 디렉터리
│   ├── models.yaml           # 기본 모델 구조 정의
│   ├── coco-17.yaml          # COCO-17 데이터셋 관련 설정
│   └── unreal-6.yaml         # Unreal Engine 데이터셋 관련 설정
├── data/
│   ├── Raw/                  # 원본 데이터 (CSV)
│   └── Processed/            # 정규화된 데이터 (CSV)
├── models/                   # 학습된 모델이 저장되는 디렉터리
│   └── v1/                   # 예시: v1 버전으로 학습된 모델 폴더
│       └── model1.pth
├── core/
│   ├── State_classifier.py   # Transformer 모델 정의
│   └── Trainer.py            # 학습/검증/평가 함수
├── script/
│   ├── Data_preprocess.py    # 데이터 정규화 스크립트
│   └── Classification_state.py # 상태 분류 학습 스크립트
├── utils/
│   ├── Dataloader.py         # 데이터 로더 및 분할 함수
│   ├── Normalizer.py         # 포즈 정규화 클래스
│   └── Preprocess.py         # 시퀀스 생성 및 H5 저장
├── Main_normalization.py     # 데이터 정규화 실행 파일
├── Main_train.py             # 모델 학습 실행 파일
├── Main_predict.py           # 모델 예측 실행 파일
└── README.md
```

---

## 설정 (`configs` 디렉터리)

모든 프로젝트 설정은 `configs/` 디렉터리의 YAML 파일을 통해 관리됩니다.

-   **`models.yaml`**: `model1`과 같은 모델의 구조와 학습 하이퍼파라미터를 정의하는 기본 설정 파일입니다.
-   **`coco-17.yaml` / `unreal-6.yaml`**: 데이터셋별 설정을 포함합니다.
    -   `dataset_type`: 데이터셋의 종류를 명시합니다.
    -   `HEADER_KEYPOINTS`: 원본 CSV 파일에 포함될 것으로 예상되는 키포인트의 이름 목록입니다. `Main_predict.py`가 이 목록을 사용하여 데이터셋 종류를 자동으로 감지합니다.
    -   `column_format`, `x_suffix`, `y_suffix`: 원본 데이터의 좌표 컬럼 이름 규칙을 정의합니다.

---

## 사용 방법: 워크플로우

주요 워크플로우는 **정규화, 학습, 예측**의 세 단계로 구성됩니다.

### **0단계: 데이터 준비**

-   원본 포즈 데이터(CSV 형식)를 `data/Raw/` 디렉터리 안에 위치시킵니다.
-   학습용 데이터의 경우, 각 프레임의 상태를 나타내는 `state` 컬럼이 포함되어야 합니다.

### **1단계: 데이터 정규화 (`Main_normalization.py`)**

`data/Raw/`에 있는 모든 CSV 파일을 찾아 정규화를 수행하고, 그 결과를 `data/Processed/`에 저장합니다.

-   **목적**: 스케일링과 좌표 중앙화를 통해 일관된 형식의 학습 데이터를 생성합니다.
-   **실행 방법**:
    ```bash
    python Main_normalization.py
    ```
-   **사용자 파라미터**:
    -   `Main_normalization.py` 스크립트 하단의 `if __name__ == '__main__':` 블록에서 아래 변수를 수정할 수 있습니다.
    -   `RAW_DATA_DIR`: 원본 데이터 디렉터리 (기본값: `data/Raw`)
    -   `PROCESSED_DATA_DIR`: 정규화된 데이터가 저장될 디렉터리 (기본값: `data/Processed`)

### **2단계: 모델 학습 (`Main_train.py`)**

`data/Processed/`에 있는 정규화된 데이터를 사용하여 상태 분류 모델을 학습합니다.

-   **목적**: 포즈 시퀀스로부터 'stand', 'sit', 'lying'과 같은 상태를 분류하는 모델을 학습시킵니다.
-   **실행 방법**:
    ```bash
    python Main_train.py
    ```
-   **사용자 파라미터**:
    -   실행 전, `Main_train.py`의 `if __name__ == '__main__':` 블록을 반드시 수정해야 합니다.
    -   `TRAIN_VERSION`: 학습 버전에 대한 이름(예: `'v1'`, `'v2_new_data'`)을 지정합니다. 이 이름으로 `models/` 디렉터리 하위에 폴더가 생성되고, 학습된 모델(`model1.pth`)이 저장됩니다.

### **3단계: 예측 (`Main_predict.py`)**

학습된 모델을 사용하여 새로운 데이터에 대한 추론을 수행합니다. 이 스크립트는 CSV 헤더를 분석하여 데이터 형식을 자동으로 감지합니다.

-   **목적**: 원본 CSV 파일의 각 프레임에 대한 상태를 예측합니다.
-   **실행 방법**:
    ```bash
    python Main_predict.py
    ```
-   **사용자 파라미터**:
    -   실행 전, `Main_predict.py`의 `if __name__ == '__main__':` 블록을 수정해야 합니다.
    -   `PREDICT_RAW_CSV`: 예측을 수행할 원본 CSV 파일의 경로(예: `'data/Raw/my_test_data.csv'`)를 지정합니다.
    -   `TRAIN_VERSION`: 사용할 학습된 모델의 버전(예: `'v1'`)을 지정합니다. 이 이름은 학습 단계에서 설정한 버전명과 일치해야 합니다.
    -   `MODEL_NAME`: 불러올 모델 파일의 이름(예: `'model1.pth'`)을 지정합니다.

---

## 요구 사항

-   Python 3.x
-   PyTorch
-   pandas, numpy, h5py, pyyaml

아래 명령어를 통해 필요한 라이브러리를 설치할 수 있습니다.
```bash
pip install torch pandas numpy h5py pyyaml
```