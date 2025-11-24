
import pandas as pd
import numpy as np

class PoseNormalizer:
    """
    COCO-17 및 Unreal-6와 같은 다양한 데이터셋 유형을 처리하는 통합 포즈 정규화 클래스입니다.
    모든 작업에 서술적인 키포인트 이름을 사용합니다.
    """
    def __init__(self, config: dict):
        """
        특정 설정을 사용하여 Normalizer를 초기화합니다.

        Args:
            config (dict): 'dataset_type'과 키포인트 정보를 포함해야 하는 설정 사전입니다.
        """
        self.config = config
        self.dataset_type = config.get('dataset_type')

        if self.dataset_type == 'coco-17':
            # COCO에 대한 새 명명 규칙을 사용합니다.
            self.shoulder_l_x, self.shoulder_l_y = 'shoulder_l_x', 'shoulder_l_y'
            self.shoulder_r_x, self.shoulder_r_y = 'shoulder_r_x', 'shoulder_r_y'
            self.hip_l_x, self.hip_l_y = 'hip_l_x', 'hip_l_y'
            self.hip_r_x, self.hip_r_y = 'hip_r_x', 'hip_r_y'
            self.knee_l_x, self.knee_l_y = 'knee_l_x', 'knee_l_y'
            self.knee_r_x, self.knee_r_y = 'knee_r_x', 'knee_r_y'
            self.x_suffix = '_x'
            self.y_suffix = '_y'

        elif self.dataset_type == 'unreal-6':
            # Unreal의 경우, 이름이 더 설명적이므로 직접 사용합니다.
            self.shoulder_l_x, self.shoulder_l_y = 'clavicle_out_l_RelLoc_X', 'clavicle_out_l_RelLoc_Y'
            self.shoulder_r_x, self.shoulder_r_y = 'clavicle_out_r_RelLoc_X', 'clavicle_out_r_RelLoc_Y'
            self.hip_l_x, self.hip_l_y = 'thigh_out_l_RelLoc_X', 'thigh_out_l_RelLoc_Y'
            self.hip_r_x, self.hip_r_y = 'thigh_out_r_RelLoc_X', 'thigh_out_r_RelLoc_Y'
            self.knee_l_x, self.knee_l_y = 'calf_knee_l_RelLoc_X', 'calf_knee_l_RelLoc_Y'
            self.knee_r_x, self.knee_r_y = 'calf_knee_r_RelLoc_X', 'calf_knee_r_RelLoc_Y'
            self.x_suffix = '_X'
            self.y_suffix = '_Y'
        else:
            raise ValueError(f"알 수 없는 데이터셋 유형입니다: {self.dataset_type}")

    def _prepare_coco_columns(self, df):
        """COCO 데이터셋의 컬럼 이름을 'left_shoulder_x' -> 'shoulder_l_x' 형식으로 변경합니다."""
        rename_map = {}
        for col in df.columns:
            if col.startswith('left_'):
                parts = col.split('_') # ['left', 'shoulder', 'x']
                if len(parts) == 3:
                    new_name = f"{parts[1]}_l_{parts[2]}" # shoulder_l_x
                    rename_map[col] = new_name
            elif col.startswith('right_'):
                parts = col.split('_') # ['right', 'shoulder', 'x']
                if len(parts) == 3:
                    new_name = f"{parts[1]}_r_{parts[2]}" # shoulder_r_x
                    rename_map[col] = new_name
        
        df.rename(columns=rename_map, inplace=True)
        return df

    def _prepare_unreal_columns(self, df):
        """
        Unreal Engine 데이터 컬럼을 2D 형식에 맞게 준비합니다.
        - Z 좌표가 존재하면, Y를 버리고 Z를 Y로 사용합니다.
        - Z 좌표가 없으면 (이미 2D이면), 아무 작업도 하지 않습니다.
        """
        # Z 좌표 컬럼이 있는지 확인 (e.g., '..._RelLoc_Z')
        z_cols = [col for col in df.columns if col.endswith('_Z')]

        if z_cols:
            print("3D Unreal 데이터 감지됨. Y를 버리고 Z를 Y로 변환합니다.")
            
            # Y 컬럼 삭제
            y_cols_to_drop = [col for col in df.columns if col.endswith('_Y')]
            df.drop(columns=y_cols_to_drop, inplace=True)

            # Z 컬럼을 Y로 이름 변경
            rename_map = {col: col[:-2] + '_Y' for col in z_cols}
            df.rename(columns=rename_map, inplace=True)
        
        return df
    
    def _cal_angle(self,df): # for문 대신 벡터 연산 사용
        # --- 왼쪽 각도 ---
        # 1. 벡터 계산 (Series 연산)
        # v1: 무릎 -> 엉덩이
        v1_left_x = df[self.hip_l_x] - df[self.knee_l_x]
        v1_left_y = df[self.hip_l_y] - df[self.knee_l_y]
        
        # v2: 엉덩이 -> 어깨
        v2_left_x = df[self.shoulder_l_x] - df[self.hip_l_x]
        v2_left_y = df[self.shoulder_l_y] - df[self.hip_l_y]

        # 2. 벡터 크기 (Norm) 계산
        norm_v1_left = np.sqrt(v1_left_x**2 + v1_left_y**2)
        norm_v2_left = np.sqrt(v2_left_x**2 + v2_left_y**2)

        # 3. 내적(Dot product) 계산
        dot_prod_left = (v1_left_x * v2_left_x) + (v1_left_y * v2_left_y)
        # 4. 코사인 값 계산 (정규화된 내적)
        # 0으로 나누기 방지 (norm이 0인 경우)
        denominator = norm_v1_left * norm_v2_left
        cos_left = np.divide(dot_prod_left, denominator, 
                             out=np.zeros_like(dot_prod_left), where=denominator!=0)
        # --- 오른쪽 각도 (동일하게 반복) ---
        v1_right_x = df[self.hip_r_x] - df[self.knee_r_x]
        v1_right_y = df[self.hip_r_y] - df[self.knee_r_y]
        v2_right_x = df[self.shoulder_r_x] - df[self.hip_r_x]
        v2_right_y = df[self.shoulder_r_y] - df[self.hip_r_y]

        norm_v1_right = np.sqrt(v1_right_x**2 + v1_right_y**2)
        norm_v2_right = np.sqrt(v2_right_x**2 + v2_right_y**2)

        dot_prod_right = (v1_right_x * v2_right_x) + (v1_right_y * v2_right_y)
        
        denominator_r = norm_v1_right * norm_v2_right
        cos_right = np.divide(dot_prod_right, denominator_r, 
                              out=np.zeros_like(dot_prod_right), where=denominator_r!=0)

        # 각도 계산
        angle_left_rad = np.arccos(np.clip(cos_left, -1.0, 1.0))
        angle_right_rad = np.arccos(np.clip(cos_right, -1.0, 1.0))

        df["angle_left_deg"] = angle_left_rad / np.pi # pi 값(0~3.14) 정규화 > (0~1)
        df["angle_right_deg"] = angle_right_rad/ np.pi

        return df
    def _calculate_unit_length(self, data):
        """몸통과 다리 치수를 기반으로 단위 길이를 계산합니다."""
        mid_shoulder = (np.array([data.get(self.shoulder_l_x), data.get(self.shoulder_l_y)]) + 
                        [data.get(self.shoulder_r_x), data.get(self.shoulder_r_y)]) / 2
        mid_hip = (np.array([data.get(self.hip_l_x), data.get(self.hip_l_y)]) + 
                   np.array([data.get(self.hip_r_x), data.get(self.hip_r_y)])) / 2
        mid_knee = (np.array([data.get(self.knee_l_x), data.get(self.knee_l_y)]) + 
                    np.array([data.get(self.knee_r_x), data.get(self.knee_r_y)])) / 2

        dist_shoulder_hip = np.linalg.norm(mid_shoulder - mid_hip, axis=0)
        dist_hip_knee = np.linalg.norm(mid_hip - mid_knee, axis=0)
        
        unit_length = dist_shoulder_hip + dist_hip_knee
        return pd.Series(unit_length, index=data.index) if isinstance(data, pd.DataFrame) else unit_length

    def process_csv_batch(self, filepath: str) -> pd.DataFrame:
        """CSV 파일을 읽고 일괄 모드로 정규화합니다."""
        df = pd.read_csv(filepath, index_col=0)

        # --- 0단계: 데이터셋별 컬럼 준비 ---
        if self.dataset_type == 'coco-17':
            df = self._prepare_coco_columns(df)
        elif self.dataset_type == 'unreal-6':
            df = self._prepare_unreal_columns(df)

        # --- 1단계: 데이터 무결성 ---
        df.replace(0.0, np.nan, inplace=True)

        # --- 2단계: 단위 길이 계산 ---
        df['unit_length'] = self._calculate_unit_length(df)
        df['unit_length'] = df['unit_length'].clip(lower=1e-6)

        # --- 3단계: 좌표 정규화 ---
        mid_hip_x_col = f'mid_hip{self.x_suffix}'
        mid_hip_y_col = f'mid_hip{self.y_suffix}'
        
        df[mid_hip_x_col] = (df[self.hip_l_x] + df[self.hip_r_x]) / 2
        df[mid_hip_y_col] = (df[self.hip_l_y] + df[self.hip_r_y]) / 2

        coord_cols = [col for col in df.columns if col.endswith(self.x_suffix) or col.endswith(self.y_suffix)]
        
        for col in coord_cols:
            if col.endswith(self.x_suffix):
                df[col] = (df[col] - df[mid_hip_x_col]) / df['unit_length']
            elif col.endswith(self.y_suffix):
                df[col] = (df[col] - df[mid_hip_y_col]) / df['unit_length']

        # --- 4단계: Y축 반전 (coco의 경우) ---
        if self.dataset_type == 'coco-17':
             y_cols = [col for col in df.columns if col.endswith(self.y_suffix)]
             df[y_cols] *= -1

        df.drop(columns=['unit_length', mid_hip_x_col, mid_hip_y_col], inplace=True)

        # --- 5단계: angle 계산 ---
        df = self._cal_angle(df)
        return df

def simulate_live_from_csv(normalizer: PoseNormalizer, input_path: str, output_path: str):
    """
    이 함수는 아직 통합 Normalizer에 맞게 조정되지 않았습니다.
    """
    print("경고: 실시간 시뮬레이션은 아직 새로운 통합 Normalizer에 완전히 적용되지 않았습니다.")
    pass
