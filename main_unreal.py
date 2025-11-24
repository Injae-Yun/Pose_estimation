import pandas as pd
import glob
import os
import numpy as np

def columns_to_drop(df):
    columns_to_drop = df.filter(regex='_Y$').columns

    # 2. 해당 열들을 DataFrame에서 삭제
    # axis=1: 열(column)을 삭제함을 지정
    # inplace=True: 원본 DataFrame을 바로 수정
    df.drop(columns=columns_to_drop, inplace=True) 
    # 또는 df = df.drop(columns=columns_to_drop)

    print("삭제된 열:", columns_to_drop.tolist())
    print(df.head())
    
    
def header_change(df):


    # 1. 변경할 헤더 딕셔너리 생성
    # (리스트 컴프리헨션으로 '기존 이름': '새 이름' 딕셔너리 자동 생성)
    rename_dict = {
        col: col.replace('_Z', '_Y') 
        for col in df.columns 
        if col.endswith('_Z')
    }

    # 결과 딕셔너리 예시: {'Name_Y': 'Name_Final', 'Score_Y': 'Score_Final'}
    print("변경할 헤더:", rename_dict)

    # 2. .rename() 함수로 열 이름 변경 적용
    # axis='columns' 또는 axis=1: 열(헤더)을 변경함을 지정
    # inplace=True: 원본 DataFrame을 바로 수정
    df.rename(columns=rename_dict, inplace=True)

    print("\n변경 후 헤더:", df.columns.tolist())
    print(df.head())





# IJ.py 파일
def unreal_process_csv_batch(df):
    """CSV 파일을 읽어 일괄적으로 정규화합니다."""
    #df = pd.read_csv(df, index_col=0)
    #df = self._normalize_unreal_column_names(df)

    # --- Step 1: 데이터 무결성 적용 ---
    df.replace(0.0, np.nan, inplace=True)

    # --- Step 2: 단위 길이 계산 ---
    df['unit_length'] = calculate_unit_length(df)
    df['unit_length'].fillna(method='ffill', inplace=True)
    df['unit_length'].fillna(method='bfill', inplace=True)
    df['unit_length'] = df['unit_length'].clip(lower=1e-6)

    # --- Step 3: 단위 길이 기반 좌표 정규화 ---
    coord_cols = [col for col in df.columns if '_X' in col or '_Y' in col]
    
    # COCO 기준: left_hip(11), right_hip(12)
    df['mid_hip_X'] = (df['thigh_out_l_RelLoc_X'] + df['thigh_out_r_RelLoc_X']) / 2
    df['mid_hip_Y'] = (df['thigh_out_l_RelLoc_Y'] + df['thigh_out_r_RelLoc_Y']) / 2
    #df['mid_hip_x'].fillna(method='ffill', inplace=True)
    #df['mid_hip_y'].fillna(method='ffill', inplace=True)
    #df['mid_hip_x'].fillna(method='bfill', inplace=True)
    #df['mid_hip_y'].fillna(method='bfill', inplace=True)

    for col in coord_cols:
        axis = col.split('_')[-1]
        df[col] = (df[col] - df[f'mid_hip_{axis}']) / df['unit_length']

    df.drop(columns=['unit_length', 'mid_hip_X', 'mid_hip_Y'], inplace=True)
    
    return df


# IJ.py 파일
def calculate_unit_length(data):
    """
    몸통 길이를 기반으로 단위 길이를 계산합니다.
    단위 길이 = (어깨 중앙-엉덩이 중앙 거리) + (엉덩이 중앙-무릎 중앙 거리)
    필요한 관절 좌표 중 하나라도 NaN이면, 계산 결과도 NaN이 됩니다.

    Args:
        data (pd.DataFrame or pd.Series): 키포인트 좌표 데이터.
            - DataFrame: 여러 프레임의 데이터.
            - Series: 단일 프레임의 데이터.

    Returns:
        pd.Series or float: 계산된 단위 길이. DataFrame 입력 시 Series, Series 입력 시 float 반환.
    """
    # COCO 기준: left_shoulder(5), right_shoulder(6), left_hip(11), right_hip(12), left_knee(13), right_knee(14)
    # 입력 데이터가 Series일 경우, .get()을 사용하여 안전하게 값에 접근
    mid_shoulder = (np.array([data.get('clavicle_out_l_RelLoc_X'), data.get('clavicle_out_l_RelLoc_Y')]) + np.array([data.get('clavicle_out_r_RelLoc_X'), data.get('clavicle_out_r_RelLoc_Y')])) / 2
    mid_hip = (np.array([data.get('thigh_out_l_RelLoc_X'), data.get('thigh_out_l_RelLoc_Y')]) + np.array([data.get('thigh_out_r_RelLoc_X'), data.get('thigh_out_r_RelLoc_Y')])) / 2
    mid_knee = (np.array([data.get('calf_knee_l_RelLoc_X'), data.get('calf_knee_l_RelLoc_Y')]) + np.array([data.get('calf_knee_r_RelLoc_X'), data.get('calf_knee_r_RelLoc_Y')])) / 2

    # 중앙점 간의 유클리드 거리 계산
    dist_shoulder_hip = np.linalg.norm(mid_shoulder - mid_hip, axis=0)
    dist_hip_knee = np.linalg.norm(mid_hip - mid_knee, axis=0)
    
    # 단위 길이 계산 (두 거리의 합)
    unit_length = dist_shoulder_hip + dist_hip_knee
    return pd.Series(unit_length, index=data.index) if isinstance(data, pd.DataFrame) else unit_length





def bone(df):
    angles_left = []
    angles_right = []

    for frame in range(len(df)):
        # joint 좌표 초기화 (3x2 배열로 수정)
        joint_left = np.zeros((3, 2))
        joint_right = np.zeros((3, 2))

        # 왼쪽 관절
        joint_left[0] = [df.loc[frame, "calf_knee_l_RelLoc_X"],
                        df.loc[frame, "calf_knee_l_RelLoc_Y"]]
        joint_left[1] = [df.loc[frame, "thigh_out_l_RelLoc_X"],
                        df.loc[frame, "thigh_out_l_RelLoc_Y"]]
        joint_left[2] = [df.loc[frame, "clavicle_out_l_RelLoc_X"],
                        df.loc[frame, "clavicle_out_l_RelLoc_Y"]]

        # 오른쪽 관절
        joint_right[0] = [df.loc[frame, "calf_knee_r_RelLoc_X"],
                        df.loc[frame, "calf_knee_r_RelLoc_Y"]]
        joint_right[1] = [df.loc[frame, "thigh_out_r_RelLoc_X"],
                        df.loc[frame, "thigh_out_r_RelLoc_Y"]]
        joint_right[2] = [df.loc[frame, "clavicle_out_r_RelLoc_X"],
                        df.loc[frame, "clavicle_out_r_RelLoc_Y"]]

        # 벡터 계산
        v1_left = joint_left[1] - joint_left[0]  # 무릎에서 허벅지 방향
        v2_left = joint_left[2] - joint_left[1]  # 허벅지에서 쇄골 방향
        
        v1_right = joint_right[1] - joint_right[0]
        v2_right = joint_right[2] - joint_right[1]

        # 벡터 정규화
        v1_left_norm = v1_left / np.linalg.norm(v1_left)
        v2_left_norm = v2_left / np.linalg.norm(v2_left)
        v1_right_norm = v1_right / np.linalg.norm(v1_right)
        v2_right_norm = v2_right / np.linalg.norm(v2_right)

        # 각도 계산
        cos_left = np.clip(np.dot(v1_left_norm, v2_left_norm), -1.0, 1.0)
        angle_left = np.degrees(np.arccos(cos_left))

        cos_right = np.clip(np.dot(v1_right_norm, v2_right_norm), -1.0, 1.0)
        angle_right = np.degrees(np.arccos(cos_right))

        angles_left.append(angle_left)
        angles_right.append(angle_right)

    df["angle_left_deg"] = angles_left
    df["angle_right_deg"] = angles_right

    return df


def main():
    file_list=glob.glob("data/*.csv")
    print(f"검색된 CSV 파일 목록: {file_list}")
    file_len=len(file_list)
    for i in range(file_len):
        print(f"\n처리 중인 파일 {i+1}/{file_len}: {file_list[i]}")
        df=pd.read_csv(file_list[i])
        columns_to_drop(df)
        header_change(df)
        unreal_process_csv_batch(df)
        bone(df)
        
        output_filename=os.path.splitext(os.path.basename(file_list[i]))[0]+'_modified.csv'
        df.to_csv(output_filename, index=False)
        print(f"수정된 파일이 저장되었습니다: {output_filename}")
    #df=pd.read_csv("data/Getting up_spine4.csv")
    
    #columns_to_drop(df)
    #header_change(df)
    #df.to_csv('Getting up.csv')
    
main()
    