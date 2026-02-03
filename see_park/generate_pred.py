import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import os

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
INPUT_CSV = '/home/elicer/Coach/data/orig/total_actions_path.csv'
OUTPUT_CSV = 'driving_log_with_pred2.csv'
MODEL_PATH = '/home/elicer/Coach/data/orig/best_parking_model2.pth'

# [핵심 수정] 실제 데이터가 있는 최상위 폴더 (절대 경로 추천)
IMG_BASE_DIR = '/home/elicer/Coach/data/orig' 

Input_W, Input_H = 160, 120
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COL_IMGS = ['front_cam', 'rear_cam', 'left_cam', 'right_cam'] 
COL_TARGET_STEER = 'angular_z'

# ==========================================
# 2. 모델 클래스
# ==========================================
class ParkingPilotNet(nn.Module):
    def __init__(self):
        super(ParkingPilotNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 3. 전처리 함수 (경로 수정됨)
# ==========================================
def load_and_preprocess(row):
    frames = []
    
    # CSV의 'path' 컬럼에 에피소드 폴더명(episode_000)이 들어있다고 가정
    episode_dir = str(row['path']).strip()
    
    for col_name in COL_IMGS:
        file_name = str(row[col_name]).strip()
        
        # [수정된 경로 조립 로직]
        # /home/elicer/Coach/data/orig + episode_011 + front_cam/000000.jpg
        full_path = os.path.join(IMG_BASE_DIR, episode_dir, file_name)
        
        # 디버깅: 경로가 맞는지 확인하고 싶으면 주석 해제
        # print(f"Loading: {full_path}")

        frame = cv2.imread(full_path)
        
        # [안전장치] 이미지가 없으면 에러 발생시켜서 바로 알림
        if frame is None:
            raise FileNotFoundError(f"❌ 이미지 로드 실패! 경로를 확인하세요:\n{full_path}")
            
        frame = cv2.resize(frame, (Input_W, Input_H))
        frames.append(frame)

    # 스티칭 (2x2 Grid)
    top = np.hstack((frames[0], frames[3]))
    bot = np.hstack((frames[2], frames[1]))
    grid = np.vstack((top, bot)) 
    
    grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(grid_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    
    return tensor.to(DEVICE)

# ==========================================
# 4. 메인 실행
# ==========================================
def main():
    print(f"🔄 모델 로딩 중: {MODEL_PATH}")
    model = ParkingPilotNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    if not os.path.exists(INPUT_CSV):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {INPUT_CSV}")
        return
        
    df = pd.read_csv(INPUT_CSV)
    print(f"📂 데이터셋 로드 완료: {len(df)}개 샘플")

    predictions = []
    
    print("🚀 추론 시작...")
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                input_tensor = load_and_preprocess(row)
                output = model(input_tensor)
                
                pred_steer = output[0, 1].item()
                predictions.append(pred_steer)
                
            except FileNotFoundError as e:
                print(e)
                break # 경로 에러나면 즉시 중단
            except Exception as e:
                print(f"Error at index {index}: {e}")
                predictions.append(0.0)

    # 에러 없이 루프가 끝났을 때만 저장
    if len(predictions) == len(df):
        df['predicted_steering'] = predictions
        df.to_csv(OUTPUT_CSV, index=False)
        print("="*40)
        print(f"✅ 완료! 저장 경로: {OUTPUT_CSV}")
        print("이제 analyze_error.py를 다시 실행하세요.")
        print("="*40)
    else:
        print("❌ 중단됨: 모든 예측을 완료하지 못했습니다.")

if __name__ == "__main__":
    main()