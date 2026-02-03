import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 설정
# ==========================================
CSV_FILE = 'total_actions_path.csv'
IMG_ROOT = '.'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_A_PATH = 'best_parking_model3.pth'             # Baseline
MODEL_B_PATH = 'best_parking_model_synthetic.pth'   # Ours

# ==========================================
# 데이터셋 (강제 노이즈 주입용)
# ==========================================
class StressDataset(Dataset):
    def __init__(self, dataframe, root_dir, shift_amount=0):
        self.data = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.shift_amount = shift_amount # 강제로 밀어버릴 픽셀 수
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_path = os.path.join(self.root_dir, str(row['path']))
        img_paths = [
            os.path.join(base_path, str(row['front_cam'])),
            os.path.join(base_path, str(row['rear_cam'])),
            os.path.join(base_path, str(row['left_cam'])),
            os.path.join(base_path, str(row['right_cam']))
        ]
        frames = []
        for p in img_paths:
            img = cv2.imread(p)
            if img is None: img = np.zeros((120, 160, 3), dtype=np.uint8)
            else: img = cv2.resize(img, (160, 120))
            frames.append(img)

        # -----------------------------------------------------------
        # [Stress Test] 강제 Shift 적용
        # -----------------------------------------------------------
        if self.shift_amount != 0:
            M = np.float32([[1, 0, self.shift_amount], [0, 1, 0]])
            frames = [cv2.warpAffine(f, M, (160, 120)) for f in frames]
            
        # 정답 라벨 (Recovery 모델은 여기서 보정을 기대함)
        # 하지만 '원래 정답'과 비교해서 얼마나 잘 버티는지 보는 것이므로
        # 라벨은 원본 그대로 둡니다. (모델이 Shift를 보고 알아서 steer를 바꿔야 함)
        
        top = np.hstack((frames[0], frames[3]))
        bot = np.hstack((frames[2], frames[1]))
        grid = np.vstack((top, bot))
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0
        label = float(row['angular_z'])
        
        # Shift 된 상황에서 '이상적인 정답'은 보정된 값이어야 함
        # Baseline은 이걸 못하고, Ours는 이걸 해내야 함
        ideal_label = label - (self.shift_amount * 0.004) 
        
        return tensor, ideal_label

class ParkingPilotNet(nn.Module):
    def __init__(self):
        super(ParkingPilotNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2)
        )
    def forward(self, x): return self.backbone(x)

def eval_stress(model, df, shift):
    ds = StressDataset(df, IMG_ROOT, shift_amount=shift)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    errors = []
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            out = model(img)
            pred = out[:, 1]
            errors.extend((pred - lbl).abs().cpu().numpy())
    return np.mean(errors)

# ==========================================
# 메인 실행
# ==========================================
def main():
    df = pd.read_csv(CSV_FILE)
    
    # 모델 로드
    model_a = ParkingPilotNet().to(DEVICE)
    model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
    model_a.eval()
    
    model_b = ParkingPilotNet().to(DEVICE)
    model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
    model_b.eval()
    
    # Shift 강도별 테스트 (-30 픽셀 ~ +30 픽셀)
    shifts = [0, 5, 10, 15, 20, 25, 30]
    res_a = []
    res_b = []
    
    print("🔥 Stress Test Running...")
    for s in tqdm(shifts):
        # 양쪽 방향 평균 에러 측정 (좌우 대칭성 고려)
        err_a_pos = eval_stress(model_a, df, s)
        err_a_neg = eval_stress(model_a, df, -s)
        res_a.append((err_a_pos + err_a_neg) / 2)
        
        err_b_pos = eval_stress(model_b, df, s)
        err_b_neg = eval_stress(model_b, df, -s)
        res_b.append((err_b_pos + err_b_neg) / 2)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(shifts, res_a, 'o-', label='Baseline (Flip)', color='gray', linewidth=2)
    plt.plot(shifts, res_b, 'o-', label='Ours (Synthetic MPC)', color='red', linewidth=3)
    
    plt.title("Robustness Test: Performance under Camera Shift Noise", fontsize=14)
    plt.xlabel("Shift Intensity (Pixels)", fontsize=12)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("stress_test_result.png")
    print("\n✅ Stress Test 완료: stress_test_result.png")
    print(f"Noise 0px  -> A: {res_a[0]:.4f} / B: {res_b[0]:.4f}")
    print(f"Noise 30px -> A: {res_a[-1]:.4f} / B: {res_b[-1]:.4f}")

if __name__ == "__main__":
    main()