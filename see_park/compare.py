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
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. 설정 (경로 확인 필수)
# ==========================================
CSV_FILE = 'total_actions_path.csv'
IMG_ROOT = '.'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 왼쪽: 기존 (Flip Only)
MODEL_A_PATH = 'best_parking_model3.pth' 
# 오른쪽: 제안 (Synthetic MPC)
MODEL_B_PATH = 'best_parking_model_synthetic.pth' 

# ==========================================
# 2. 데이터셋 & 모델 (평가용)
# ==========================================
class EvalDataset(Dataset):
    def __init__(self, dataframe, root_dir):
        self.data = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        
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
            
        top = np.hstack((frames[0], frames[3]))
        bot = np.hstack((frames[2], frames[1]))
        grid = np.vstack((top, bot))
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0
        label = float(row['angular_z'])
        return tensor, label

class ParkingPilotNet(nn.Module):
    def __init__(self):
        super(ParkingPilotNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2)
        )
    def forward(self, x): return self.backbone(x)

def get_predictions(model_path, loader):
    model = ParkingPilotNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except:
        print(f"❌ 모델 없음: {model_path}")
        return None, None
    
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Loading {model_path}"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds.extend(outputs[:, 1].cpu().numpy()) # Steering
            truths.extend(labels.numpy())
    return np.array(preds), np.array(truths)

# ==========================================
# 3. 메인 분석 및 시각화
# ==========================================
def main():
    # 데이터 로드
    df = pd.read_csv(CSV_FILE)
    dataset = EvalDataset(df, IMG_ROOT)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 예측 수행
    pred_a, true_a = get_predictions(MODEL_A_PATH, loader) # Left
    pred_b, true_b = get_predictions(MODEL_B_PATH, loader) # Right
    
    if pred_a is None or pred_b is None: return

    # DataFrame 생성 (시각화용)
    df_a = pd.DataFrame({'Prediction': pred_a, 'Truth': true_a, 'Model': 'Baseline (Flip Only)'})
    df_a['Error'] = df_a['Prediction'] - df_a['Truth']
    # [수정] 개별 DF에도 Abs_Error 미리 계산
    df_a['Abs_Error'] = df_a['Error'].abs()
    
    df_b = pd.DataFrame({'Prediction': pred_b, 'Truth': true_b, 'Model': 'Ours (Synthetic MPC)'})
    df_b['Error'] = df_b['Prediction'] - df_b['Truth']
    # [수정] 개별 DF에도 Abs_Error 미리 계산
    df_b['Abs_Error'] = df_b['Error'].abs()
    
    combined_df = pd.concat([df_a, df_b])

    # -------------------------------------------------------
    # 🎨 [Visualization] 대시보드 그리기
    # -------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3) # 2행 3열 레이아웃

    # 1. Error Distribution (KDE Plot) - 오차 분포 곡선
    ax1 = fig.add_subplot(gs[0, :]) # 첫 줄 전체 사용
    sns.kdeplot(data=combined_df, x='Error', hue='Model', fill=True, common_norm=False, palette=['grey', 'red'], alpha=0.3, linewidth=2, ax=ax1)
    ax1.set_title('Step 1. Error Distribution Density (Low Variance is Better)', fontsize=14, fontweight='bold')
    ax1.set_xlim(-1.5, 1.5)
    ax1.axvline(0, color='black', linestyle='--')
    ax1.text(0.5, 0.8, "Target: Tall & Narrow Peak", transform=ax1.transAxes, color='red', fontsize=12)

    # 2. Scatter Plot (Baseline) - 산점도
    ax2 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(data=df_a, x='Truth', y='Prediction', alpha=0.3, color='grey', ax=ax2)
    ax2.plot([-1, 1], [-1, 1], 'k--', linewidth=1.5) # 정답선
    ax2.set_title(f'Baseline: Flip Only\n(R2 Score: {r2_score(true_a, pred_a):.3f})', fontsize=12)
    ax2.set_xlim(-1.0, 1.0); ax2.set_ylim(-1.0, 1.0)

    # 3. Scatter Plot (Ours) - 산점도
    ax3 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(data=df_b, x='Truth', y='Prediction', alpha=0.3, color='red', ax=ax3)
    ax3.plot([-1, 1], [-1, 1], 'k--', linewidth=1.5) # 정답선
    ax3.set_title(f'Ours: Synthetic MPC\n(R2 Score: {r2_score(true_b, pred_b):.3f})', fontsize=12, fontweight='bold', color='red')
    ax3.set_xlim(-1.0, 1.0); ax3.set_ylim(-1.0, 1.0)

    # 4. Box Plot (Absolute Error) - 절대 오차 비교
    ax4 = fig.add_subplot(gs[1, 2])
    # combined_df에는 이미 합쳐져 있으므로 그대로 사용 가능
    sns.boxplot(data=combined_df, x='Model', y='Abs_Error', palette=['grey', 'red'], ax=ax4)
    ax4.set_title('Error Magnitude Comparison', fontsize=12)
    ax4.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig('rich_comparison_dashboard.png', dpi=300)
    print("✅ 분석 완료: rich_comparison_dashboard.png 저장됨")

    # -------------------------------------------------------
    # 📊 [Metrics] 수치 출력
    # -------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Baseline (Flip)':<20} | {'Ours (Synthetic MPC)':<20}")
    print("-" * 65)
    
    metrics = {
        'Bias (Mean Error)': (df_a['Error'].mean(), df_b['Error'].mean()),
        'Std (Stability)':   (df_a['Error'].std(), df_b['Error'].std()),
        'RMSE (Total Err)':  (np.sqrt(mean_squared_error(true_a, pred_a)), np.sqrt(mean_squared_error(true_b, pred_b))),
        # [수정] 이제 에러 안 남
        'Max Error (Safety)':(df_a['Abs_Error'].max(), df_b['Abs_Error'].max())
    }
    
    for k, v in metrics.items():
        print(f"{k:<20} | {v[0]:.4f}               | {v[1]:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()