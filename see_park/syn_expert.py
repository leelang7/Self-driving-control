import os
import time
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CSV_FILE = 'total_actions_path.csv' # 가지고 계신 파일명
IMG_ROOT = '.' 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15
VAL_RATIO = 0.2
SAVE_PATH = 'best_parking_model_synthetic.pth' # 파일명 변경

SUBSAMPLE_RATE = 1   # 데이터가 적으니 솎아내지 않고 다 씁니다 (중요!)
STEER_WEIGHT = 20.0  

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 데이터셋 클래스 (Synthetic Recovery 적용)
# ==========================================
class ParkingDataset(Dataset):
    def __init__(self, dataframe, root_dir, is_train=False):
        if is_train:
             # 데이터가 적을 땐 Subsampling 끄는게 낫습니다.
             self.data = dataframe.reset_index(drop=True)
        else:
             self.data = dataframe.reset_index(drop=True)
             
        self.root_dir = root_dir
        self.is_train = is_train
        
        # 색감 증강
        self.color_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_path = os.path.join(self.root_dir, str(row['path']))
        
        # 이미지 로드
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

        speed_val = float(row['linear_x'])
        steer_val = float(row['angular_z'])

        if self.is_train:
            # -----------------------------------------------------------
            # [기법 1] Random Flip (Bias 잡기 - 기존 유지)
            # -----------------------------------------------------------
            if np.random.rand() < 0.5:
                frames = [cv2.flip(f, 1) for f in frames]
                frames[2], frames[3] = frames[3], frames[2] # Swap LR
                steer_val = -steer_val

            # -----------------------------------------------------------
            # [기법 2] Random Shift (Std 잡기 - 신규 추가!)
            # -----------------------------------------------------------
            # 이미지를 가로로 살짝(-20 ~ +20 픽셀) 밉니다.
            # 이것이 "차가 경로를 이탈한 상황"을 시뮬레이션합니다.
            shift_x = np.random.randint(-20, 20)
            
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            frames = [cv2.warpAffine(f, M, (160, 120)) for f in frames]

            # [핵심] 밀린 만큼 핸들을 반대로 꺾어야 한다고 가르칩니다.
            # 0.004는 실험적인 보정 계수입니다. (픽셀당 0.004 rad 보정)
            # 이미지가 오른쪽(+)으로 밀림 -> 차가 왼쪽으로 감 -> 핸들을 오른쪽(-)으로 꺾어야 함
            # 이미지가 왼쪽(-)으로 밀림 -> 차가 오른쪽으로 감 -> 핸들을 왼쪽(+)으로 꺾어야 함
            steer_val -= (shift_x * 0.004) 

        # 스티칭 & 텐서 변환
        top = np.hstack((frames[0], frames[3]))
        bot = np.hstack((frames[2], frames[1]))
        grid = np.vstack((top, bot))
        
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0
        
        if self.is_train:
            image_tensor = self.color_transform(image_tensor)
            
        label = torch.FloatTensor([speed_val, steer_val])
        return image_tensor, label

# ==========================================
# 3. 모델 (동일)
# ==========================================
class ParkingPilotNet(nn.Module):
    def __init__(self):
        super(ParkingPilotNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.backbone(x)

# ==========================================
# 4. 학습 루프 (동일)
# ==========================================
def train():
    df = pd.read_csv(CSV_FILE)
    train_df, val_df = train_test_split(df, test_size=VAL_RATIO, random_state=42)
    
    # Subsampling 끄고 전체 데이터 사용 권장
    train_dataset = ParkingDataset(train_df, IMG_ROOT, is_train=True)
    val_dataset = ParkingDataset(val_df, IMG_ROOT, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Total Samples: {len(df)} | Train: {len(train_dataset)} (No Subsample)")
    
    model = ParkingPilotNet().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_cnt = 0
    
    train_losses, val_losses = [], []
    
    print("🚀 Training Started with Synthetic Recovery (Shift & Steer)...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss: Speed + (Steer * Weight)
            loss = criterion(outputs[:, 0], labels[:, 0]) + (criterion(outputs[:, 1], labels[:, 1]) * STEER_WEIGHT)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train = running_loss / len(train_loader)
        
        model.eval()
        run_val = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs[:, 0], labels[:, 0]) + (criterion(outputs[:, 1], labels[:, 1]) * STEER_WEIGHT)
                run_val += loss.item()
                
        avg_val = run_val / len(val_loader)
        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1}: Train {avg_train:.4f} | Val {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), SAVE_PATH)
            patience_cnt = 0
            print("  --> Model Saved!")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early Stopping!")
                break
                
    # 그래프 저장
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.savefig('loss_graph_synthetic.png')

if __name__ == "__main__":
    train()