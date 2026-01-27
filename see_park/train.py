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
CSV_FILE = 'total_actions_path.csv'
IMG_ROOT = '.'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10
VAL_RATIO = 0.2
SAVE_PATH = 'best_parking_model.pth'
GRAPH_PATH = 'loss_graph.png'

# [NEW] 중요 설정 추가
SUBSAMPLE_RATE = 3   # 30fps 데이터를 10fps로 줄임 (3장 중 1장만 사용)
STEER_WEIGHT = 20.0  # 조향 오차에 20배 가중치 부여 (핸들 실수 용납 X)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. 데이터셋 클래스 (Augmentation & Grid)
# ==========================================
class ParkingDataset(Dataset):
    def __init__(self, dataframe, root_dir, is_train=False):
        # [NEW] 데이터 솎아내기 (Sub-sampling)
        # 30fps 데이터는 너무 중복이 심함 -> 3배수로 건너뛰어 학습 효율 증대
        if is_train:
             self.data = dataframe.iloc[::SUBSAMPLE_RATE].reset_index(drop=True)
        else:
             self.data = dataframe.reset_index(drop=True) # 검증은 꼼꼼하게 다 봄
             
        self.root_dir = root_dir
        self.is_train = is_train
        
        # 데이터 증강 (학습용)
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        base_path = os.path.join(self.root_dir, row['path'])
        img_paths = [
            os.path.join(base_path, row['front_cam']),
            os.path.join(base_path, row['rear_cam']),
            os.path.join(base_path, row['left_cam']),
            os.path.join(base_path, row['right_cam'])
        ]
        
        frames = []
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((120, 160, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (160, 120))
            frames.append(img)
            
        # 2x2 Grid Stitching
        top = np.hstack((frames[0], frames[3]))
        bot = np.hstack((frames[2], frames[1]))
        grid = np.vstack((top, bot))
        
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0
        
        if self.is_train:
            image_tensor = self.transform(image_tensor)
            
        # Label: [Speed, Steering]
        label = torch.FloatTensor([row['linear_x'], row['angular_z']])
        
        return image_tensor, label

# ==========================================
# 3. 모델 정의 (Transfer Learning)
# ==========================================
class ParkingPilotNet(nn.Module):
    def __init__(self):
        super(ParkingPilotNet, self).__init__()
        
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Backbone Freeze (눈 얼리기)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) # Output: [Speed, Steering]
        )
        
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 4. 유틸리티: Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss:
            self.counter += 1
            print(f'   [EarlyStopping] Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'   [Save] Val loss improved to {val_loss:.5f}. Saving model...')
        torch.save(model.state_dict(), self.path)

# ==========================================
# 5. 메인 학습 루프 (Loss 분리 및 가중치 적용)
# ==========================================
def train():
    df = pd.read_csv(CSV_FILE)
    train_df, val_df = train_test_split(df, test_size=VAL_RATIO, random_state=42)
    
    train_dataset = ParkingDataset(train_df, IMG_ROOT, is_train=True)
    val_dataset = ParkingDataset(val_df, IMG_ROOT, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Total Raw: {len(df)} | Train(Subsampled): {len(train_dataset)} | Val: {len(val_dataset)}")
    
    model = ParkingPilotNet().to(device)
    
    # [NEW] Loss 함수 분리 (각각 계산하기 위함)
    criterion_speed = nn.MSELoss()
    criterion_steer = nn.MSELoss()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE, path=SAVE_PATH)
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    print(f"Start Training with Steering Weight: {STEER_WEIGHT}")

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        running_loss = 0.0
        running_steer_loss = 0.0 # 로그용
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # [NEW] Loss 분리 계산 및 가중치 적용
            # Output: [Speed, Steering], Label: [Speed, Steering]
            loss_speed = criterion_speed(outputs[:, 0], labels[:, 0])
            loss_steer = criterion_steer(outputs[:, 1], labels[:, 1])
            
            # 조향에 가중치 곱하기
            total_loss = loss_speed + (loss_steer * STEER_WEIGHT)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_steer_loss += loss_steer.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_train_steer = running_steer_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Val ---
        model.eval()
        running_val_loss = 0.0
        val_steer_loss = 0.0
        val_speed_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss_speed = criterion_speed(outputs[:, 0], labels[:, 0])
                loss_steer = criterion_steer(outputs[:, 1], labels[:, 1])
                
                # 검증 Loss도 가중치 적용된 기준으로 평가 (그래야 Early Stopping이 조향 위주로 작동)
                total_loss = loss_speed + (loss_steer * STEER_WEIGHT)
                
                running_val_loss += total_loss.item()
                val_steer_loss += loss_steer.item()
                val_speed_loss += loss_speed.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_steer = val_steer_loss / len(val_loader)
        avg_val_speed = val_speed_loss / len(val_loader)
        
        val_losses.append(avg_val_loss)
        
        # [NEW] 상세 로그 출력 (속도 vs 조향 오차 확인 가능)
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Total Val Loss: {avg_val_loss:.4f} | "
              f"Steer Loss: {avg_val_steer:.4f} (Speed Loss: {avg_val_speed:.4f})")
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print(f"Training finished in {(time.time() - start_time)/60:.2f} mins")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Weighted Train Loss')
    plt.plot(val_losses, label='Weighted Validation Loss')
    plt.title(f'Training Loss (Steer Weight x{STEER_WEIGHT})')
    plt.xlabel('Epochs')
    plt.ylabel('Weighted MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(GRAPH_PATH)
    print(f"Loss graph saved to {GRAPH_PATH}")

if __name__ == "__main__":
    train()