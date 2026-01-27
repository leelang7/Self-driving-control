import cv2
import torch
import numpy as np
import time
from torchvision import models
import torch.nn as nn

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
MODEL_PATH = 'best_parking_model.pth'
CAMERA_INDEXES = [0, 2, 4, 6]  # [Front, Rear, Left, Right] 순서 확인 필수!
Input_W, Input_H = 160, 120    # 학습할 때 썼던 크기

# 라즈베리파이는 보통 CPU 추론 (혹은 가속기 사용 시 'cuda')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference Device: {DEVICE}")

# ==========================================
# 2. 모델 클래스 정의 (학습 코드와 100% 동일해야 함)
# ==========================================
class ParkingPilotNet(nn.Module):
    def __init__(self):
        super(ParkingPilotNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=None) # 추론 땐 weights 다운로드 불필요
        
        # 구조 맞추기
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
# 3. 전처리 함수 (2x2 Grid Stitching)
# ==========================================
def preprocess_frames(frames):
    resized = []
    for frame in frames:
        if frame is None: # 카메라 에러 시 검은 화면
            frame = np.zeros((Input_H, Input_W, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (Input_W, Input_H))
        resized.append(frame)
    
    # [학습 코드와 순서 일치 필수]
    # frames 순서: 0:Front, 1:Rear, 2:Left, 3:Right
    # Grid 구성:
    # [ Front ] [ Right ]
    # [ Left  ] [ Rear  ]
    top = np.hstack((resized[0], resized[3]))
    bot = np.hstack((resized[2], resized[1]))
    grid = np.vstack((top, bot)) # 320x240
    
    # 시각화용 원본 저장 (BGR 상태)
    vis_image = grid.copy()
    
    # 모델 입력용 변환 (BGR -> RGB -> Tensor)
    grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(grid_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0) # 배치 차원 추가 (1, 3, 240, 320)
    
    return tensor.to(DEVICE), vis_image

# ==========================================
# 4. 메인 실행 루프
# ==========================================
def main():
    # 모델 로드
    print("Loading AI Model...")
    model = ParkingPilotNet().to(DEVICE)
    # map_location='cpu'는 PC(GPU)에서 학습한 걸 라즈베리(CPU)에서 켤 때 필수
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # 평가 모드 (Dropout 끄기)
    print("Model Loaded!")

    # 카메라 열기
    caps = [cv2.VideoCapture(idx) for idx in CAMERA_INDEXES]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Warning: Camera {CAMERA_INDEXES[i]} failed to open.")
            
    try:
        while True:
            start_time = time.time()
            
            # 1. 4개 카메라 읽기
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                frames.append(frame if ret else None)
            
            # 2. 전처리 및 추론
            input_tensor, vis_image = preprocess_frames(frames)
            
            with torch.no_grad():
                output = model(input_tensor)
                
            # 결과 파싱
            speed = output[0, 0].item()
            steer = output[0, 1].item()
            
            # 3. 제어 명령 전송 (여기에 모터 제어 코드 연결)
            # send_motor_command(speed, steer) 
            print(f"Speed: {speed:.2f} | Steer: {steer:.2f}")

            # 4. 화면 출력 (디버깅용)
            # 화면에 값 띄우기
            cv2.putText(vis_image, f"SPD: {speed:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, f"STR: {steer:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 조향 시각화 (막대 그래프)
            center_x = 160
            bar_len = int(steer * 100) # 조향각에 비례
            cv2.line(vis_image, (center_x, 200), (center_x + bar_len, 200), (0, 255, 255), 5)
            
            cv2.imshow("Tesla Parking AI", vis_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 10Hz 제어 루프 유지 (옵션)
            # time.sleep(max(0, 0.1 - (time.time() - start_time)))

    except KeyboardInterrupt:
        print("Stopping...")
        
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()