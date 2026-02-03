import torch
import torch.nn as nn
from torchvision import models

class ParkingPilotNet(nn.Module):
    def __init__(self):
        super(ParkingPilotNet, self).__init__()
        
        # 1. 가벼운 백본 (MobileNetV3 Small)
        # pretrained=True를 쓰면 이미지넷의 특징(선, 면 인식)을 가져와서 학습이 빠릅니다.
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        
        # 2. 마지막 레이어 교체
        # MobileNetV3의 출력 특징 개수 가져오기
        num_features = self.backbone.classifier[0].in_features
        
        # 기존 분류기(Classifier) 제거하고 내 입맛에 맞는 헤드 장착
        self.backbone.classifier = nn.Identity() 
        
        # 3. 제어 헤드 (Regression)
        self.control_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # 과적합 방지
            nn.Linear(128, 2) # 출력: [Steering Angle, Throttle]
        )
        
    def forward(self, x):
        # x shape: (Batch, 3, 240, 320) -> 2x2 그리드 이미지
        features = self.backbone(x)
        output = self.control_head(features)
        return output

# 모델 테스트
if __name__ == "__main__":
    model = ParkingPilotNet()
    dummy_input = torch.randn(1, 3, 240, 320) # (B, C, H, W)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # [1, 2] 나오면 성공