import cv2
import numpy as np

def make_grid_input(frames):
    """
    frames: [front, rear, left, right] 순서의 이미지 리스트 (numpy array)
    return: 2x2로 합쳐진 하나의 이미지
    """
    if len(frames) != 4:
        return None
        
    # 1. 모든 이미지를 동일한 작은 크기로 리사이즈 (RPi 부하 감소)
    # 160x120 정도면 주차 라인 식별에 충분합니다.
    target_w, target_h = 160, 120
    resized_frames = []
    
    for frame in frames:
        if frame is None: # 카메라 오류 시 검은 화면 대체
            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (target_w, target_h))
        resized_frames.append(frame)
    
    # 2. 2x2 그리드로 배치
    # [ Front ] [ Right ]
    # [ Left  ] [ Rear  ] 
    # 배치 순서는 학습할 때만 고정하면 됩니다.
    
    top_row = np.hstack((resized_frames[0], resized_frames[3])) # Front + Right
    bot_row = np.hstack((resized_frames[2], resized_frames[1])) # Left + Rear
    
    grid_image = np.vstack((top_row, bot_row))
    
    # 결과 이미지는 320x240 크기가 됨
    return grid_image