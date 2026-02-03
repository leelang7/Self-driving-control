import cv2
import numpy as np
import torch
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
import threading

# === [1] 모델 로드 (MiDaS - Depth Estimation) ===
print("Loading AI Model...")
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# === [2] FastAPI 앱 설정 ===
app = FastAPI()

# 전역 변수 (최신 프레임 저장용)
global_frame = None
lock = threading.Lock()

# --- A. 웹페이지 (브라우저가 접속하는 곳) ---
@app.get("/")
async def get():
    # 간단한 HTML 페이지 반환
    html_content = """
    <html>
        <head>
            <title>Tesla-style Depth Vision</title>
            <style>
                body { background-color: #111; color: white; text-align: center; font-family: sans-serif; }
                h1 { margin-top: 20px; color: #00ffcc; }
                img { border: 2px solid #00ffcc; border-radius: 10px; width: 80%; max-width: 800px; }
                .status { color: #888; font-size: 0.9em; margin-bottom: 20px;}
            </style>
        </head>
        <body>
            <h1>Real-time Depth Perception</h1>
            <div class="status">Powered by FastAPI & MiDaS (GPU)</div>
            <img src="/video_feed" />
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- B. 라즈베리파이 연결 (웹소켓) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[System] Raspberry Pi Connected!")
    global global_frame
    
    try:
        while True:
            # 1. 파이에서 데이터 수신
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None: continue

            # 2. AI 추론 (Depth Map 생성)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # 3. 시각화 (히트맵 컬러 입히기 - Predator Vision 효과)
            depth_map = prediction.cpu().numpy()
            # 정규화 (0~255)
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
            depth_norm = (depth_norm * 255).astype(np.uint8)
            
            # 컬러맵 적용 (Plasma or Inferno)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
            # 원본과 나란히 붙이기 (선택사항)
            # combined = np.hstack((img, depth_color)) 
            
            # 4. 결과 저장 (전역 변수 업데이트)
            with lock:
                # JPEG로 인코딩해둠 (브라우저 전송용)
                _, buffer = cv2.imencode('.jpg', depth_color)
                global_frame = buffer.tobytes()

    except WebSocketDisconnect:
        print("[System] Raspberry Pi Disconnected")
    except Exception as e:
        print(f"[Error] {e}")

# --- C. 브라우저 영상 송출 (MJPEG 스트리밍) ---
def generate_frames():
    global global_frame
    while True:
        with lock:
            if global_frame is None:
                continue
            frame_data = global_frame
        
        # MJPEG 포맷으로 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # 포트 8002번 개방
    uvicorn.run(app, host="0.0.0.0", port=8002)