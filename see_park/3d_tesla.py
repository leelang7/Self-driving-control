import cv2
import numpy as np
import torch
import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import time
import concurrent.futures
from ultralytics import YOLO
import json

# === [1] 모델 로드 ===
print("Loading Hybrid System (Depth + YOLO)...")

# Depth Model (지형/장애물 파악용)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()
if device.type == 'cuda': midas.half()
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# YOLO Model (객체 식별용) - 감도 높임
yolo_model = YOLO('yolov8n.pt') 

ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) # 스레드 2개

app = FastAPI()
global_frame = None
frame_lock = threading.Lock()
last_ai_time = 0

# === [2] 하이브리드 뷰어 (HTML) ===
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Tesla Hybrid FSD</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #000; font-family: sans-serif; }
        #hud { position: absolute; top: 10px; left: 10px; pointer-events: none; z-index: 20; }
        .box { border: 1px solid #333; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; }
        .title { color: #fff; font-size: 18px; font-weight: bold; margin-bottom: 5px; }
        .info { color: #00ffcc; font-size: 14px; }
        #cam-box {
            position: absolute; bottom: 20px; right: 20px; width: 260px; height: 195px;
            border: 2px solid #444; background: #111; z-index: 10; border-radius: 8px; overflow: hidden;
        }
        #real-cam { width: 100%; height: 100%; object-fit: cover; opacity: 1.0; }
        .label { color: yellow; background: rgba(0,0,0,0.5); padding: 2px 5px; border-radius: 3px; font-size: 12px; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/renderers/CSS2DRenderer.js"></script>
</head>
<body>
    <div id="hud">
        <div class="box">
            <div class="title">TESLA HYBRID NET</div>
            <div class="info">Voxels: Active (Obstacles)</div>
            <div class="info">YOLO: <span id="yolo-status">Scanning...</span></div>
        </div>
    </div>
    <div id="cam-box">
        <img id="real-cam" src="/video" onerror="this.src='/video?t='+new Date().getTime()" />
    </div>

    <script>
        // 1. Scene Setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505); 
        scene.fog = new THREE.Fog(0x050505, 30, 100);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(0, 60, 50);
        camera.lookAt(0, 0, -10);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0, -10);
        controls.enableDamping = true;

        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(20, 50, 20);
        scene.add(dirLight);

        // === [1] 복셀 그리드 (기본 장애물 표시) ===
        const gridSize = 64;
        const count = gridSize * gridSize;
        const geometry = new THREE.BoxGeometry(0.8, 1, 0.8);
        const material = new THREE.MeshStandardMaterial({ color: 0xdddddd });
        const mesh = new THREE.InstancedMesh(geometry, material, count);
        scene.add(mesh);
        
        const dummy = new THREE.Object3D();
        const color = new THREE.Color();
        let voxelHeights = new Uint8Array(count).fill(0);

        // === [2] YOLO 객체 박스 ===
        const boxGroup = new THREE.Group();
        scene.add(boxGroup);

        // === [3] 주행 경로 (초록색 매트) ===
        const pathGeo = new THREE.PlaneGeometry(10, 40);
        const pathMat = new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.1, side: THREE.DoubleSide });
        const path = new THREE.Mesh(pathGeo, pathMat);
        path.rotation.x = -Math.PI / 2;
        path.position.set(0, 0.1, -10); // 바닥에 깔기
        scene.add(path);

        // 내 차
        const carGroup = new THREE.Group();
        const carBody = new THREE.Mesh(new THREE.BoxGeometry(8, 4, 14), new THREE.MeshStandardMaterial({ color: 0x333333 }));
        carBody.position.y = 2;
        carGroup.add(carBody);
        carGroup.position.set(0,0,10);
        scene.add(carGroup);

        // Grid Helper
        const gridHelper = new THREE.GridHelper(100, 50, 0x222222, 0x111111);
        gridHelper.position.z = -20;
        scene.add(gridHelper);

        // WebSocket
        function connect() {
            const wsUrl = (location.protocol==='https:'?'wss:':'ws:') + '//' + location.host + '/ws/web';
            const ws = new WebSocket(wsUrl);
            ws.binaryType = "arraybuffer";

            ws.onmessage = (e) => {
                // 데이터 패킷 구조: [헤더(4byte) | 복셀데이터 | JSON텍스트]
                // 복잡하므로 두 개의 채널로 분리하지 않고,
                // 간단하게 JSON 안에 base64로 퉁치거나, 
                // 이번엔 '복셀'만 바이너리로 받고, YOLO는 텍스트로 받는건 렉걸림.
                // 해결책: 그냥 JSON으로 통일 (복셀 데이터 작음)
                
                try {
                    const msg = JSON.parse(e.data);
                    
                    // 1. 복셀 업데이트
                    if (msg.voxels) {
                        // base64 -> array
                        const bin = atob(msg.voxels);
                        for (let i=0; i<count; i++) {
                            voxelHeights[i] = bin.charCodeAt(i);
                        }
                        updateVoxels();
                    }

                    // 2. YOLO 박스 업데이트
                    updateBoxes(msg.objects);
                    
                } catch (err) { console.error(err); }
            };
            ws.onclose = () => setTimeout(connect, 1000);
        }
        connect();

        function updateVoxels() {
            let i = 0;
            for (let z = 0; z < gridSize; z++) {
                for (let x = 0; x < gridSize; x++) {
                    let val = voxelHeights[i];
                    if (val < 15) { // 노이즈 컷
                        dummy.position.set(0, -100, 0);
                        dummy.scale.set(0,0,0);
                    } else {
                        let posX = (x - gridSize/2) * 1.0;
                        let posZ = -(z) * 1.0;
                        let h = (val / 255.0) * 15.0;
                        
                        dummy.position.set(posX, h/2, posZ);
                        dummy.scale.set(1, h, 1);
                        dummy.updateMatrix();
                        
                        // 장애물은 회색/흰색 (테슬라풍)
                        let intensity = 0.5 + (val/510.0);
                        color.setRGB(intensity, intensity, intensity);
                        mesh.setColorAt(i, color);
                    }
                    mesh.setMatrixAt(i, dummy.matrix);
                    i++;
                }
            }
            mesh.instanceMatrix.needsUpdate = true;
            mesh.instanceColor.needsUpdate = true;
        }

        function updateBoxes(objects) {
            // 기존 박스 제거
            while(boxGroup.children.length > 0){ 
                boxGroup.remove(boxGroup.children[0]); 
            }
            
            if (!objects || objects.length === 0) {
                document.getElementById('yolo-status').innerText = "Clear";
                return;
            }
            document.getElementById('yolo-status').innerText = objects.length + " Detected";

            objects.forEach(obj => {
                // 3D 박스 생성
                const geometry = new THREE.BoxGeometry(obj.w, obj.h, obj.d);
                const material = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true });
                const cube = new THREE.Mesh(geometry, material);
                
                cube.position.set(obj.x, obj.h/2, obj.z);
                boxGroup.add(cube);
            });
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        setInterval(() => {
            const img = document.getElementById('real-cam');
            if(!img.complete || img.naturalHeight === 0) img.src = "/video?t=" + new Date().getTime();
        }, 3000);
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(content=html_content)

@app.websocket("/ws/pi")
async def pi_endpoint(websocket: WebSocket):
    await websocket.accept()
    global global_frame
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                with frame_lock: global_frame = img
    except: pass

# === 데이터 처리 로직 ===
def process_data(img_bgr):
    # --- 1. Depth (복셀) 생성 ---
    # 이전의 '깨끗한 바닥' 로직 사용
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)
    if device.type == 'cuda': input_batch = input_batch.half()
    
    with torch.no_grad():
        prediction = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(128, 128),
            mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # 바닥 깎기
    h, w = depth_map.shape
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-5)
    
    y_indices = np.linspace(0, 1, h)
    floor_gradient = np.tile(y_indices[:, np.newaxis] ** 0.5, (1, w))
    depth_clean = np.maximum(depth_norm - (floor_gradient * 0.8), 0)
    
    depth_uint8 = (depth_clean * 255).astype(np.uint8)

    # BEV 변환
    grid_res = 64
    src_pts = np.float32([[0, h], [w, h], [w*0.8, h*0.35], [w*0.2, h*0.35]])
    dst_pts = np.float32([[grid_res*0.3, grid_res], [grid_res*0.7, grid_res], [grid_res, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev = cv2.warpPerspective(depth_uint8, M, (grid_res, grid_res))
    
    # Erosion (노이즈 제거)
    bev = cv2.erode(bev, np.ones((3,3), np.uint8), iterations=1)
    bev = np.flipud(bev) # 상하반전
    
    # Base64 인코딩 (JSON 전송용)
    import base64
    voxels_b64 = base64.b64encode(bev.flatten()).decode('utf-8')

    # --- 2. YOLO (객체) 생성 ---
    # 컵, 병, 핸드폰 등 다양한 물체 인식 허용
    results = yolo_model(img_bgr, verbose=False, conf=0.15, classes=[0, 39, 41, 67]) 
    # conf=0.15: 인식 감도를 매우 높임 (작은 텀블러도 잡게)
    
    objects = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1+x2)/2, (y2) # 바닥 기준
            
            # 화면 좌표 -> 3D 좌표 대략 변환
            # 중앙에서 멀어질수록 X좌표 증가
            pos_x = (cx - img_bgr.shape[1]/2) * 0.05
            # 화면 아래일수록 가까움 (Z 증가 -> 0에 가까움)
            # 화면 위일수록 멈 (Z 감소)
            pos_z = - ((img_bgr.shape[0] - cy) * 0.15) 
            
            objects.append({
                "x": float(pos_x),
                "z": float(pos_z) - 5, # 약간 오프셋
                "w": 3, "h": 5, "d": 3 # 박스 크기
            })

    return json.dumps({"voxels": voxels_b64, "objects": objects})

@app.websocket("/ws/web")
async def web_endpoint(websocket: WebSocket):
    await websocket.accept()
    global last_ai_time
    loop = asyncio.get_running_loop()

    try:
        while True:
            img = None
            with frame_lock:
                if global_frame is not None: img = global_frame.copy()
            
            if img is None:
                await asyncio.sleep(0.05); continue

            if time.time() - last_ai_time > 0.1:
                img_small = cv2.resize(img, (640, 480))
                data = await loop.run_in_executor(ai_executor, process_data, img_small)
                await websocket.send_text(data)
                last_ai_time = time.time()

            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"[Web Error] {e}")

def gen_frames():
    while True:
        with frame_lock:
            if global_frame is None:
                blank = np.zeros((180, 240, 3), np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
            else:
                small_frame = cv2.resize(global_frame, (240, 180))
                _, buffer = cv2.imencode('.jpg', small_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, ws_ping_interval=None, ws_ping_timeout=None)