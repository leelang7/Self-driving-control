# server.py (전체 수정 버전)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# --- 연결 관리자 (Connection Manager) ---
class ConnectionManager:
    def __init__(self):
        self.robot_ws: WebSocket = None
        self.user_ws: WebSocket = None

    async def connect_robot(self, websocket: WebSocket):
        await websocket.accept()
        self.robot_ws = websocket
        print("🤖 로봇이 연결되었습니다.")

    async def connect_user(self, websocket: WebSocket):
        await websocket.accept()
        self.user_ws = websocket
        print("👤 사용자가 연결되었습니다.")

    def disconnect_robot(self):
        self.robot_ws = None
        print("🤖 로봇 연결 끊김")

    def disconnect_user(self):
        self.user_ws = None
        print("👤 사용자 연결 끊김")

    # 사용자가 보낸 명령 -> 로봇에게 전달
    async def send_command_to_robot(self, command: str):
        if self.robot_ws:
            try:
                await self.robot_ws.send_text(command)
            except Exception as e:
                print(f"명령 전달 실패: {e}")

    # 로봇이 보낸 영상 -> 사용자에게 전달
    async def send_video_to_user(self, data: bytes):
        if self.user_ws:
            try:
                await self.user_ws.send_bytes(data)
            except Exception as e:
                pass # 사용자 연결 불안정 시 무시

manager = ConnectionManager()

# --- 1. 로봇 접속 엔드포인트 ---
@app.websocket("/ws/robot")
async def robot_endpoint(websocket: WebSocket):
    await manager.connect_robot(websocket)
    try:
        while True:
            # 로봇에게서 영상 데이터를 받음
            data = await websocket.receive_bytes()
            # 사용자에게 중계
            await manager.send_video_to_user(data)
    except WebSocketDisconnect:
        manager.disconnect_robot()
    except Exception as e:
        print(f"로봇 통신 에러: {e}")
        manager.disconnect_robot()

# --- 2. 사용자(웹) 접속 엔드포인트 ---
@app.websocket("/ws/user")
async def user_endpoint(websocket: WebSocket):
    await manager.connect_user(websocket)
    try:
        while True:
            # 사용자에게서 명령(text)을 받음
            command = await websocket.receive_text()
            # 로봇에게 전달
            await manager.send_command_to_robot(command)
    except WebSocketDisconnect:
        manager.disconnect_user()

# server_brain.py (UI 업데이트 버전)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# --- 연결 관리자 (이전과 동일) ---
class ConnectionManager:
    def __init__(self):
        self.robot_ws: WebSocket = None
        self.user_ws: WebSocket = None

    async def connect_robot(self, websocket: WebSocket):
        await websocket.accept()
        self.robot_ws = websocket
        print("🤖 로봇 연결됨")

    async def connect_user(self, websocket: WebSocket):
        await websocket.accept()
        self.user_ws = websocket
        print("👤 사용자 연결됨")

    def disconnect_robot(self):
        self.robot_ws = None
        print("🤖 로봇 끊김")

    def disconnect_user(self):
        self.user_ws = None

    async def send_command_to_robot(self, command: str):
        if self.robot_ws:
            try: await self.robot_ws.send_text(command)
            except: pass

    async def send_video_to_user(self, data: bytes):
        if self.user_ws:
            try: await self.user_ws.send_bytes(data)
            except: pass

manager = ConnectionManager()

@app.websocket("/ws/robot")
async def robot_endpoint(websocket: WebSocket):
    await manager.connect_robot(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            await manager.send_video_to_user(data)
    except: manager.disconnect_robot()

@app.websocket("/ws/user")
async def user_endpoint(websocket: WebSocket):
    await manager.connect_user(websocket)
    try:
        while True:
            cmd = await websocket.receive_text()
            await manager.send_command_to_robot(cmd)
    except: manager.disconnect_user()

# --- [수정된 UI] FPS 오버레이 추가 ---
@app.get("/", response_class=HTMLResponse)
def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Cockpit</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            body { 
                background-color: #222; 
                color: white; 
                text-align: center; 
                font-family: 'Consolas', sans-serif; /* 해커 느낌 폰트 */
                margin: 0; padding: 0;
                touch-action: manipulation;
            }
            
            /* 비디오 컨테이너 (기준점) */
            .video-container {
                position: relative; /* 중요: 이것을 기준으로 FPS 위치를 잡음 */
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
                background: #000;
                border-bottom: 2px solid #444;
            }
            
            #stream { width: 100%; display: block; min-height: 240px; }

            /* FPS 오버레이 스타일 */
            #fps-counter {
                position: absolute;
                top: 10px;
                left: 10px;
                color: #00ff00; /* 형광 초록 */
                font-weight: bold;
                font-size: 16px;
                background-color: rgba(0, 0, 0, 0.5); /* 반투명 배경 */
                padding: 4px 8px;
                border-radius: 4px;
                pointer-events: none; /* 클릭 통과 */
            }

            .control-pad {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                max-width: 380px;
                margin: 20px auto;
                padding: 15px;
            }

            button {
                width: 100%;
                aspect-ratio: 1;
                font-size: 40px;
                background-color: #444;
                border: 2px solid #666;
                border-radius: 15px;
                color: white;
                cursor: pointer;
                user-select: none;
                -webkit-tap-highlight-color: transparent;
            }
            button:active { background-color: #00cc00; transform: scale(0.92); }
            .stop-btn { background-color: #cc0000; font-size: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="video-container">
            <img id="stream" src="" alt="Waiting..." />
            <div id="fps-counter">FPS: 0</div>
        </div>

        <div class="control-pad">
            <div></div>
            <button onmousedown="send('forward')" onmouseup="send('stop')" ontouchstart="send('forward')" ontouchend="send('stop')">⬆️</button>
            <div></div>

            <button onmousedown="send('left')" onmouseup="send('stop')" ontouchstart="send('left')" ontouchend="send('stop')">⬅️</button>
            <button class="stop-btn" onclick="send('stop')">STOP</button>
            <button onmousedown="send('right')" onmouseup="send('stop')" ontouchstart="send('right')" ontouchend="send('stop')">➡️</button>

            <div></div>
            <button onmousedown="send('backward')" onmouseup="send('stop')" ontouchstart="send('backward')" ontouchend="send('stop')">⬇️</button>
            <div></div>
        </div>

        <script>
            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            var ws = new WebSocket(protocol + "//" + window.location.host + "/ws/user");

            // --- FPS 계산 로직 ---
            var frameCount = 0;
            var lastTime = Date.now();
            var fpsDisplay = document.getElementById("fps-counter");

            ws.onmessage = function(event) {
                // 1. 이미지 갱신
                var url = URL.createObjectURL(event.data);
                document.getElementById("stream").src = url;

                // 2. 프레임 카운트 증가
                frameCount++;
                var now = Date.now();
                
                // 3. 1초마다 FPS 갱신
                if (now - lastTime >= 1000) {
                    fpsDisplay.innerText = "FPS: " + frameCount;
                    
                    // 색상 변경 (느리면 빨강, 빠르면 초록)
                    if(frameCount < 10) fpsDisplay.style.color = "red";
                    else if(frameCount < 20) fpsDisplay.style.color = "orange";
                    else fpsDisplay.style.color = "#00ff00";

                    frameCount = 0;
                    lastTime = now;
                }
            };

            function send(cmd) { 
                if(ws.readyState === WebSocket.OPEN) ws.send(cmd); 
            }
            
            document.addEventListener('keydown', (e) => {
                if(e.repeat) return;
                if(e.key=="ArrowUp") send("forward");
                else if(e.key=="ArrowDown") send("backward");
                else if(e.key=="ArrowLeft") send("left");
                else if(e.key=="ArrowRight") send("right");
                else if(e.key==" ") send("stop");
            });

            document.addEventListener('keyup', (e) => {
                if(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) send("stop");
            });
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)