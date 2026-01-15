# client.py (ROS2 초기화 순서 수정 완료)
import asyncio
import websockets
import cv2
import rclpy
from robot_control import RobotController

# [중요] 엘리스 터널 주소 (wss://...)
SERVER_URL = "wss://awdanmxyaxxcgabw.tunnel.elice.io/ws/robot"

# 1. 전역 변수 선언 (여기서는 비워둡니다)
robot = None 

# 2. 카메라 초기화 (카메라는 ROS와 상관없으니 미리 해도 됨)
camera = cv2.VideoCapture(0)
camera_available = False

if camera.isOpened():
    camera.set(3, 320)
    camera.set(4, 240)
    camera_available = True
    print("🎥 카메라 연결 성공")
else:
    print("⚠️ 경고: 카메라를 찾을 수 없습니다.")

async def run_robot():
    print(f"🔗 서버 연결 시도: {SERVER_URL}")
    async with websockets.connect(SERVER_URL) as websocket:
        print("✅ 서버에 연결되었습니다!")
        
        while True:
            # --- 영상 전송 ---
            if camera_available:
                ret, frame = camera.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    await websocket.send(buffer.tobytes())
            
            # --- 명령 수신 ---
            try:
                wait_time = 0.01 if camera_available else 0.1
                command = await asyncio.wait_for(websocket.recv(), timeout=wait_time)
                
                if isinstance(command, str) and robot is not None:
                    print(f"명령 수신: {command}")
                    if command == 'forward': robot.move_forward(100)
                    elif command == 'backward': robot.move_backward(100)
                    elif command == 'left': robot.turn_left(100)
                    elif command == 'right': robot.turn_right(100)
                    elif command == 'stop': robot.stop()
                    
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed:
                print("❌ 서버 연결 끊김")
                break
            except Exception as e:
                print(f"⚠️ 에러: {e}")
                break

if __name__ == "__main__":
    try:
        # [핵심] 반드시 여기서 먼저 초기화를 해야 합니다!
        print("🚀 ROS2 시스템 시작...")
        rclpy.init() 
        
        # [핵심] 초기화가 끝난 뒤에 로봇 컨트롤러 생성
        robot = RobotController()
        
        # 비동기 루프 시작
        asyncio.run(run_robot())
        
    except KeyboardInterrupt:
        print("프로그램 종료")
    finally:
        # 종료 처리
        if robot:
            robot.stop()
            robot.cleanup()
        if camera_available:
            camera.release()
        
        # ROS2 종료
        if rclpy.ok():
            rclpy.shutdown()