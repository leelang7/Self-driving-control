import asyncio
import websockets
import cv2
import rclpy
from meca_robot_control import RobotController

SERVER_URL = "wss://awdanmxyaxxcgabw.tunnel.elice.io/ws/robot"

robot = None 
camera = cv2.VideoCapture(0)
camera_available = camera.isOpened()

if camera_available:
    camera.set(3, 320)
    camera.set(4, 240)
    print("🎥 카메라 연결 성공")
else:
    print("⚠️ 카메라를 찾을 수 없습니다.")

async def run_robot():
    print(f"🔗 서버 연결 시도: {SERVER_URL}")
    async with websockets.connect(SERVER_URL) as websocket:
        print("✅ 서버에 연결되었습니다!")
        
        while True:
            # 1. 영상 전송
            if camera_available:
                ret, frame = camera.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    await websocket.send(buffer.tobytes())
            
            # 2. 명령 수신 (AttributeError 방지 로직)
            try:
                wait_time = 0.01 if camera_available else 0.1
                command = await asyncio.wait_for(websocket.recv(), timeout=wait_time)
                
                if isinstance(command, str) and robot is not None:
                    cmd = command.lower().strip()
                    print(f"명령 수신: {cmd}")
                    
                    if cmd == 'forward':    robot.move_forward(100)
                    elif cmd == 'backward': robot.move_backward(100)
                    # 매카넘은 left/right를 게걸음으로 쓰는 것이 일반적입니다.
                    elif cmd == 'left':     robot.move_left(100)
                    elif cmd == 'right':    robot.move_right(100)
                    # 만약 회전 명령이 따로 들어온다면:
                    elif cmd == 'turn_left':  robot.turn_left(100)
                    elif cmd == 'turn_right': robot.turn_right(100)
                    elif cmd == 'stop':     robot.stop()
                    
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"⚠️ 루프 에러: {e}")
                break

if __name__ == "__main__":
    try:
        print("🚀 ROS2 시스템 시작...")
        rclpy.init() 
        robot = RobotController()
        asyncio.run(run_robot())
    except KeyboardInterrupt:
        print("프로그램 종료")
    finally:
        if robot:
            robot.stop()
            robot.cleanup()
        if camera_available:
            camera.release()
        if rclpy.ok():
            rclpy.shutdown()