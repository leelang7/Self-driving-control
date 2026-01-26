import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
import cv2
import numpy as np
import math
import asyncio
import websockets
import threading

# [중요] 로봇 제어 모듈
try:
    from robot_control import RobotController
    HAS_ROBOT_CONTROL = True
except ImportError:
    print("⚠️ 경고: 'robot_control.py'를 찾을 수 없습니다.")
    HAS_ROBOT_CONTROL = False

ROBOT_ID = "robot1" 
SERVER_URI = f"wss://hsdstnapptmqhcmc.tunnel.elice.io/ws/robot/{ROBOT_ID}"

class IntegratedFleetNode(Node):
    def __init__(self):
        super().__init__('integrated_fleet_node')
        
        # 1. 로봇 제어
        self.robot = None
        if HAS_ROBOT_CONTROL:
            try:
                self.robot = RobotController()
                self.get_logger().info("✅ RobotController 연결 성공")
            except Exception as e:
                self.get_logger().error(f"❌ RobotController 초기화 실패: {e}")

        # 2. 카메라
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.get_logger().info(f"🎥 카메라 연결 성공 (ID: {ROBOT_ID})")
        else:
            self.get_logger().error("⚠️ 카메라를 열 수 없습니다!")

        # 3. ROS 구독
        self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        
        self.latest_scan = None
        
        # ==========================================
        # ★ [지도 고정 모드 설정]
        # ==========================================
        self.display_yaw = 0.0       
        self.last_raw_yaw = None     
        
        # 방향 설정 (왼쪽 돌 때 화살표도 왼쪽 가도록 1 설정)
        self.YAW_DIRECTION = 1       
        
        # 떨림 방지 (너무 민감하면 0.01로 올리세요)
        self.DRIFT_THRESHOLD = 0.005

        # 맵 설정
        self.MAP_SIZE = 400
        self.MAX_DIST = 4.0
        self.SCALE = (self.MAP_SIZE / 2) / self.MAX_DIST
        self.CENTER = int(self.MAP_SIZE / 2)

        self.get_logger().info(f"🚀 [Fleet Client] 로봇 ID: {ROBOT_ID} 시작")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def imu_callback(self, msg):
        q = msg.orientation
        
        # 1. 현재 센서값(Raw)
        current_raw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        current_raw *= self.YAW_DIRECTION

        if self.last_raw_yaw is None:
            self.last_raw_yaw = current_raw
            return

        # 2. 변화량 계산
        delta = current_raw - self.last_raw_yaw

        # 각도 끊김 방지
        if delta > math.pi: delta -= 2 * math.pi
        elif delta < -math.pi: delta += 2 * math.pi

        # 3. 떨림 방지
        if abs(delta) < self.DRIFT_THRESHOLD:
            delta = 0.0

        # 4. 각도 누적 (여기가 로봇의 절대 방향)
        self.display_yaw += delta
        self.last_raw_yaw = current_raw

    def reset_gyro(self):
        self.display_yaw = 0.0
        self.last_raw_yaw = None
        self.get_logger().info("🔄 리셋됨")

    def get_camera_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret: return frame
        return None

    def draw_premium_radar(self):
        img = np.full((self.MAP_SIZE, self.MAP_SIZE, 3), 20, dtype=np.uint8)
        
        COLOR_GRID = (60, 60, 60)       
        COLOR_TEXT = (150, 150, 150)    
        COLOR_LIDAR = (0, 255, 200)     
        COLOR_ROBOT = (0, 100, 255)     
        COLOR_HEADING = (0, 0, 255)     

        # 그리드
        for r in range(1, int(self.MAX_DIST) + 1):
            radius = int(r * self.SCALE)
            cv2.circle(img, (self.CENTER, self.CENTER), radius, COLOR_GRID, 1, cv2.LINE_AA)
            cv2.putText(img, f"{r}m", (self.CENTER + 5, self.CENTER - radius + 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)

        # 각도선 (배경이므로 고정)
        max_r_pixel = int(self.MAX_DIST * self.SCALE)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x_end = self.CENTER - int(max_r_pixel * math.sin(rad))
            y_end = self.CENTER - int(max_r_pixel * math.cos(rad))
            cv2.line(img, (self.CENTER, self.CENTER), (x_end, y_end), COLOR_GRID, 1, cv2.LINE_AA)
            
            disp_angle = angle if angle <= 180 else angle - 360
            text_x = self.CENTER - int((max_r_pixel + 20) * math.sin(rad)) - 15
            text_y = self.CENTER - int((max_r_pixel + 20) * math.cos(rad)) + 5
            cv2.putText(img, f"{disp_angle}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)

        cv2.line(img, (self.CENTER, 0), (self.CENTER, self.MAP_SIZE), COLOR_GRID, 1)
        cv2.line(img, (0, self.CENTER), (self.MAP_SIZE, self.CENTER), COLOR_GRID, 1)

        # ★ [핵심 수정] 라이다 그리기
        if self.latest_scan:
            ranges = np.array(self.latest_scan.ranges)
            angle_min = self.latest_scan.angle_min
            angle_inc = self.latest_scan.angle_increment
            
            # 로봇의 현재 회전각 (이걸 더해줘야 벽이 고정됨)
            robot_yaw = self.display_yaw

            for i, r in enumerate(ranges):
                if 0.1 < r < self.MAX_DIST:
                    # 기존: theta = angle_min + i * angle_inc (로봇 기준)
                    # 수정: 로봇 회전각(robot_yaw)을 더해서 '절대 좌표'로 변환
                    theta = angle_min + i * angle_inc + robot_yaw
                    
                    x = self.CENTER - int(r * self.SCALE * math.sin(theta)) 
                    y = self.CENTER - int(r * self.SCALE * math.cos(theta))
                    
                    if 0 <= x < self.MAP_SIZE and 0 <= y < self.MAP_SIZE:
                        cv2.circle(img, (x, y), 1, COLOR_LIDAR, -1, cv2.LINE_AA)

            # ★ [핵심 수정] 화살표 그리기
            # 화살표는 로봇의 방향이므로 robot_yaw 방향으로 회전해야 함
            heading_deg = math.degrees(robot_yaw)
            heading_deg = (heading_deg + 180) % 360 - 180
            
            cv2.putText(img, f"ID: {ROBOT_ID} | YAW: {heading_deg:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            imu_len = self.MAP_SIZE // 3
            # 화살표 끝점 계산 (로봇 각도 반영)
            imu_x = self.CENTER - int(imu_len * math.sin(robot_yaw))
            imu_y = self.CENTER - int(imu_len * math.cos(robot_yaw))
            
            # 화살표 그리기 (이제 화살표가 돌아갑니다!)
            cv2.arrowedLine(img, (self.CENTER, self.CENTER), (imu_x, imu_y), 
                            COLOR_HEADING, 3, cv2.LINE_AA, tipLength=0.2)
        else:
            cv2.putText(img, f"ID: {ROBOT_ID} | SCANNING...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # 로봇 본체 마커 (얘는 항상 가운데 고정)
        pt1 = (self.CENTER, self.CENTER - 8)
        pt2 = (self.CENTER - 6, self.CENTER + 6)
        pt3 = (self.CENTER + 6, self.CENTER + 6)
        cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, COLOR_ROBOT, -1)

        return img

    def get_fused_image(self):
        frame = self.get_camera_frame()
        if frame is None:
            cam_img = np.zeros((300, 400, 3), np.uint8)
            cv2.putText(cam_img, "NO CAMERA", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cam_img = cv2.resize(frame, (400, 300))
        lidar_img = self.draw_premium_radar()
        if cam_img.shape[1] != lidar_img.shape[1]:
            lidar_img = cv2.resize(lidar_img, (cam_img.shape[1], 400))
        return np.vstack([cam_img, lidar_img])

    def execute_command(self, cmd):
        if self.robot is None: return
        
        if cmd == 'reset_gyro' or cmd == 'reset':
            self.reset_gyro()
            return

        if cmd == 'forward': self.robot.move_forward(100)
        elif cmd == 'backward': self.robot.move_backward(100)
        elif cmd == 'left': self.robot.turn_left(100)
        elif cmd == 'right': self.robot.turn_right(100)
        elif cmd == 'stop': self.robot.stop()

    def destroy_node(self):
        if self.cap.isOpened(): self.cap.release()
        super().destroy_node()

async def main_loop(node):
    print(f"🔗 관제 서버 접속 시도: {SERVER_URI}")
    async with websockets.connect(SERVER_URI) as ws:
        print(f"✅ 서버 접속 완료! (ID: {ROBOT_ID})")
        while True:
            fused = node.get_fused_image()
            _, buf = cv2.imencode('.jpg', fused, [cv2.IMWRITE_JPEG_QUALITY, 60])
            await ws.send(buf.tobytes())

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                node.execute_command(msg)
            except asyncio.TimeoutError: pass
            except Exception:
                print("⚠️ 서버 연결 끊김")
                break
            await asyncio.sleep(0.03)

def main():
    rclpy.init()
    node = IntegratedFleetNode()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    try:
        asyncio.run(main_loop(node))
    except KeyboardInterrupt:
        print("종료 중...")
    finally:
        if node.robot: 
            node.robot.stop()
            try: node.robot.cleanup()
            except: pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()