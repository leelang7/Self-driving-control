import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
# [핵심] QoS 관련 모듈 임포트
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import csv
import math
import os

class WaypointFollowerImuFusion(Node):
    def __init__(self):
        super().__init__('waypoint_follower_imu_fusion')
        
        # ---------------------------------------------------------
        # [QoS 설정 분리]
        # 1. 센서용 (Best Effort): 데이터가 자주 오므로 최신값만 받음
        # 2. 제어용 (Reliable): 명령이 유실되면 안 되므로 확실히 보냄
        # ---------------------------------------------------------
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_ctrl = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------------------------------------------------------
        # 1. 토픽 구독 (센서용 QoS 적용)
        # ---------------------------------------------------------
        self.sub_odom = self.create_subscription(
            Odometry, 
            '/odom',  
            self.odom_cb, 
            qos_sensor
        )

        self.sub_imu = self.create_subscription(
            Imu,
            '/imu',
            self.imu_cb,
            qos_sensor
        )
        
        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan_raw', 
            self.scan_cb,
            qos_sensor
        )

        # ---------------------------------------------------------
        # 2. 토픽 발행 (제어용 QoS 적용)
        # ---------------------------------------------------------
        self.pub_cmd = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            qos_ctrl
        )

        # ---------------------------------------------------------
        # 3. 변수 및 경로 로드
        # ---------------------------------------------------------
        self.file_path = 'path.csv'
        self.waypoints = []
        self.load_path()
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.imu_yaw_offset = None 
        self.obstacle_detected = False
        self.current_idx = 0

        # 제어 파라미터
        self.target_dist_tol = 0.15  # 목표 도달 범위 (m)
        self.linear_speed = 0.2      # 주행 속도
        self.angular_k = 1.0         # 회전 민감도

        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        self.get_logger().info("System Ready. Waiting for Sensor Data...")

    def load_path(self):
        """CSV 파일에서 경로 읽기"""
        if not os.path.exists(self.file_path):
            self.get_logger().error(f"File not found: {self.file_path}")
            return
        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try: self.waypoints.append((float(row[0]), float(row[1])))
                except: pass

    def odom_cb(self, msg):
        """위치 정보 업데이트"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def imu_cb(self, msg):
        """방향(Yaw) 정보 업데이트"""
        q = msg.orientation
        raw_yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)

        # 첫 IMU 데이터가 들어오면 현재 방향을 0도로 초기화
        if self.imu_yaw_offset is None:
            self.imu_yaw_offset = raw_yaw
            self.get_logger().info("✅ IMU Connected! Starting Navigation...")
        
        # 보정된 Yaw 계산
        self.current_yaw = raw_yaw - self.imu_yaw_offset
        
        # 각도 정규화 (-PI ~ PI)
        while self.current_yaw > math.pi: self.current_yaw -= 2 * math.pi
        while self.current_yaw < -math.pi: self.current_yaw += 2 * math.pi

    def scan_cb(self, msg):
        """장애물 감지"""
        if len(msg.ranges) == 0: return
        
        # 전방 30도 범위 데이터 추출
        n = len(msg.ranges)
        idx = int(15 * (n / 360.0))
        front = msg.ranges[:idx] + msg.ranges[-idx:]
        
        # 유효 거리 필터링 (0.3m 이내 장애물 감지)
        valid = [r for r in front if 0.01 < r < 10.0]
        if valid and min(valid) < 0.3: 
            self.obstacle_detected = True
        else: 
            self.obstacle_detected = False

    def control_loop(self):
        """주행 제어 루프"""
        twist = Twist()
        
        # 1. IMU 초기화 대기
        if self.imu_yaw_offset is None:
            return 

        # 2. 도착 완료 확인
        if self.current_idx >= len(self.waypoints):
            self.pub_cmd.publish(twist) # 정지
            self.get_logger().info("🏁 Goal Reached! Navigation Finished.")
            self.timer.cancel()
            return

        # 3. 장애물 감지 시 정지
        if self.obstacle_detected:
            self.pub_cmd.publish(twist)
            self.get_logger().warn("🚨 Obstacle Detected! Stopping...")
            return

        # 4. 목표 지점 계산
        target_x, target_y = self.waypoints[self.current_idx]
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        dist = math.hypot(dx, dy)

        # 5. 웨이포인트 도달 확인
        if dist < self.target_dist_tol:
            self.current_idx += 1
            self.get_logger().info(f"📍 Reached Waypoint {self.current_idx}/{len(self.waypoints)}")
            return

        # 6. 주행 제어 (P-Control)
        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - self.current_yaw
        
        # 각도 에러 정규화
        while yaw_error > math.pi: yaw_error -= 2 * math.pi
        while yaw_error < -math.pi: yaw_error += 2 * math.pi

        # 각도가 많이 틀어졌으면 제자리 회전, 아니면 직진하며 회전
        if abs(yaw_error) > math.radians(20):
            twist.linear.x = 0.0
            twist.angular.z = self.angular_k * yaw_error
        else:
            twist.linear.x = self.linear_speed
            twist.angular.z = self.angular_k * yaw_error

        self.pub_cmd.publish(twist)

    def euler_from_quaternion(self, x, y, z, w):
        """쿼터니언 -> 오일러(Yaw) 변환 함수"""
        t0 = +2.0 * (w * z + x * y)
        t1 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t0, t1)

def main():
    rclpy.init()
    node = WaypointFollowerImuFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by User.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()