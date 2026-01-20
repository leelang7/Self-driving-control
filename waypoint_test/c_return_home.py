import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math
import csv
import os
import numpy as np

class ReturnHomeMecanum(Node):
    def __init__(self):
        super().__init__('return_home_mecanum')

        # === 튜닝 파라미터 ===
        self.LOOKAHEAD_DIST = 0.6   # 전방 주시 거리 (m)
        self.TARGET_SPEED = 0.25    # 복귀 속도 (m/s)
        self.ROTATION_GAIN = 1.5    # 회전 민감도 (메카넘 전용)

        # === 상태 변수 ===
        self.is_running = False
        self.path = []
        self.current_idx = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # === ROS 통신 ===
        # 1. 복귀 명령 수신 (True가 들어오면 출발)
        self.sub_trigger = self.create_subscription(Bool, '/start_return', self.trigger_cb, 10)
        
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info("Ready to Return. Waiting for '/start_return' topic...")

    def trigger_cb(self, msg):
        """명령을 받으면 경로를 로드하고 주행 시작"""
        if msg.data and not self.is_running:
            if self.load_and_reverse_path():
                self.is_running = True
                self.get_logger().info("RETURN SEQUENCE STARTED!")

    def load_and_reverse_path(self):
        file_path = 'path.csv'
        if not os.path.exists(file_path):
            self.get_logger().error(f"'{file_path}' not found!")
            return False
            
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                raw_path = [[float(row[0]), float(row[1])] for row in reader]
            
            if not raw_path:
                return False

            # [핵심] 경로 뒤집기 (도착점 -> 시작점)
            self.path = raw_path[::-1]
            self.current_idx = 0
            self.get_logger().info(f"Path Loaded: {len(self.path)} points. (Reversed)")
            return True
        except Exception as e:
            self.get_logger().error(f"Error loading path: {e}")
            return False

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        if self.is_running:
            self.control_loop()

    def control_loop(self):
        # 1. 목표점(Target Point) 찾기
        target_idx = self.current_idx
        found = False
        for i in range(self.current_idx, len(self.path)):
            dist = math.hypot(self.path[i][0] - self.x, self.path[i][1] - self.y)
            if dist >= self.LOOKAHEAD_DIST:
                target_idx = i
                found = True
                break
        
        if not found: target_idx = len(self.path) - 1
        self.current_idx = target_idx

        # 2. 도착 판정 (마지막 점과 20cm 이내)
        dist_to_end = math.hypot(self.path[-1][0] - self.x, self.path[-1][1] - self.y)
        if dist_to_end < 0.2:
            self.stop_robot()
            return

        # 3. Pure Pursuit 계산 (메카넘 버전)
        tx, ty = self.path[target_idx]
        dx = tx - self.x
        dy = ty - self.y

        # 로봇 기준 좌표계 변환
        local_x = math.cos(-self.yaw) * dx - math.sin(-self.yaw) * dy
        local_y = math.sin(-self.yaw) * dx + math.cos(-self.yaw) * dy

        # 곡률 계산
        lookahead_sq = local_x**2 + local_y**2
        curvature = 2.0 * local_y / lookahead_sq if lookahead_sq > 0 else 0.0

        # 명령 생성
        cmd = Twist()
        cmd.linear.x = self.TARGET_SPEED
        cmd.linear.y = 0.0
        # 각속도 = 선속도 * 곡률 * 게인
        cmd.angular.z = (curvature * self.TARGET_SPEED) * self.ROTATION_GAIN

        self.pub_cmd.publish(cmd)

    def stop_robot(self):
        self.pub_cmd.publish(Twist())
        self.is_running = False
        self.get_logger().info("ARRIVED HOME. Stopping.")

def main():
    rclpy.init()
    node = ReturnHomeMecanum()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()