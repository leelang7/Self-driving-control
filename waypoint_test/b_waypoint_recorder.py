import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import math
import os

class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        self.file_path = 'path.csv'
        self.last_x = 0.0
        self.last_y = 0.0
        self.min_dist = 0.1 # 10cm 이동마다 기록
        self.is_first = True
        
        # 파일 초기화 (기존 파일 삭제)
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            
        self.get_logger().info("Recording Started... Drive the robot!")

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 첫 위치는 무조건 저장
        if self.is_first:
            self.save_point(x, y)
            self.last_x, self.last_y = x, y
            self.is_first = False
            return

        # 이전 점과 거리가 min_dist 이상일 때만 저장
        dist = math.hypot(x - self.last_x, y - self.last_y)
        if dist > self.min_dist:
            self.save_point(x, y)
            self.last_x = x
            self.last_y = y

    def save_point(self, x, y):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([x, y])
        self.get_logger().info(f"Saved: {x:.2f}, {y:.2f}")

def main():
    rclpy.init()
    node = WaypointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Recording Finished. 'path.csv' saved.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()