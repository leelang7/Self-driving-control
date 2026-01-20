import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import sys

class MoveOneMeter(Node):
    def __init__(self):
        super().__init__('move_one_meter')
        
        # === 설정 ===
        self.TARGET_DIST = 1.0  # 목표 거리 (m)
        self.SPEED = 0.2        # 이동 속도 (m/s) - 천천히
        
        # === 상태 변수 ===
        self.start_x = None
        self.start_y = None
        self.current_dist = 0.0
        self.is_finished = False

        # === ROS 통신 ===
        self.sub_odom = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10
        )
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info(f"STARTING: Move Forward {self.TARGET_DIST}m")

    def odom_callback(self, msg):
        if self.is_finished:
            return

        # 현재 위치 파싱
        cur_x = msg.pose.pose.position.x
        cur_y = msg.pose.pose.position.y

        # 1. 시작 위치가 없으면 현재 위치를 시작점으로 설정
        if self.start_x is None:
            self.start_x = cur_x
            self.start_y = cur_y
            self.get_logger().info(f"Start Point Saved: x={cur_x:.2f}, y={cur_y:.2f}")
            return

        # 2. 이동 거리 계산 (피타고라스)
        dx = cur_x - self.start_x
        dy = cur_y - self.start_y
        self.current_dist = math.hypot(dx, dy)

        # 3. 제어 로직
        if self.current_dist < self.TARGET_DIST:
            self.move_robot()
            # 진행 상황 로그 (0.1m 단위로)
            # self.get_logger().info(f"Dist: {self.current_dist:.2f}m") 
        else:
            self.stop_robot()

    def move_robot(self):
        cmd = Twist()
        cmd.linear.x = self.SPEED
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.pub_cmd.publish(cmd)

    def stop_robot(self):
        # 정지 명령
        cmd = Twist()
        self.pub_cmd.publish(cmd)
        
        self.is_finished = True
        self.get_logger().info("="*30)
        self.get_logger().info(f"TARGET REACHED! Odom says: {self.current_dist:.4f}m")
        self.get_logger().info("Please measure the REAL distance with a tape measure.")
        self.get_logger().info("="*30)
        
        # 노드 종료
        raise SystemExit

def main():
    rclpy.init()
    node = MoveOneMeter()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Main").info('Test Finished.')
    except KeyboardInterrupt:
        node.get_logger().info('Forced Stop.')
        node.pub_cmd.publish(Twist()) # 강제 종료 시 정지 명령
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()