import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController:
    def __init__(self):
        print("🚀 멘토파이(매카넘) 컨트롤러 초기화")
        try:
            self.node = rclpy.create_node('web_teleop_controller')
            self.publisher_ = self.node.create_publisher(Twist, '/cmd_vel', 10)
        except Exception as e:
            print(f"⚠️ ROS2 노드 생성 실패: {e}")
        
        # [설정] 속도 값 (0.0 ~ 1.0 사이 권장)
        self.LINEAR_SPEED = 0.5   
        self.ANGULAR_SPEED = 0.8  

    def publish_cmd(self, lx=0.0, ly=0.0, az=0.0):
        msg = Twist()
        msg.linear.x = float(lx)   # 전/후
        msg.linear.y = float(ly)   # 좌/우 (매카넘 핵심)
        msg.angular.z = float(az)  # 회전
        self.publisher_.publish(msg)

    # 모든 함수에 speed=100 같은 인자 수용 가능하도록 설정
    def move_forward(self, speed=None):
        print("⬆️ 전진")
        self.publish_cmd(lx=self.LINEAR_SPEED)

    def move_backward(self, speed=None):
        print("⬇️ 후진")
        self.publish_cmd(lx=-self.LINEAR_SPEED)

    def move_left(self, speed=None):
        print("⬅️ 왼쪽 게걸음")
        self.publish_cmd(ly=self.LINEAR_SPEED)

    def move_right(self, speed=None):
        print("➡️ 오른쪽 게걸음")
        self.publish_cmd(ly=-self.LINEAR_SPEED)

    def turn_left(self, speed=None):
        print("🔄 제자리 좌회전")
        self.publish_cmd(az=self.ANGULAR_SPEED)

    def turn_right(self, speed=None):
        print("🔄 제자리 우회전")
        self.publish_cmd(az=-self.ANGULAR_SPEED)

    def stop(self, speed=None):
        print("🛑 정지")
        self.publish_cmd(0.0, 0.0, 0.0)

    def cleanup(self):
        self.node.destroy_node()