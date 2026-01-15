# robot_control.py (아커만 주행용 수정)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController:
    def __init__(self):
        print("🤖 멘토파이(아커만) 컨트롤러 초기화")
        self.node = rclpy.create_node('web_teleop_controller')
        
        # 아커만 방식은 보통 /cmd_vel 또는 /ackermann_cmd 등을 씁니다.
        # 기존에 잘 되던 topic 이름(/controller/cmd_vel)을 유지하세요.
        self.publisher_ = self.node.create_publisher(Twist, '/cmd_vel', 10)
        
        # [설정] 속도와 조향각 조절
        self.SPEED = 0.2        # 전진 속도 (m/s)
        self.STEERING_ANGLE = 0.5  # 조향 각도 (radian) - 너무 크면 서보 무리감
                                   # 0.5 라디안은 약 28도 정도입니다.

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.publisher_.publish(msg)

    def move_forward(self, speed=100):
        print("⬆️ 전진")
        self.publish_cmd(self.SPEED, 0.0) # 핸들 중앙, 전진

    def move_backward(self, speed=100):
        print("⬇️ 후진")
        self.publish_cmd(-self.SPEED, 0.0) # 핸들 중앙, 후진

    # [핵심 수정] 아커만은 전진하면서 핸들을 꺾어야 돕니다!
    def turn_left(self, speed=80):
        print("↖️ 좌회전 (전진+핸들)")
        # 전진(SPEED) + 좌측 핸들(STEERING_ANGLE)
        self.publish_cmd(self.SPEED, self.STEERING_ANGLE)

    def turn_right(self, speed=80):
        print("↗️ 우회전 (전진+핸들)")
        # 전진(SPEED) + 우측 핸들(-STEERING_ANGLE)
        self.publish_cmd(self.SPEED, -self.STEERING_ANGLE)

    def stop(self):
        print("🛑 정지")
        self.publish_cmd(0.0, 0.0)

    def cleanup(self):
        self.node.destroy_node()