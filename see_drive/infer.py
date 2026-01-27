#!/usr/bin/env python3
import os
import sys

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

import torch
import numpy as np
import cv2

# YOLO 라이브러리 추가
from ultralytics import YOLO

# DeepLabV3_MobileNet.py 파일이 같은 폴더나 파이썬 경로에 있어야 함
from DeepLabV3_MobileNet import DeepLabMobileNet_RegCmdVel, DeepLabV3Regressor2Head


print("=== RUNNING FILE:", __file__)


class DeepLabYoloInferenceNode(Node):
    def __init__(self):
        super().__init__('deeplab_yolo_inference_node')

        # -------------------------
        # DeepLab Params (주행용)
        # -------------------------
        self.declare_parameter('CROP_HEIGHT', 100)
        self.declare_parameter('LINEAR_GAIN', 0.8)
        self.declare_parameter('STEERING_GAIN', 1.0)
        self.declare_parameter('MODEL_PATH', 'best_model_reg_only.pth')
        
        # -------------------------
        # YOLO Params (장애물 감지용)
        # -------------------------
        self.declare_parameter('YOLO_PATH', 'yolov8n.pt')       # YOLO 모델 경로
        self.declare_parameter('YOLO_CONF', 0.5)                # 감지 임계값
        self.declare_parameter('STOP_CLASSES', [0])             # 멈출 클래스 ID (0: Person)
        self.declare_parameter('STOP_AREA_RATIO', 0.10)         # 화면의 10% 이상 크기면 정지 (거리 판단용)

        # -------------------------
        # Common Params
        # -------------------------
        self.declare_parameter('input_video', '/front_cam/image/compressed')
        self.declare_parameter('output_cmd_topic', '/controller/cmd_vel')
        self.declare_parameter('OUT_W', 320)
        self.declare_parameter('OUT_H', 192)

        # Safety Clamps
        self.declare_parameter('CLAMP_LINEAR', True)
        self.declare_parameter('LINEAR_MIN', -0.5)
        self.declare_parameter('LINEAR_MAX',  0.5)
        self.declare_parameter('ANGULAR_MIN', -2.0)
        self.declare_parameter('ANGULAR_MAX',  2.0)

        self.declare_parameter('LOG_EVERY_N', 10)

        # Segmentation Visualization
        self.declare_parameter('PUB_SEG', True)
        self.declare_parameter('seg_overlay_topic', '/deeplab/seg_yolo_overlay')
        self.declare_parameter('SEG_ALPHA', 0.45) 

        # -------------------------
        # Read Params
        # -------------------------
        self.CROP_HEIGHT = self.get_parameter('CROP_HEIGHT').value
        self.LINEAR_GAIN = self.get_parameter('LINEAR_GAIN').value
        self.STEERING_GAIN = self.get_parameter('STEERING_GAIN').value
        self.MODEL_PATH = self.get_parameter('MODEL_PATH').value

        self.YOLO_PATH = self.get_parameter('YOLO_PATH').value
        self.YOLO_CONF = self.get_parameter('YOLO_CONF').value
        self.STOP_CLASSES = self.get_parameter('STOP_CLASSES').value  # List[int]
        self.STOP_AREA_RATIO = self.get_parameter('STOP_AREA_RATIO').value

        self.input_video = self.get_parameter('input_video').value
        self.output_cmd_topic = self.get_parameter('output_cmd_topic').value
        self.OUT_W = self.get_parameter('OUT_W').value
        self.OUT_H = self.get_parameter('OUT_H').value

        self.CLAMP_LINEAR = self.get_parameter('CLAMP_LINEAR').value
        self.LINEAR_MIN = self.get_parameter('LINEAR_MIN').value
        self.LINEAR_MAX = self.get_parameter('LINEAR_MAX').value
        self.ANGULAR_MIN = self.get_parameter('ANGULAR_MIN').value
        self.ANGULAR_MAX = self.get_parameter('ANGULAR_MAX').value

        self.LOG_EVERY_N = self.get_parameter('LOG_EVERY_N').value
        self.PUB_SEG = self.get_parameter('PUB_SEG').value
        self.seg_overlay_topic = self.get_parameter('seg_overlay_topic').value
        self.SEG_ALPHA = self.get_parameter('SEG_ALPHA').value

        self._frame_count = 0

        # -------------------------
        # ROS Setup
        # -------------------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            CompressedImage, self.input_video, self.image_callback, 1
        )
        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.overlay_pub = self.create_publisher(Image, self.seg_overlay_topic, 10)

        # -------------------------
        # Load Models
        # -------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. DeepLab (Driving)
        self.driving_model = self.load_deeplab_model(self.MODEL_PATH, self.device)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 2. YOLO (Obstacle)
        self.get_logger().info(f"Loading YOLO model from {self.YOLO_PATH}...")
        self.yolo_model = YOLO(self.YOLO_PATH)
        
        self.get_logger().info("✅ DeepLab + YOLO Node Ready!")

    def load_deeplab_model(self, ckpt_path, device):
        self.get_logger().info(f"Loading Driving Model from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        keys = list(ckpt.keys())

        if any(k.startswith("seg.backbone.") for k in keys) or any(k.startswith("seg.") for k in keys):
            model = DeepLabMobileNet_RegCmdVel(pretrained=True, freeze_backbone=False, head_hidden=256, dropout=0.2)
        elif any(k.startswith("head_lin.") for k in keys):
            model = DeepLabV3Regressor2Head(pretrained=True)
        else:
            raise RuntimeError("Unknown DeepLab checkpoint format.")

        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        model.to(device).eval()
        return model

    # -------------------------
    # Main Callback
    # -------------------------
    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None: return

        # 1. Driving Inference (DeepLab) [기존 로직]
        try:
            v_base, w_base, seg_mask, seg_overlay = self.run_driving_inference(image_bgr)
        except Exception as e:
            self.get_logger().error(f"Driving Inference Error: {e}")
            return

        # Gain 적용
        v_final = v_base * self.LINEAR_GAIN
        w_final = w_base * self.STEERING_GAIN

        # 2. Obstacle Detection (YOLO) [추가된 로직]
        try:
            # YOLO는 원본 이미지를 사용 (또는 리사이즈된 이미지)
            is_obstacle, yolo_vis_img = self.run_yolo_inference(image_bgr, seg_overlay)
            
            if is_obstacle:
                # 🛑 장애물 감지 시 제어 개입 (Override)
                # 예: 즉시 정지. (필요시 감속 로직으로 변경 가능)
                self.get_logger().warn("🚨 OBSTACLE DETECTED! STOPPING! 🚨")
                v_final = 0.0
                # w_final = 0.0 # 조향도 멈출지, 회피할지 결정 (일단 정지)
        except Exception as e:
            self.get_logger().error(f"YOLO Inference Error: {e}")
            yolo_vis_img = seg_overlay # 에러나면 기존 오버레이 사용

        # 3. Safety Clamp
        if self.CLAMP_LINEAR:
            v_final = float(np.clip(v_final, self.LINEAR_MIN, self.LINEAR_MAX))
        w_final = float(np.clip(w_final, self.ANGULAR_MIN, self.ANGULAR_MAX))

        # 4. Publish Command
        twist = Twist()
        twist.linear.x = float(v_final)
        twist.angular.z = float(w_final)
        self.cmd_pub.publish(twist)

        # 5. Publish Visualization (Seg + YOLO)
        if self.PUB_SEG and yolo_vis_img is not None:
            # yolo_vis_img는 DeepLab 오버레이 위에 YOLO 박스가 그려진 상태
            vis_msg = self.bridge.cv2_to_imgmsg(yolo_vis_img, encoding='bgr8')
            vis_msg.header = msg.header
            self.overlay_pub.publish(vis_msg)

        # Log
        self._frame_count += 1
        if self.LOG_EVERY_N > 0 and (self._frame_count % self.LOG_EVERY_N == 0):
            status = "STOP" if is_obstacle else "GO"
            self.get_logger().info(f"[{status}] v={v_final:.3f}, w={w_final:.3f} | Base v={v_base:.3f}")

    # -------------------------
    # Logic: DeepLab (Driving)
    # -------------------------
    def run_driving_inference(self, image_bgr):
        # Preprocess
        img = image_bgr[self.CROP_HEIGHT:480, :]
        img_h, img_w = img.shape[:2] # for resize overlay back later
        
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_in, (self.OUT_W, self.OUT_H), interpolation=cv2.INTER_LINEAR)
        img_in = img_in.astype(np.float32) / 255.0
        img_in = (img_in - self.mean) / self.std
        x = torch.from_numpy(img_in).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            out = self.driving_model(x, return_seg=True)
        
        # Output Parsing (User's logic)
        reg, seg = self.split_outputs(out)
        v = float(reg[0, 0].item())
        w = float(reg[0, 1].item())

        # Segmentation Overlay 생성
        seg_overlay_bgr = None
        seg_mask = None
        if seg is not None and self.PUB_SEG:
            logits = seg[0].detach().cpu().float()
            mask = torch.argmax(logits, dim=0).numpy().astype(np.uint8)
            seg_mask = mask
            
            # Create color mask
            color_mask = self.colorize_mask(mask) # (OUT_H, OUT_W, 3)
            
            # Resize base image to model output size for blending
            base_small = cv2.resize(img, (self.OUT_W, self.OUT_H))
            
            # Blend
            seg_overlay_small = cv2.addWeighted(base_small, 1.0, color_mask, self.SEG_ALPHA, 0.0)
            
            # 오버레이를 다시 원본 크기(혹은 보기 좋은 크기)로 키울 수도 있지만
            # 여기서는 편의상 모델 출력 크기(320x192)를 그대로 사용하거나 원본 비율로 복원
            # YOLO 시각화를 위해 원본 크기인 image_bgr에 맞춰주면 좋지만, 
            # 성능상 DeepLab 출력 크기 베이스로 YOLO 결과를 그리는 게 나음.
            seg_overlay_bgr = seg_overlay_small 

        return v, w, seg_mask, seg_overlay_bgr

    # -------------------------
    # Logic: YOLO (Obstacle)
    # -------------------------
    def run_yolo_inference(self, image_bgr, overlay_img):
        """
        image_bgr: 원본 이미지 (YOLO 감지용)
        overlay_img: DeepLab 결과가 그려진 작은 이미지 (시각화용)
        """
        # YOLO Inference
        # verbose=False로 로그 줄임
        results = self.yolo_model(image_bgr, conf=self.YOLO_CONF, verbose=False)
        result = results[0]

        is_obstacle_detected = False
        img_h, img_w = image_bgr.shape[:2]
        total_area = img_h * img_w

        # 시각화용 이미지 (DeepLab 오버레이가 없으면 원본 리사이즈 사용)
        if overlay_img is None:
            vis_img = cv2.resize(image_bgr, (self.OUT_W, self.OUT_H))
        else:
            vis_img = overlay_img.copy()

        # 스케일 비율 계산 (YOLO좌표(원본) -> 시각화이미지(320x192))
        vis_h, vis_w = vis_img.shape[:2]
        scale_x = vis_w / img_w
        scale_y = vis_h / img_h

        # 감지된 객체 루프
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 관심 클래스(사람 등)인지 확인
            if cls_id in self.STOP_CLASSES:
                # Bounding Box 좌표 (xyxy)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                
                # 거리 판단: 박스 면적이 화면 전체의 N% 이상이면 "가깝다"고 판단
                area_ratio = box_area / total_area
                
                # 시각화 (박스 그리기)
                # 좌표를 vis_img 크기에 맞게 변환
                vx1, vy1 = int(x1 * scale_x), int(y1 * scale_y)
                vx2, vy2 = int(x2 * scale_x), int(y2 * scale_y)

                # 위험하면 빨간색, 멀면 노란색
                if area_ratio >= self.STOP_AREA_RATIO:
                    color = (0, 0, 255) # Red (STOP)
                    is_obstacle_detected = True
                    label = f"STOP! {area_ratio:.2f}"
                else:
                    color = (0, 255, 255) # Yellow (Warning)
                    label = f"Obj {area_ratio:.2f}"

                cv2.rectangle(vis_img, (vx1, vy1), (vx2, vy2), color, 2)
                cv2.putText(vis_img, label, (vx1, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return is_obstacle_detected, vis_img

    # -------------------------
    # Helper Methods
    # -------------------------
    def split_outputs(self, out):
        # (기존 코드와 동일) Reg/Seg 분리 로직
        reg, seg = None, None
        if isinstance(out, (tuple, list)):
            for t in out:
                if torch.is_tensor(t):
                    if t.dim() == 2: reg = t
                    elif t.dim() == 4: seg = t
            return reg, seg
        if torch.is_tensor(out):
            if out.dim() == 2: reg = out
            elif out.dim() == 4: seg = out
        return reg, seg

    def colorize_mask(self, mask_u8):
        # (기존 코드와 동일) Segmentation 컬러링
        h, w = mask_u8.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        palette = [
            (0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)
        ]
        for cls_id, bgr in enumerate(palette):
            out[mask_u8 == cls_id] = bgr
        return out


def main(args=None):
    print("=== ENTER main() ===")
    rclpy.init(args=args)
    node = DeepLabYoloInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()