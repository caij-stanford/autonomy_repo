#!/usr/bin/env python3

import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool

class PerceptionController(BaseHeadingController):
    def __init__(self, kp = 2.0):
      super().__init__("perception_controller")
      self.declare_parameter("kp", kp)
      
      self.image_detected = False
    
      self.detected_sub = self.create_subscription(Bool, "/detector_bool", self.detected_callback, 10)
      
    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value  
    
    def detected_callback(self, msg: Bool):
      # If detect stop sign, don't turn
      if msg.data:
        self.image_detected = True 
      else: 
        self.image_detected = False
    
    def compute_control_with_goal(self, curr: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
      if not self.image_detected: 
        return TurtleBotControl(omega=0.2)
      else:
        return TurtleBotControl(omega=0.0)
      

def main():
  rclpy.init()
  controller = PerceptionController()
  rclpy.spin(controller)
  rclpy.shutdown()

if __name__ == "__main__":
    main()