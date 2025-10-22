#!/usr/bin/env python3

import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self, kp = 2.0):
      super().__init__()
      self.kp = kp
      
    def compute_control_with_goal(self, curr: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
      heading_error = wrap_angle(goal.theta - curr.theta)
      omega_control = self.kp * heading_error 
      return TurtleBotControl(omega=omega_control)
      

def main():
  rclpy.init()
  controller = HeadingController()
  rclpy.spin(controller)
  rclpy.shutdown()

if __name__ == "__main__":
    main()