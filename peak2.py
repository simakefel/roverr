#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Twist
import numpy as np
import math

class AdaptivePeakFollower(Node):
    def __init__(self):
        super().__init__('adaptive_peak_follower')

        # Parameters
        self.adaptive_threshold = 0.1
        self.min_threshold = 0.05
        self.max_threshold = 0.3
        self.climb_increment = 0.02

        self.current_height = 0.0
        self.last_height = 0.0
        self.consecutive_climbs = 0
        self.target_position = None

        self.scan_radius = 3.0
        self.target_reached_threshold = 0.25

        self.create_subscription(
            PointCloud2,
            '/zed2/point_cloud/cloud_registered',
            self.pc_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("AdaptivePeakFollower node initialized.")

    def pc_callback(self, msg):
        try:
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if not points:
                return

            height_points = [p for p in points if abs(p[0]) < 0.5 and abs(p[2]) < 0.5]
            if height_points:
                self.current_height = min(p[1] for p in height_points)

                if self.current_height > self.last_height + 0.01:
                    self.consecutive_climbs += 1
                    self.adaptive_threshold = min(
                        self.max_threshold,
                        self.min_threshold + self.consecutive_climbs * self.climb_increment
                    )
                else:
                    self.consecutive_climbs = max(0, self.consecutive_climbs - 1)

                self.last_height = self.current_height

            filtered = [p for p in points if (
                math.sqrt(p[0]**2 + p[2]**2) <= self.scan_radius and
                p[1] > self.current_height + self.adaptive_threshold
            )]

            if filtered:
                target = min(filtered, key=lambda p: math.sqrt(p[0]**2 + p[2]**2))
                self.target_position = np.array([target[0], target[1], target[2]])
                self.get_logger().info(
                    f"New target: x={target[0]:.2f}, y={target[1]:.2f}, z={target[2]:.2f}, threshold={self.adaptive_threshold:.2f}"
                )

        except Exception as e:
            self.get_logger().error(f"Point cloud processing error: {str(e)}")

    def control_loop(self):
        cmd = Twist()

        if self.target_position is None:
            cmd.angular.z = 0.3
            self.cmd_pub.publish(cmd)
            return

        dx = self.target_position[0]
        dz = self.target_position[2]
        dist = math.sqrt(dx**2 + dz**2)
        angle = math.atan2(dx, dz)

        if dist < self.target_reached_threshold:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info(f"Target reached at height: {self.current_height:.2f} m")
            self.target_position = None
        else:
            cmd.linear.x = 0.2 * min(dist, 1.0)
            cmd.angular.z = 0.5 * angle

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = AdaptivePeakFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
