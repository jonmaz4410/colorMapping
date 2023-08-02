#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2
from rclpy.qos import qos_profile_sensor_data ,QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy


class ExtrinsicCalibrationNode(Node):
    def __init__(self):
        super().__init__("extrinsic_calibration")
        self.get_logger().info("Extrinsic Calibration Node has started")

        self.camera_matrix = None
        self.distortion_coeffs = None
        self.bridge = CvBridge()
        self.image_height = 512
        self.image_width = 896
        self.board_size = (6, 9)  # Change this to your actual checkerboard size
        self.square_size = 0.04  # Change this to the actual square size in meters
        self.calibration_data = []  # List to store calibration data
        self.calibration_complete = False

        qos_profile_sync = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10
        )       

        # Subscriber to gather CameraInfo parameters from left camera
        self.sub_cam_info_left = self.create_subscription(
            CameraInfo,
            '/zed2i/zed_node/left/camera_info',
            self.callback_info_left,
            10
        )

        # Subscribers to raw RGB image and point cloud
        self.sub_image_raw = Subscriber(self,
                                        Image,
                                        '/zed2i/zed_node/left_raw/image_raw_gray',
                                        qos_profile=qos_profile_sync)
        self.sub_pcl = Subscriber(self,
                                  PointCloud2,
                                  '/lidar_0/m1600/pcl2',
                                  qos_profile=qos_profile_sync)

        # Synchronize RGB image and point cloud
        self.time_sync = ApproximateTimeSynchronizer([self.sub_image_raw, self.sub_pcl],
                                                      30,
                                                      1)
        
        self.time_sync.registerCallback(self.callback_time_sync)

        self.calibration_complete = False

    def callback_info_left(self, info):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(info.k).reshape(3, 3)
            self.distortion_coeffs = np.array(info.d)

    def callback_time_sync(self, image_msg, pcl2_msg):
        if not self.calibration_complete:
            
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgra8')

            except Exception as e:
                self.get_logger().error(
                    "Error converting between ROS Image and OpenCV image: {}".format(e))
                return

            # Detect checkerboard corners in the raw RGB image
            ret, corners = cv2.findChessboardCorners(
                cv_image, self.board_size, None)

            if ret:
                # Convert checkerboard corners to sub-pixel accuracy
                cv2.cornerSubPix(cv_image, corners, (11, 11), (-1, -1),
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                # Generate checkerboard points in 3D space (assuming z=0)
                checkerboard_points = np.zeros(
                    (self.board_size[0] * self.board_size[1], 3), np.float32)
                checkerboard_points[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
                checkerboard_points *= self.square_size

                # Store checkerboard points and corners
                self.calibration_data.append((checkerboard_points, corners))
            
            else:
                self.get_logger().info("Did not find a chessboard in this image")

            try:
                xyz = point_cloud2.read_points(
                    cloud=pcl2_msg, field_names=('x', 'y', 'z'), skip_nans=True)
                points_3d = np.array(list(xyz))
            except Exception as e:
                self.get_logger().error("Error reading point cloud data: {}".format(e))
                return

            if len(self.calibration_data) > 20:
                self.calibrate_extrinsics()
                self.calibration_complete = True

    def calibrate_extrinsics(self):
        obj_points = []
        img_points = []
        for checkerboard_points, corners in self.calibration_data:
            obj_points.append(checkerboard_points)
            img_points.append(corners)

        # Perform extrinsic calibration
        _, self.rotation_vector, self.translation_vector, _, _ = cv2.calibrateCamera(
            obj_points, img_points, (self.image_width, self.image_height), self.camera_matrix, self.distortion_coeffs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)

        # Convert rotation vector to rotation matrix
        self.rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)

        # Print out the extrinsic calibration results
        self.get_logger().info("Extrinsic Calibration Results:")
        self.get_logger().info("Rotation Matrix:")
        self.get_logger().info(self.rotation_matrix)
        self.get_logger().info("Translation Vector:")
        self.get_logger().info(self.translation_vector)
        self.get_logger().info("")

        img_points_projected, _ = cv2.projectPoints(obj_points,
                                                    self.rotation_vector,
                                                    self.translation_vector,
                                                    self.camera_matrix,
                                                    self.distortion_coeffs)

        total_error = 0
        for i in range(len(img_points)):
            error = cv2.norm(img_points[i], img_points_projected[i][0], cv2.NORM_L2) / len(img_points_projected[i][0])
            total_error += error

        mean_error = total_error / len(img_points)
        self.get_logger().info("Mean Reprojection Error: {}".format(mean_error))
        

def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicCalibrationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
