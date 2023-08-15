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

########################################################################################################################
### Name:   Jonathan Mazurkiewicz
### Topic:  USV Perception Stack Design
### Task:   Perform extrinsic calibration between lidar and camera
### Dates:  June 2023 - August 2023
### 
### Notes:  1. Uses ApproximateTimeSynchronizer() to ensure timestamps match up between messages.
###
###         2. Uses a point cloud, an image, and camera intrinsic parameters
###
###         3. Adapt board size and pattern to your current set up. Exact measurements for size, count internal vertices in row, column format for pattern.
###
###         4. Uncomment cv2.imshow() to be able to visually see if the algorithm is properly finding checkerboard corners.
###
###         5. Uncomment error function at the bottom (IN PROGRESS) to test reprojection error.
###
###         6. There are many ways to perform extrinsic calibration. This is only a start. It does not incorporate the lidar yet. Consider using this code ONLY as a skeleton.
###         The eventual solution will involve much more work, finding a way to achieve correspondance between checkerboard corner in image and pointcloud, reprojecting for
###         error estimation, and getting EXACT time alignment.
########################################################################################################################


class ExtrinsicCalibrationNode(Node):
    def __init__(self):
        super().__init__("extrinsic_calibration")
        self.get_logger().info("Extrinsic Calibration Node has started")

        self.camera_matrix = None
        self.distortion_coeffs = None
        self.bridge = CvBridge()
        self.image_height = 512
        self.image_width = 896
        self.board_size = (6, 8)  # Change this to your actual checkerboard size
        self.square_size = 0.025  # Change this to the actual square size in meters
        self.calibration_data = []  # List to store calibration data
        self.calibration_complete = False

        qos_profile_sync = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20  # Set the depth to the desired value
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
                                        '/zed2i/zed_node/left/image_rect_color',
                                        qos_profile=qos_profile_sync)
        self.sub_pcl = Subscriber(self,
                                  PointCloud2,
                                  '/lidar_0/m1600/pcl2',
                                  qos_profile=qos_profile_sync)

        # Synchronize RGB image and point cloud
        self.time_sync = ApproximateTimeSynchronizer([self.sub_image_raw, self.sub_pcl],
                                                      30,
                                                      .08)
        
        self.time_sync.registerCallback(self.callback_time_sync)

        self.calibration_complete = False

    def callback_info_left(self, info):
        #Camera intrinsic matrix
        if self.camera_matrix is None:
            self.camera_matrix = np.array(info.k).reshape(3, 3)
            self.distortion_coeffs = np.array(info.d)

    def callback_time_sync(self, image_msg, pcl2_msg):
        if not self.calibration_complete:
            
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgra8')
                gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2GRAY)  # Convert to grayscale


            except Exception as e:
                self.get_logger().error(
                    "Error converting between ROS Image and OpenCV image: {}".format(e))
                return
            
            try:
                xyz = point_cloud2.read_points(
                    cloud=pcl2_msg, field_names=('x', 'y', 'z'), skip_nans=True, reshape_organized_cloud=True)
                x = xyz['x']
                y = xyz['y']
                z = xyz['z']

                # lidar_points stores the point cloud data in an organized fashion. rotated_lidar_points performs the rotation from the world to optical frame for alignment.
                # Consider a better solution in the future.
                lidar_points = np.column_stack((x, y, z))
                rotated_lidar_points = np.column_stack((-y, -z, x))

            except Exception as e:
                self.get_logger().error("Error reading point cloud data: {}".format(e))
                return
            
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK


            # Detect checkerboard corners in the raw RGB image
            ret, corners = cv2.findChessboardCorners(
                gray_image, self.board_size, flags)

            if ret:
                # Convert checkerboard corners to sub-pixel accuracy
                self.get_logger().info("Found chessboard!")
                # When I changed subpixel window, I began to be able to find the right checkerboard corners. It needed to be smaller since our image is low resolution. If a higher
                # resolution image is used, consider experimenting with the parameters in the function below.
                cv2.cornerSubPix(gray_image, corners, (3, 3), (-1,-1), 
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Draw the corners on the image
                cv2.drawChessboardCorners(cv_image, self.board_size, corners, ret)

            # Display the image -- USED TO CHECK IF CORRECT CORNERS WERE FOUND
                # cv2.imshow("Checkerboard Corners", cv_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                checkerboard_points = np.zeros(
                    (self.board_size[0] * self.board_size[1], 3), np.float32)
                checkerboard_points[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
                checkerboard_points *= self.square_size

                # checkerboard_points = self.transform_corners_from_camera_to_lidar(corners, rotated_lidar_points)

                # # Store checkerboard points and corners
                # self.calibration_data.append((checkerboard_points, corners))
                # corners_squeezed = corners.squeeze()
                # cv_image_corners = cv_image.copy()
                # for corner in corners_squeezed:
                #     cv2.circle(cv_image_corners, tuple(corner), 5, (0, 0, 255), -1)

                #     lidar_corners = np.array([point[:2] for point in checkerboard_points], dtype=np.int32)
                #     lidar_image = np.zeros_like(cv_image)
                # for corner in lidar_corners:
                #     cv2.circle(lidar_image, tuple(corner), 5, (0, 255, 0), -1)

                # # Display the images
                #     cv2.imshow("Camera Corners", cv_image_corners)
                #     cv2.imshow("LiDAR Corners", lidar_image)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                # Store checkerboard points and corners
                self.calibration_data.append((checkerboard_points, corners))
                
            else:
                self.get_logger().info("Did not find a chessboard in this image")



            if len(self.calibration_data) > 20:
                self.calibrate_extrinsics()
                self.calibration_complete = True


    #Experimental function. Meant to find checkerboard corners in the point cloud using Euclidean distance. Have not tested yet.
    def transform_corners_from_camera_to_lidar(self, corners, lidar_points):
        transformed_corners = []
        for corner in corners:
            closest_point_idx = np.argmin(np.linalg.norm(lidar_points[:, :2] - corner[0], axis=1))
            transformed_corners.append(lidar_points[closest_point_idx])
        return transformed_corners


    def calibrate_extrinsics(self):
        obj_points = []
        img_points = []
        for checkerboard_points, corners in self.calibration_data:
            obj_points.append(checkerboard_points)
            img_points.append(corners)

        # Perform extrinsic calibration using solvePnP
        # Another option is to use calibrateCamera() from openCV. This will also estimate camera intrinsics. Experiment with both.

        _, self.rotation_vector, self.translation_vector = cv2.solvePnP(
        obj_points[0], img_points[0], self.camera_matrix, self.distortion_coeffs)


        # Convert rotation vector to rotation matrix
        self.rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)

        # Print out the extrinsic calibration results
        self.get_logger().info("Extrinsic Calibration Results:")
        self.get_logger().info("Rotation Matrix:")
        print(self.rotation_matrix)
        self.get_logger().info("Translation Vector:")
        print(self.translation_vector)
        self.get_logger().info("")

        ### Error function. Consider placing into a member function.

        # img_points_projected, _ = cv2.projectPoints(obj_points,
        #                                             self.rotation_vector,
        #                                             self.translation_vector,
        #                                             self.camera_matrix,
        #                                             self.distortion_coeffs)

        # total_error = 0
        # for i in range(len(img_points)):
        #     error = cv2.norm(img_points[i], img_points_projected[i][0], cv2.NORM_L2) / len(img_points_projected[i][0])
        #     total_error += error

        # mean_error = total_error / len(img_points)
        # self.get_logger().info("Mean Reprojection Error: {}".format(mean_error))
        

def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicCalibrationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
