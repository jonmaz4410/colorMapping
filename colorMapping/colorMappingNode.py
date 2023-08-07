#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import numpy as np
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs_py import point_cloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
import cv2

#Not Currently being used
# import spatialmath as sm
# from scipy.spatial.transform import Rotation

# import timeit

# import open3d as o3d
# from geometry_msgs.msg import Point


########################################################################################################################
### Name:   Jonathan Mazurkiewicz
### Topic:  USV Perception Stack Design
### Task:   Read data from an image and pointcloud message, publish new point cloud with color
### Dates:  June 2023 - August 2023
### 
### Notes:  1. Uses ROS2 / rclpy to get information. Uses ZED 2i Stereovision camera and Velodyne M1600 Lidar
###
###         2. There is a choice between the image_geometry model and the openCV model for projection. Comment out
###         whichever one you aren't using! If you select openCV, follow comments in cameraInfo callback.
###
###         3. Takes advantage of Numpy arrays, Boolean indexing, and slicing for near-optimal speed
###
###         4. Messages between pointcloud and image are roughly 0.02 to 0.28 seconds apart. This can cause problems with the new point cloud
###         Adjust the argument in ApproximateTimeSynchronizer as needed.
###
###         5. Code is modular. All of the work is done inside of functions except for printing time difference. See member
###         functions for analysis.
###
###         6. Currently, extrinsic calibration has not been performed between Lidar and Zed. There exists another node
###         in this package called extrinsicCalibrationNode.py. To perform calibration, run this node while capturing data
###         of a checkerboard simultaneously from various angles, It will only need to be done one time. Then, take the translation
###         vector and pass it to the manual_translation() function as dx, dy, and dz. Currently, there is not a function to handle
###         the rotation matrix or distortion coefficients with pinholeCamera() model. If there is a rotation, I recommend uncommenting 
###         scipy package above and using functions from there OR using openCV model which takes rotation matrix as an argument.
###
###         7. This code subscribes to an already rectified image for improved projection accuracy (dist coeff. (0,0,0)).
###
###         8. filter_by_angle() and is_within_bounds() are meant to do the same job: get rid of projected points that would cause
###         errors. For speed, consider only using one or the other. filter_by_angle gets rid of all points that lie outside of the 
###         specified angle and provides 100% correct filtration when using an angle of 50 degrees. is_within_bounds() will always
###         catch the projected points that are incorrect, but may be more computationally intensive.
###         
###         9. In the project3dtoPixel() function, I am passing x, y, and z in a different order to account for the rotation
###         from the world frame to the optical frame. If a rotation matrix is used, for example, from the extrinsic calibration node,
###         it may not work properly as it is. 
###
###         10. The openCV model should be superior to the pinholeCamera model. It does not require any for loops and is able to process
###         the entire image at once. For future development, I would recommend to move forward with openCV

### Future: 1. ZED SDK offers a way to provide time alignment with external sensors in its interface. This could provide a more
###         elegant and accurate way to align the messages from point cloud and image.
###
###         2. During abrupt jerky motions, rgb values are not being grabbed correctly. The point cloud is being published but there
###         is an offset of the rgb values that has not been fixed yet.  
########################################################################################################################

class colorMappingNode(Node):

    def __init__(self):

        super().__init__("color_mapping")
        self.get_logger().info("Node has been started")

        #####################################################################################
        # Declarations
        #####################################################################################
        self.camera_matrix = None
        # Set flags to synchronize data
        self.need_info = True
        self.need_image = True

        # Declare base image size for later calculations
        self.image_height = 512
        self.image_width = 896

        # CV Bridge to be used later for image processing
        self.bridge = CvBridge()

        # Pinhole Camera model to perform 3d -> 2d projection
        self.model = PinholeCameraModel()

        #Declare the fields for the published point cloud at the end. Constant
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset= 8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]


        #When doing time sync, the publisher requires a valid QoS profile. Without this,
        #pointcloud was not able to publish at all.
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20  # Set the depth to the desired value
        )

        #####################################################################################
        # Subscriptions
        #####################################################################################

        # Subscriber to gather CameraInfo parameters from left camera
        self.sub_cam_info_left = self.create_subscription(CameraInfo,
                                                          'zed2i/zed_node/left/camera_info',
                                                          self.callback_info_left,
                                                          10)
        # Subscriber to an RGB image
        self.sub_image_rgb = Subscriber(self, Image, '/zed2i/zed_node/rgb/image_rect_color', qos_profile=qos_profile_sensor_data)

        # Subscriber to a point cloud
        self.sub_pcl = Subscriber(self, PointCloud2, '/lidar_0/m1600/pcl2', qos_profile=qos_profile_sensor_data)

        #Synchronize PCL + RGB. ARGS ([subscriptions], queue_size, time_window)
        self.time_sync = ApproximateTimeSynchronizer([self.sub_image_rgb, self.sub_pcl], 30, .08)
        self.time_sync.registerCallback(self.callback_sync)

        #####################################################################################
        # Publishers
        #####################################################################################
        self.pub_pcl = self.create_publisher(
            PointCloud2, '/rgb_pointcloud', qos_profile)

        self.get_logger().info("color_mapping node has initialized successfully")

    #####################################################################################
    # Callback functions
    #####################################################################################

    # Camera parameter callback -- only to be done one time upon startup###
    # Gathers camera intrinsics from the ZED
    def callback_info_left(self, info):
        if self.need_info:
            self.get_logger().info('Inside camera info callback')

            # Gather camera intrinsic parameters and save to pinhole model.
            self.model.fromCameraInfo(info)
            # Reset need_info, gathering camera info only needs to be done once.
            self.need_info = False

            # ONLY NEEDED IF NOT USING PINHOLE MODEL
            # Get focal point and principal point coordinates from cameraInfo, then turn into matrix
            self.fx = info.k[0]
            self.fy = info.k[4]
            self.cx = info.k[2]
            self.cy = info.k[5]
            self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                           [0, self.fy, self.cy],
                                           [0,       0,      1]])

    ##########################################################################################
    # Pointcloud and image synchronized callback + processing

    def callback_sync(self, image_msg, pcl2_msg):

        self.get_logger().info('Entering image callback')

        ###Retrieving data from image and pointcloud messages with exception handling
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgra8')
            self.get_logger().info('Image retrieved successfully!')            
            self.need_image = False
        except CvBridgeError as e:
            self.get_logger().warn("Error converting between ROS Image and OpenCV image:" .format(e))

        self.get_logger().info("Entering pointcloud callback")

        
        try:
            # Extract x, y, and z from structured array. Optional: ring, intensity
            xyz = point_cloud2.read_points(cloud=pcl2_msg,
                                           field_names=('x', 'y', 'z'),
                                           reshape_organized_cloud=True)
            x = xyz['x']
            y = xyz['y']
            z = xyz['z']
            self.get_logger().info('Pointcloud retrieved successfully!')            

        except Exception as e:
            self.get_logger().warn("Failed to retrieve pointcloud: {}".format(e))
            return
        
        #Printing time difference between messages
        self.image_timestamp = image_msg.header.stamp.sec + (image_msg.header.stamp.nanosec * 1e-9)
        self.pcl2_timestamp = pcl2_msg.header.stamp.sec + (pcl2_msg.header.stamp.nanosec * 1e-9)
        self.time_difference = abs(self.image_timestamp - self.pcl2_timestamp)
        self.get_logger().info(f"Time difference between messages: {self.time_difference: .5f} seconds")

        # Stack the columns to create array of shape (n, 3)
        lidar_points = np.column_stack((x, y, z))

        #############################################################################################################
        # # # USING PINHOLECAMERA MODEL -- COMMENT OUT THIS BLOCK IF USING OPENCV INSTEAD

        # # Manual translation of point cloud coordinates to attempt extrinsic calibration
        # # dx, dy, and dz are the amounts you would like to translate pointcloud (in meters)
        # translated_lidar_points = np.empty_like(lidar_points.shape)
        # translated_lidar_points = self.manual_translation(lidar_points, dx = 0, dy = -.11, dz = 0)
        
        # # Filter points based on angle (Currently 50 degrees filters all incorrect points)
        # translated_lidar_points = self.filter_by_angle(translated_lidar_points)  # Optional param: limiting_angle

        
        # if not self.need_info:
        #     # Obtain u,v pixel coordinates from projection
        #     pixel_coordinates = self.project3dtoPixel(translated_lidar_points)
        #     #Filter out lidar points that produce incorrect pixel values
        #     valid_pixel_coordinates, valid_lidar_points = self.is_within_bounds(pixel_coordinates, translated_lidar_points)
        #     breakpoint()
        #     #Print the percentage of valid points
        #     self.filter_percentage(pixel_coordinates.shape[0], valid_pixel_coordinates.shape[0])
        #     if not self.need_image:
        #         # Grab rgb values from all pixel coordinates u,v
        #         rgb_values = self.grab_pixel_rgb(valid_pixel_coordinates)
        #         # Publish new point cloud
        #         self.publish_point_cloud(valid_lidar_points, rgb_values)
        
        #############################################################################################################
        # USING OPENCV MODEL -- COMMENT OUT THIS BLOCK IF USING PINHOLECAMERA MODEL INSTEAD

        if not self.need_info:
        
            #Filter by angle. Currently 50 degrees filters all incorrect points. Consider only using is_within_bounds()
            lidar_points = self.filter_by_angle(lidar_points)

            pixel_coordinates = self.openCV_projectPoints(lidar_points)
            valid_pixel_coordinates, valid_lidar_points = self.is_within_bounds(pixel_coordinates, lidar_points)
            #Print the percentage of valid points
            self.filter_percentage(pixel_coordinates.shape[0], valid_pixel_coordinates.shape[0])
            if not self.need_image:
                # Grab rgb values from all pixel coordinates u,v
                rgb_values = self.grab_pixel_rgb(valid_pixel_coordinates)
                # Publish new point cloud
                self.publish_point_cloud(valid_lidar_points, rgb_values)


########################################################################################
# Supplementary functions
########################################################################################


#1. Filter out pixel values that are not in bounds of the image height and width
    def is_within_bounds(self, pixel_coords, lidar_points):

        # Mask = Pixel coords that are negative or greater than max width or length
        mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < self.image_width - 1) & \
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < self.image_height - 1)
        
        valid_pixel_coords = pixel_coords[mask]
        valid_lidar_points = lidar_points[mask]

        
        return valid_pixel_coords, valid_lidar_points

#2. Print percentage of points that passed filtering
    def filter_percentage(self, original_size, filtered_size):
        percentage = np.around((filtered_size / original_size * 100), 2)
        self.get_logger().info(f'Percentage of lidar points remaining after being filtered: {percentage:.2f}%')




#3. Filter points based upon the given angles of Lidar to match ZED (roughly)
    # To avoid extra data processing, I noticed all of my points out of bounds were due to
    # the lidar capturing more data horizontally than the image.
    # This is one method of filtering that should be less computationally expensive.
    def filter_by_angle(self, lidar_points, limiting_angle=50):
        angle_array_w = np.arctan2(lidar_points[:, 1], lidar_points[:, 0])
        # This was calculated experimentally
        acceptable_radians = np.radians(limiting_angle)
        valid_indices = np.where(np.logical_and(
            angle_array_w >= -acceptable_radians, angle_array_w <= acceptable_radians))
        valid_lidar_points = lidar_points[valid_indices]
        return valid_lidar_points

#4. Input 3D point. Output rectified pixel coordinates (u, v)
    def project3dtoPixel(self, lidar_points):
        valid_pixel_coordinates = []
        for p in lidar_points:
            # To perform the matrix rotation, the columns are input in a different order to save processing power.
            # Apparently works the same way. (Instead of passing x, y, z, I pass (-y, -z, x))
            pixel_coordinates = np.array(
                self.model.project3dToPixel((-p[1], -p[2], p[0])))
            valid_pixel_coordinates.append(pixel_coordinates)

        # Convert the list of pixel coordinates to a NumPy array as integer
        valid_pixel_coordinates = np.around(
            valid_pixel_coordinates).astype(int)

        return valid_pixel_coordinates

#5. Input pixel coordinate. Output rgb value at that point 
    def grab_pixel_rgb(self, pixel_coords):
        # cv_image comes in the shape (height, width, 4)
        # Slicing permits efficient way to grab all pixel values at once
        bgr_values = self.cv_image[pixel_coords[:, 1], pixel_coords[:, 0], 0:3]

        # Convert BGR values to UINT32 RGB values
        r = np.uint32(bgr_values[:, 2])
        g = np.uint32(bgr_values[:, 1])
        b = np.uint32(bgr_values[:, 0])

        # Bitwise operations to combine r, g, b into one value of type UINT32
        rgb_values = (r << 16) | (g<< 8) | b

        return rgb_values

#6. Input original 3D pointcloud and amount to translate in x, y, and/or z. Output translated pointcloud
    def manual_translation(self, lidar_points, dx, dy, dz):

        T = np.array    ([[1, 0, 0, dx],
                          [0, 1, 0, dy],
                          [0, 0, 1, dz],
                          [0, 0, 0, 1]])
        
        lidar_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
        translated_lidar_points = np.dot(lidar_points, T.T)
        translated_lidar_points = translated_lidar_points[:, :3] / translated_lidar_points[:, 3:]
        return translated_lidar_points

#7. Publish pointcloud
    def publish_point_cloud(self, xyz_values, rgb_values):

        num_points = len(xyz_values)

        # Create structured array with datatypes that we want for rgb image and labels
        points = np.empty(num_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])

        #Assign values to points message with proper label
        points['x'] = xyz_values[:, 0]
        points['y'] = xyz_values[:, 1]
        points['z'] = xyz_values[:, 2]
        points['rgb'] = rgb_values

        # Create the PointCloud2 message
        # Currently, I have selected the left camera frame. This may need to change in the future.
        msg = PointCloud2()
        msg.header.frame_id = 'zed2i_left_camera_frame'
        msg.height = 1
        msg.width = num_points
        msg.fields = self.fields
        msg.is_bigendian = False
        msg.point_step = 16  # 3 (xyz) + 1 (rgb)
        msg.row_step = msg.point_step * num_points
        msg.data = points.tobytes()
        msg.is_dense = True

        # Publish the message
        self.pub_pcl.publish(msg)
        self.get_logger().info('Published point cloud')

        return self

#8. openCV package for projection

    def openCV_projectPoints(self, lidar_points):

        #Rotate the lidar points from the world to optical frame by passing -y, -z, x instead of x, y, z    
        rotated_lidar_points = np.empty_like(lidar_points)
        rotated_lidar_points = np.column_stack((-lidar_points[:, 1], -lidar_points[:, 2], lidar_points[:, 0]))

        #Rotation vector declared as empty float array of shape 3x1. (Can also be a 3x3 matrix). Each value is an angle in degrees to rotate about that axis (x, y, z). Update with values from calibration.
        rotation_vector = np.array([0.0, 0.0, 0.0])

        #Translation vector declared as empty float array of shape 3x1. Update with values from calibration.
        translation_vector = np.array([.12, 0, 0])

        # Obtain u,v pixel coordinates from projection
        # Currently, since using a rectified image, there will not be distortion coefficients.
        pixel_coordinates, _ = cv2.projectPoints(objectPoints=rotated_lidar_points,
                                                    rvec=rotation_vector,
                                                    tvec=translation_vector,
                                                    cameraMatrix=self.camera_matrix,
                                                    distCoeffs=None
                                                    )
        
        #Squeeze dimensionality to obtain data in useable format
        pixel_coordinates = pixel_coordinates.squeeze()
        #Round and convert to int
        pixel_coordinates = np.around(pixel_coordinates).astype(int)
        return pixel_coordinates


def main(args=None):
    rclpy.init(args=args)
    node = colorMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Handle keyboard interrupt (Ctrl+C)
    finally:
        # Clean up resources and shutdown ROS 2
        node.destroy_node()
        rclpy.shutdown()