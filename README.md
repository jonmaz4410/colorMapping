# colorMapping

This is a package built with rclpy / colcon / ROS2. Its purpose is to communicate between lidar and stereovision camera and form a perception stack.

In my profile, there exists a dockerfile (go back one directory to find it). This will include all of the required dependencies and is the preferred method for running this code.
However, there were no overtly complicated steps for installing dependencies. If you do not want to use docker, install the packages used onto your system and you should be good to go.

If you clone this code, check to make sure that the nodes have executable privelege. FROM CLI >> chmod +x <filename.py>

To run, enter FROM CLI >> ros2 run colorMapping color_mapping OR
                       >> ros2 run colorMapping extrinsic_calibration

The new pointcloud is published to the ZED optical frame. Consider experimenting with changing this frame to the new lidar_0 frame that was created by another team member or another frame for proper sensor fusion. My approach is a temporary solution.

Be very careful when messing with the QoS Profiles or if you change the QoS profile of subscribers or publishers. Since I am using ApproximateTimeSynchronizer(), if the QoS profiles do not match up,
the subscriptions will never occur and will not give you an error code as to why. They just won't run. Consider having a secondary version of code that does not use approximateTimeSynchronizer() or TimeSynchronizer() and uses the a regular subscriber for troubleshooting.

Currently, the subscribers are meant to filter out messages whose timestamps are not within .08 seconds of each other. This can be changed experimentally. <b>In the future, consider implementing the ZED SDK features for sensor fusion with external lidar. Visit <a>https://www.stereolabs.com/docs/sensors/time-synchronization/</a> for more info. </b> I believe this is the real way to create true sensor fusion between lidar and camera.

In addition, extrinsic calibration still needs to be performed. See comments inside extrinsic_calibration.py for more details.

There are in-depth comments inside of the code itself. As of August 15, 2023:

color_mapping.py

      1. is resource efficient. Boolean indexing, slicing, and other techniques prevent the need for any for loops / list comprehensions.
      
      2. accurately grabs color from pixels, reads from a point cloud, and forms correspondance between pixel and pointcloud. Then, publishes new cloud with rgb.
      
      3. still needs proper extrinsic calibration for even better results. Currently, a manual translation is being performed to better line up data.
      
      4. is complete.

extrinsic_calibration.py

      1. is not complete.
      
      2. correctly finds checkerboard in an image
      
      3. does not incorporate lidar / point cloud yet.

If you have any questions, please send me an email at jonmaz4410@gmail.com and I will be happy to assist further. Good luck!

