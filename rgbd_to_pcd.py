import os
import open3d as o3d
import numpy as np
import cv2

class PointCloudConverter:
    def __init__(self, rgb_path, depth_path, fx, fy, cx, cy):
        """
        Initialize the PointCloudConverter class.

        Args:
        rgb_path (str): Path to the RGB image file.
        depth_path (str): Path to the depth image file.
        fx (float): Focal length of the camera in x direction.
        fy (float): Focal length of the camera in y direction.
        cx (float): Optical center of the camera in x direction.
        cy (float): Optical center of the camera in y direction.
        """
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Load RGB and Depth images

        # rgb = cv2.imread(self.rgb_path)
        # cv2.imshow('rgb', rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.rgb_image = o3d.io.read_image(self.rgb_path)
        self.depth_image = o3d.io.read_image(self.depth_path)
        # o3d.visualization.draw_geometries(self.rgb_image)

        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(self.rgb_image, self.depth_image)
        
        print(self.rgbd_image)
        dimensions = self.rgbd_image.dimension
        # Define camera intrinsics
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

    def create_rgbd_image(self, depth_scale=1000.0, depth_trunc=3.0):
        """
        Create an RGB-D image from the loaded RGB and depth images.

        Args:
        depth_scale (float): The scale used for depth values.
        depth_trunc (float): Maximum depth value to be used.

        Returns:
        o3d.geometry.RGBDImage: The created RGB-D image.
        """
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            self.rgb_image, self.depth_image, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

    def create_point_cloud(self, rgbd_image):
        """
        Create a point cloud from an RGB-D image.

        Args:
        rgbd_image (o3d.geometry.RGBDImage): The RGB-D image to convert.

        Returns:
        o3d.geometry.PointCloud: The created point cloud.
        """
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsics)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd

    def save_point_cloud(self, point_cloud, filename="output_point_cloud.ply"):
        """
        Save the point cloud to a file.

        Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud to save.
        filename (str): The path to the file where the point cloud should be saved.
        """
        o3d.io.write_point_cloud(filename, point_cloud)

    def visualize_point_cloud(self, point_cloud):
        """
        Visualize the point cloud.

        Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud to visualize.
        """
        o3d.visualization.draw_geometries([point_cloud])

# Example of using the PointCloudConverter class
# Intrinsic camera matrix for the raw (distorted) images.
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]
rgb_path = './4_30_data/right/color1.png'

depth_path = './4_30_data/right/depth1.png'
converter = PointCloudConverter(rgb_path, depth_path, fx=615.7490234375, fy=615.5992431640625, cx=323.75396728515625, cy=237.6147003173828)
point_cloud = converter.create_point_cloud(converter.rgbd_image)
converter.visualize_point_cloud(point_cloud)
# converter.save_point_cloud(point_cloud, './4_30_data/point_cloud/right_1.ply')
