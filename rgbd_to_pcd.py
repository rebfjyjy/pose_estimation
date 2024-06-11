import os
import open3d as o3d
import numpy as np
import cv2

class PointCloudConverter:
    def __init__(self, rgb_path=None, depth_path=None, rgb_img=None, depth_img=None, fx=615.7490234375, fy=615.5992431640625, cx=323.75396728515625, cy=237.6147003173828):
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

        if rgb_path is not None and depth_path is not None:
            self.rgb_path = rgb_path
            self.depth_path = depth_path
            self.rgb_image = o3d.io.read_image(self.rgb_path)
            self.depth_image = o3d.io.read_image(self.depth_path)
        
        # self.rgb_image = o3d.geometry.Image(rgb_img)
        # self.depth_image = o3d.geometry.Image(depth_img)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        print(self.fx, self.fy, self.cx, self.cy)

        # Load RGB and Depth images

        # rgb = cv2.imread(self.rgb_path)
        # cv2.imshow('rgb', rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        # o3d.visualization.draw_geometries(self.rgb_image)

        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(self.rgb_image, self.depth_image)
        
        print(self.rgbd_image)
        dimensions = self.rgbd_image.dimension # what is this being used for
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

    def save_point_cloud(self, point_cloud, idx=0):
        """
        Save the point cloud to a file.

        Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud to save.
        filename (str): The path to the file where the point cloud should be saved.
        """
        filename = f"./data/6.11/point_cloud_{idx}.ply"
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
idx = 3
rgb_path = f"./data/6.11/color{idx}.png"
depth_path = f"./data/6.11/depth{idx}.png"
# converter = PointCloudConverter(rgb_path, depth_path, fx=615.7490234375, fy=615.5992431640625, cx=323.75396728515625, cy=237.6147003173828)
# Intrinsic of "Infrared 2" / 640x480 / {Y8}
#   Width:      	640
#   Height:     	480
#   PPX:        	321.350402832031
#   PPY:        	240.833572387695
#   Fx:         	599.930725097656
#   Fy:         	599.930725097656
#   Distortion: 	Brown Conrady
#   Coeffs:     	0  	0  	0  	0  	0  
#   FOV (deg):  	56.15 x 43.61
converter = PointCloudConverter(rgb_path, depth_path, fx=599.930725097656, fy=599.930725097656, cx=321.350402832031, cy=240.833572387695)

point_cloud = converter.create_point_cloud(converter.rgbd_image)
converter.visualize_point_cloud(point_cloud)
converter.save_point_cloud(point_cloud, idx)
