import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import re

class PoseEstimation:
    def __init__(self, pcd_path, camera_intrinsic):
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        self.fx, self.fy, self.cx, self.cy = camera_intrinsic
        # cv2.imshow('image', self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    def create_rgbd_from_point_cloud(self, height, width, img_path):
        # Load point cloud
        self.original_points = np.full((height, width, 3), np.nan, dtype=np.float32)  # Initialize with NaNs
        # Get points and colors from point cloud
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)
        
        # Create blank images
        depth_image = np.zeros((height, width), dtype=np.float32)
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Parameters for projecting 3D points to 2D plane
        
        for point, color in zip(points, colors):
            x, y, z = point
            # Project 3D point to 2D plane 
            u = int(self.fx * x / z + self.cx)
            v = int(self.fy * y / z + self.cy)
            u = width - u - 1
            if 0 <= u < width and 0 <= v < height:
                
                depth_image[v, u] = z
                # print(depth_image[v, u])
                rgb_image[v, u, :] = (color * 255).astype(np.uint8)
                self.original_points[v, u] = [x, y, z]
        # Normalize depth image for visualization 
        depth_image_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Save images
        cv2.imwrite(f'{img_path}/rgb_image.png', self.image)
        cv2.imwrite(f'{img_path}/depth_image.png', depth_image_norm)

    def detect_landmarks(self):
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find landmarks
        self.results = self.holistic.process(rgb_image)
        annotated_image = self.image.copy()
        # Draw the landmarks on the image
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(annotated_image, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            landmarks = self.results.pose_landmarks.landmark
            self.coordinates = np.array([[landmark.x * self.width, landmark.y * self.height] for landmark in landmarks])
        
        cv2.imwrite('./6.11/annotated_rgb_image.png', annotated_image)
        return self.results

    def save_landmarks_to_file(self, results):
        landmarks = results.pose_landmarks.landmark
        filename = './landmakrs.txt'
        with open(filename, 'w') as f:
            for landmark in landmarks:
                f.write(f"{landmark.x * self.width}, {landmark.y * self.height}\n")
        return filename

    def get_pose(self):
        # Load the point cloud
        pose = []
        for (x, y) in self.coordinates:
            # Convert coordinates to integer values
            u, v = int(x), int(y)
            # Retrieve the depth value (z-coordinate) from the depth image
            if 0 <= u < self.width and 0 <= v < self.height:
                try:
                    coord = self.original_points[v, u]
                except Exception as e:
                    print(f"Failed to process the image: {e}")
            if not np.all(np.isnan(coord)):
                pose.append(coord)
        return np.array(pose)

    def create_visualize_point_cloud(self, pose):
        # Load the original point cloud
        # original_pcd = o3d.io.read_point_cloud(pcd_file)
        
        # Create the combined point cloud
        pcd_combined = o3d.geometry.PointCloud()
        pcd_combined.points = o3d.utility.Vector3dVector(pose)
        
        # Set colors for better visibility
        pcd_combined.paint_uniform_color([1, 0, 0])  # Red color

        # Create a visualization object
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_combined)
        vis.add_geometry(self.pcd)
        
        # Get the render options and set the point size
        render_option = vis.get_render_option()
        render_option.point_size = 10.0
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
# camera_intrinsic = (599.930725097656, 599.930725097656, 321.350402832031, 240.833572387695)
camera_intrinsic = (379.794, 379.794, 319.211, 237.715)
pose_estimator = PoseEstimation("./7.15/2.ply", camera_intrinsic)
pose_estimator.create_rgbd_from_point_cloud(480, 640, './7.15')
# landmarks = pose_estimator.detect_landmarks()
# landmarks_file = pose_estimator.save_landmarks_to_file(landmarks)
# pose = pose_estimator.get_pose()
# print(pose)
# pose_estimator.create_visualize_point_cloud(pose)