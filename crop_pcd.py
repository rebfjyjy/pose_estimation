import numpy as np
import pandas as pd
import open3d as o3d

idx = 1
# Load the mask CSV file
mask_df = pd.read_csv(f'./7.15_mask{idx}.csv', header=None)
print(mask_df.shape)
mask_flattened = mask_df.values.flatten()

# Load the point cloud PLY file
point_cloud = o3d.io.read_point_cloud(f'./7.15/{idx}.ply')
points = np.zeros((307200-75344, 3))
colors = np.ones((307200-75344, 3))
temp_pcd = o3d.geometry.PointCloud()
temp_pcd.points = o3d.utility.Vector3dVector(points)
temp_pcd.colors = o3d.utility.Vector3dVector(colors)
combined_points = np.vstack((np.asarray(point_cloud.points), np.asarray(temp_pcd.points)))
combined_colors = np.vstack((np.asarray(point_cloud.colors), np.asarray(temp_pcd.colors)))
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(combined_points)
new_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
o3d.visualization.draw_geometries([new_pcd])

# Convert Open3D PointCloud to NumPy array
points = np.asarray(new_pcd.points)
colors = np.asarray(new_pcd.colors)
 
# Filter points based on mask
filtered_points = points[mask_flattened]
filtered_colors = colors[mask_flattened]

# Create a new Open3D PointCloud object with the filtered points
filtered_point_cloud = o3d.geometry.PointCloud()
filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

# Optionally save the filtered point cloud back to a new PLY file
output_path = f'./7.15/filtered_point_cloud_{idx}.ply'
o3d.io.write_point_cloud(output_path, filtered_point_cloud)
o3d.visualization.draw_geometries([filtered_point_cloud])

print(f"Filtered point cloud saved to: {output_path}")
 