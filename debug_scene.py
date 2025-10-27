import numpy as np
import open3d as o3d
import os
from scene.dataset_readers import readLLFFScene  # 确保这能正确导入

def visualize_scene(scene_info):
    # 获取点云数据
    pcd_points = scene_info.point_cloud.points
    pcd_colors = scene_info.point_cloud.colors

    # 创建 Open3D 点云对象
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd_points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # 可视化对象列表
    geometries = [o3d_pcd]

    # 为每个相机创建坐标系和视锥
    for cam_info in scene_info.train_cameras:
        # 从 W2C (R, T) 计算 C2W
        W2C = np.eye(4)
        W2C[:3, :3] = cam_info.R
        W2C[:3, 3] = cam_info.T
        C2W = np.linalg.inv(W2C)

        # 创建一个代表相机的坐标系
        cam_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_mesh.transform(C2W)
        geometries.append(cam_mesh)
    
    # 添加一个世界坐标系方便观察
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(world_frame)


    print("正在显示场景... 您应该能看到点云被相机包围。")
    print("如果相机和点云分离得很远，说明归一化存在问题。")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    # --- 请修改为您的路径 ---
    llff_scene_path = "/root/autodl-tmp/gaussian-splatting/data/LLFF/nerf_llff_data/fern"
    # -------------------------

    print(f"正在从 {llff_scene_path} 加载 LLFF 场景...")
    
    # eval=True, images="images" 是您脚本中的配置
    scene_info = readLLFFScene(path=llff_scene_path, images="images", eval=True)

    visualize_scene(scene_info)
