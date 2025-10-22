# generate_alignment.py
import open3d as o3d
import numpy as np
import json
import os
import sys
from argparse import ArgumentParser
import copy

def calculate_alignment_transform(pred_pcd_path, gt_pcd_path, voxel_size):
    """
    使用全局+局部配准算法，自动计算从预测点云到真值点云的对齐变换矩阵。

    Args:
        pred_pcd_path (str): 预测点云 (.ply) 的路径。
        gt_pcd_path (str): 真值点云 (.ply) 的路径。
        voxel_size (float): 用于下采样和特征计算的体素大小。

    Returns:
        np.ndarray: 4x4 的变换矩阵，如果失败则返回 None。
    """
    print("─"*80)
    print("🚀 开始自动计算对齐变换矩阵...")
    
    # 1. 加载点云
    print(f"  1. 正在加载点云:\n     - 预测: {pred_pcd_path}\n     - 真值: {gt_pcd_path}")
    try:
        pred_pcd = o3d.io.read_point_cloud(pred_pcd_path)
        gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
        if not pred_pcd.has_points() or not gt_pcd.has_points():
            print("  ❌ 错误: 一个或两个点云文件为空或无法加载。")
            return None
    except Exception as e:
        print(f"  ❌ 错误: 加载点云失败: {e}")
        return None

    # 2. 预处理：下采样以提高速度和鲁棒性
    print(f"  2. 正在进行体素下采样 (voxel_size = {voxel_size})...")
    pred_down = pred_pcd.voxel_down_sample(voxel_size)
    gt_down = gt_pcd.voxel_down_sample(voxel_size)

    # 3. 计算FPFH特征用于全局配准
    print("  3. 正在计算法线和FPFH特征...")
    radius_normal = voxel_size * 2
    pred_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    gt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pred_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pred_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    gt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(gt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # 4. 全局配准 (RANSAC)
    print("  4. 正在执行全局配准 (RANSAC)...")
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pred_down, gt_down, pred_fpfh, gt_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    coarse_transformation = result_ransac.transformation
    print(f"     -> 全局配准完成. Fitness: {result_ransac.fitness:.4f}, Inlier RMSE: {result_ransac.inlier_rmse:.4f}")

    # 5. 局部配准 (ICP)
    print("  5. 正在执行局部精细配准 (ICP)...")
    # 使用上一步的粗对齐结果作为ICP的初始变换
    icp_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        pred_down, gt_down, icp_threshold, coarse_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    final_transformation = result_icp.transformation
    print(f"     -> ICP完成. Fitness: {result_icp.fitness:.4f}, Inlier RMSE: {result_icp.inlier_rmse:.4f}")
    
    if result_icp.fitness < 0.8: # Fitness衡量重叠区域的比例，值太低说明对齐效果差
        print("  ⚠️ 警告: ICP Fitness 值较低，对齐效果可能不佳。请尝试调整 --voxel_size。")
    
    print("✅ 对齐矩阵计算成功!")
    print("─"*80)

    # 返回最终的变换矩阵
    return final_transformation

def visualize_alignment(pred_pcd_path, gt_pcd_path, transform):
    """可视化对齐前后的效果，用于调试"""
    print("\n[可选] 正在生成可视化结果以供检查...")
    source = o3d.io.read_point_cloud(pred_pcd_path)
    target = o3d.io.read_point_cloud(gt_pcd_path)

    # 原始状态（不同颜色）
    source.paint_uniform_color([1, 0.706, 0])  # 黄色
    target.paint_uniform_color([0, 0.651, 0.929]) # 蓝色
    # o3d.visualization.draw_geometries([source, target], window_name="对齐前")

    # 对齐后状态
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transform)
    o3d.visualization.draw_geometries([source_transformed, target], window_name="对齐后 (请检查是否重合)")

def main():
    parser = ArgumentParser(description="自动计算3DGS预测点云到真值点云的对齐矩阵，并保存为JSON文件。")
    parser.add_argument("--pred_pcd_path", type=str, required=True, 
                        help="预测点云的路径。从任意一个实验的输出中选取即可, e.g., 'output/exp1/point_cloud/iteration_30000/point_cloud.ply'")
    parser.add_argument("--gt_pcd_path", type=str, required=True, 
                        help="真值点云的路径。e.g., 'data/scene/gt.ply'")
    parser.add_argument("--output_json", type=str, required=True, 
                        help="输出的对齐JSON文件的路径。e.g., 'eth3d_alignments_v2.json'")
    parser.add_argument("--scene_name", type=str, required=True, 
                        help="当前场景的名称，将作为JSON文件中的key。 e.g., 'electro'")
    parser.add_argument("--voxel_size", type=float, default=0.02, 
                        help="用于点云下采样的体素大小(米)。这是最重要的参数，需要根据场景尺度进行调整。")
    parser.add_argument("--visualize", action="store_true", 
                        help="计算后显示一个3D窗口来可视化对齐效果。")
    
    args = parser.parse_args()

    # 计算变换矩阵
    transform_matrix = calculate_alignment_transform(args.pred_pcd_path, args.gt_pcd_path, args.voxel_size)

    if transform_matrix is None:
        print("\n❌ 计算失败，未生成JSON文件。")
        sys.exit(1)

    # 加载或创建现有的JSON文件，以支持追加新场景
    if os.path.exists(args.output_json):
        print(f"\n🔄 发现现有JSON文件 '{args.output_json}', 将更新或添加场景 '{args.scene_name}'...")
        with open(args.output_json, 'r') as f:
            data = json.load(f)
    else:
        print(f"\n✨ 未发现现有JSON文件，将创建新文件 '{args.output_json}'...")
        data = {}

    # 更新数据并保存
    data[args.scene_name] = transform_matrix.tolist() # Numpy array转为list才能序列化

    with open(args.output_json, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ 成功将场景 '{args.scene_name}' 的对齐矩阵保存到 '{args.output_json}'!")
    print("\n现在您可以在主评估脚本中使用这个JSON文件了。")
    
    # 可视化检查
    if args.visualize:
        visualize_alignment(args.pred_pcd_path, args.gt_pcd_path, transform_matrix)

if __name__ == "__main__":
    main()