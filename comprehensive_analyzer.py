import open3d as o3d
import numpy as np
import json
import os
import sys

# ==============================================================================
#                      硬编码配置参数 (Courtyard 场景)
# ==============================================================================

# 1. Colmap/GS 点云路径 (从 GS 模型导出的 .ply 文件)
GS_PLY_PATH = "/root/autodl-tmp/gaussian-splatting/NORMAL_EXPERIMENTS_ALPHA_EXPANDED/courtyard/exp1_base/point_cloud/iteration_20000/point_cloud.ply" 

# 2. 真值点云路径 (Courtyard GT)
GT_PLY_PATH = "/root/autodl-tmp/gaussian-splatting/data/courtyard/dslr_scan_eval/scan1.ply" 

# 3. 输出 JSON 文件路径 (用于更新 eth3d_alignments_v2.json)
OUTPUT_JSON_PATH = "/root/autodl-tmp/gaussian-splatting/eth3d_alignments_v2.json"

# 4. 场景名称 (必须与 JSON 文件的键名匹配)
SCENE_NAME = "courtyard"

# ==============================================================================
#                      配准函数
# ==============================================================================

def load_and_preprocess_pcd(path, voxel_size):
    """加载点云并进行下采样、估计法线。"""
    print(f"-> 正在加载点云: {os.path.basename(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"点云文件未找到: {path}")
    
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"点云文件 {path} 为空。")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"-> 下采样后剩余 {len(pcd_down.points)} 个点")
    
    # 法线估计是FPHF特征计算的前提
    # 增加搜索半径以确保找到足够的邻居来计算法线
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def compute_fpfh_features(pcd, voxel_size):
    """计算 FPFH 特征。"""
    radius_feature = voxel_size * 5
    print(f"-> 正在计算 FPFH 特征 (半径: {radius_feature:.4f})")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def execute_global_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    """使用 RANSAC (基于 FPFH) 进行全局粗配准。"""
    distance_threshold = voxel_size * 1.5
    print(f"-> 正在进行 RANSAC 全局粗配准 (阈值: {distance_threshold:.4f})")
    
    # ------------------ 最终修复尝试 ------------------
    # 强制使用关键字参数传递，并移除 mutual_filter (假设您的版本不接受这个位置参数)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source, 
        target=target, 
        source_feature=source_fpfh, 
        target_feature=target_fpfh,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    # -----------------------------------------------
    
    print(f"-> RANSAC 粗配准 Fitness: {result.fitness:.4f}, Inlier RMSE: {result.inlier_rmse:.4f}")
    return result

def refine_registration(source, target, result_ransac, icp_threshold):
    """使用 ICP 进行精细配准。"""
    print(f"-> 正在进行 ICP 精细配准 (阈值: {icp_threshold:.4f})")
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, icp_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    print(f"-> ICP 精配准 Fitness: {result.fitness:.4f}, Inlier RMSE: {result.inlier_rmse:.4f}")
    return result

# ==============================================================================
#                      主函数
# ==============================================================================
def main_compute_alignment(gs_ply_path, gt_ply_path, output_json_path, scene_name):
    # 配置参数
    VOXEL_SIZE = 0.05 # 用于配准的下采样大小，可调
    ICP_THRESHOLD = VOXEL_SIZE * 0.5 

    try:
        # 1. 加载和预处理
        source_pcd = load_and_preprocess_pcd(gs_ply_path, VOXEL_SIZE)
        target_pcd = load_and_preprocess_pcd(gt_ply_path, VOXEL_SIZE)
        
        # 2. 特征计算
        source_fpfh = compute_fpfh_features(source_pcd, VOXEL_SIZE)
        target_fpfh = compute_fpfh_features(target_pcd, VOXEL_SIZE)

        # 3. 全局粗配准 (RANSAC)
        ransac_result = execute_global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, VOXEL_SIZE)
        
        # 检查 RANSAC 结果，如果 Fitness 过低，则直接失败
        if ransac_result.fitness < 0.2:
             print("⚠️ 警告: RANSAC 粗配准 Fitness 过低，全局配准可能失败。请尝试调整 VOXEL_SIZE 或 ICP_THRESHOLD。")

        # 4. 精细配准 (ICP)
        final_result = refine_registration(source_pcd, target_pcd, ransac_result, ICP_THRESHOLD)
        
        transformation_matrix = final_result.transformation
        print("\n✅ 最终变换矩阵 (GS/Colmap -> World):")
        print(transformation_matrix)
        
        # 5. 更新 JSON 文件
        if not os.path.exists(output_json_path):
            print(f"创建新的对齐文件: {output_json_path}")
            alignments = {}
        else:
            with open(output_json_path, 'r') as f:
                alignments = json.load(f)
        
        alignments[scene_name] = transformation_matrix.tolist()
        
        with open(output_json_path, 'w') as f:
            json.dump(alignments, f, indent=4)
        
        print(f"\n📄 成功将 '{scene_name}' 的新对齐矩阵保存到: {output_json_path}")
        print("---")
        print("下一步：确保 evaluate_comparison.py 中粗略对齐不取逆，并重新运行评估脚本。")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_compute_alignment(GS_PLY_PATH, GT_PLY_PATH, OUTPUT_JSON_PATH, SCENE_NAME)