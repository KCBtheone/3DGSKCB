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
#                      自动化预对齐方案 (关键修改)
# ==============================================================================

def get_pre_alignment_strategies():
    """
    返回一个包含多种预对齐变换矩阵及其名称的字典。
    脚本将自动遍历这些方案，直到找到一个可行的为止。
    """
    strategies = {}
    
    # 辅助函数，用于创建旋转矩阵
    def get_rotation_matrix(axis, angle_deg):
        th = np.deg2rad(angle_deg)
        cos_th, sin_th = np.cos(th), np.sin(th)
        if axis.lower() == 'y':
            return np.array([
                [cos_th,  0, sin_th, 0],
                [0,       1, 0,      0],
                [-sin_th, 0, cos_th, 0],
                [0,       0, 0,      1]
            ])
        elif axis.lower() == 'z':
            return np.array([
                [cos_th, -sin_th, 0, 0],
                [sin_th,  cos_th, 0, 0],
                [0,       0,      1, 0],
                [0,       0,      0, 1]
            ])
        elif axis.lower() == 'x':
            return np.array([
                [1, 0,       0,      0],
                [0, cos_th, -sin_th, 0],
                [0, sin_th,  cos_th, 0],
                [0, 0,       0,      1]
            ])
        return np.identity(4)

    # 定义要尝试的策略
    strategies["Rotate Y by 90 deg"] = get_rotation_matrix('y', 90)
    strategies["Rotate Y by -90 deg"] = get_rotation_matrix('y', -90)
    strategies["Rotate Y by 180 deg"] = get_rotation_matrix('y', 180)
    strategies["Rotate Z by 180 deg"] = get_rotation_matrix('z', 180)
    strategies["Rotate X by 180 deg"] = get_rotation_matrix('x', 180)
    strategies["No Pre-alignment"] = np.identity(4) # 作为最后的尝试
    
    return strategies

# ==============================================================================
#                      配准函数 (无需修改)
# ==============================================================================

def load_and_preprocess_pcd(path, voxel_size):
    print(f"-> 正在加载点云: {os.path.basename(path)}")
    if not os.path.exists(path): raise FileNotFoundError(f"点云文件未找到: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points(): raise ValueError(f"点云文件 {path} 为空。")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"-> 下采样后剩余 {len(pcd_down.points)} 个点")
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def compute_fpfh_features(pcd, voxel_size):
    radius_feature = voxel_size * 5
    print(f"-> 正在计算 FPFH 特征 (半径: {radius_feature:.4f})")
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

def execute_global_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(f"-> 正在进行 RANSAC 全局粗配准 (阈值: {distance_threshold:.4f})")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source, target=target, source_feature=source_fpfh, target_feature=target_fpfh,
        mutual_filter=True, max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=4000000, confidence=0.999)
    )
    print(f"-> RANSAC 粗配准 Fitness: {result.fitness:.4f}, Inlier RMSE: {result.inlier_rmse:.4f}")
    return result

def refine_registration(source, target, result_ransac, icp_threshold):
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
    VOXEL_SIZE = 0.2  # 对于大场景，使用更大的体素可以更好地捕捉宏观结构
    ICP_THRESHOLD = VOXEL_SIZE * 0.4
    RANSAC_SUCCESS_THRESHOLD = 0.2 # RANSAC Fitness 超过此值即视为成功

    try:
        # 1. 加载目标点云并计算其特征 (只需一次)
        target_pcd = load_and_preprocess_pcd(gt_ply_path, VOXEL_SIZE)
        target_fpfh = compute_fpfh_features(target_pcd, VOXEL_SIZE)

        # 2. 循环尝试所有预对齐方案
        strategies = get_pre_alignment_strategies()
        best_ransac_result = None
        successful_pre_alignment = None
        successful_source_pcd = None

        for name, pre_alignment_matrix in strategies.items():
            print(f"\n{'='*25}\n attempting STRATEGY: {name}\n{'='*25}")
            
            # 每次都加载一个新的源点云，避免变换累积
            source_pcd_original = load_and_preprocess_pcd(gs_ply_path, VOXEL_SIZE)
            
            # 应用预对齐变换
            source_pcd_transformed = o3d.geometry.PointCloud(source_pcd_original) # 创建副本
            source_pcd_transformed.transform(pre_alignment_matrix)
            
            # 计算变换后点云的特征
            source_fpfh = compute_fpfh_features(source_pcd_transformed, VOXEL_SIZE)
            
            # 运行RANSAC
            ransac_result = execute_global_registration(source_pcd_transformed, target_pcd, source_fpfh, target_fpfh, VOXEL_SIZE)

            if ransac_result.fitness > RANSAC_SUCCESS_THRESHOLD:
                print(f"\n✅ SUCCESS! Strategy '{name}' achieved a fitness of {ransac_result.fitness:.4f}.")
                best_ransac_result = ransac_result
                successful_pre_alignment = pre_alignment_matrix
                successful_source_pcd = source_pcd_transformed
                break # 找到可行方案，跳出循环
            else:
                print(f"❌ Strategy '{name}' failed. Fitness ({ransac_result.fitness:.4f}) is below threshold ({RANSAC_SUCCESS_THRESHOLD}).")

        # 3. 检查是否有任何方案成功
        if not successful_pre_alignment is not None:
            print("\n\n❌❌❌ ALL AUTOMATED STRATEGIES FAILED. ❌❌❌")
            print("Could not find a valid initial alignment for the point clouds.")
            print("Please consider visualizing the point clouds manually to find a better pre-alignment.")
            sys.exit(1)

        # 4. 精细配准 (ICP)
        icp_result = refine_registration(successful_source_pcd, target_pcd, best_ransac_result, ICP_THRESHOLD)
        
        # 5. 合并变换矩阵 (最终变换 = ICP变换 * 预对齐变换)
        final_transformation = np.dot(icp_result.transformation, successful_pre_alignment)
        
        print("\n✅ 最终变换矩阵 (GS/Colmap -> World):")
        print(final_transformation)
        
        # 6. 更新 JSON 文件
        alignments = {}
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                alignments = json.load(f)
        
        alignments[scene_name] = final_transformation.tolist()
        
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