import open3d as o3d
import numpy as np
import json
from pathlib import Path
import sys
import struct
from tqdm import tqdm
import argparse
import xml.etree.ElementTree as ET

# --- COLMAP点云读取函数 (保持不变) ---
def read_points3D_text(path):
    points3D = []
    with open(path, "r") as fid:
        for line in fid:
            if line.startswith("#"):
                continue
            elems = line.strip().split()
            xyz = np.array([float(x) for x in elems[1:4]])
            points3D.append(xyz)
    return np.array(points3D)

def read_points3D_binary(path):
    points3D = []
    with open(path, "rb") as fid:
        num_points = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num_points):
            fid.read(8)
            xyz = struct.unpack("ddd", fid.read(24))
            points3D.append(np.array(xyz))
            fid.read(3)
            fid.read(8)
            track_length = struct.unpack("Q", fid.read(8))[0]
            fid.read(12 * track_length)
    return np.array(points3D)

# --- [新增] 智能加载并合并多个PLY扫描文件的函数 ---
def load_and_merge_gt_plys(gt_dir_path: Path):
    """
    智能地加载一个目录下的所有scan*.ply文件，
    并使用scan_alignment.mlp文件将它们合并。
    """
    if not gt_dir_path.is_dir():
        return None

    ply_files = sorted(list(gt_dir_path.glob("scan*.ply")))
    mlp_path = gt_dir_path / "scan_alignment.mlp"

    if not ply_files:
        return None

    # 解析MLP文件以获取变换矩阵
    transforms = {}
    if mlp_path.exists():
        try:
            tree = ET.parse(mlp_path)
            root = tree.getroot()
            # 找到所有的Mesh层
            for mesh_elem in root.findall(".//Mesh"):
                filename = mesh_elem.get("filename")
                matrix_elem = mesh_elem.find("MLMatrix")
                if filename and matrix_elem is not None:
                    values = [float(v) for v in matrix_elem.text.strip().split()]
                    if len(values) == 16:
                        transforms[filename] = np.array(values).reshape(4, 4)
        except Exception as e:
            print(f"  警告: 解析 '{mlp_path}' 文件失败: {e}")

    # 加载、变换并合并点云
    merged_pcd = o3d.geometry.PointCloud()
    for ply_path in ply_files:
        pcd_part = o3d.io.read_point_cloud(str(ply_path))
        if ply_path.name in transforms:
            print(f"  应用变换到 {ply_path.name}...")
            pcd_part.transform(transforms[ply_path.name])
        merged_pcd += pcd_part
    
    print(f"  成功合并 {len(ply_files)} 个PLY文件，总点数: {len(merged_pcd.points)}")
    return merged_pcd


def main(args):
    dataset_root = Path(args.dataset_root)
    output_file = Path(args.output_file)
    all_transformations = {}

    if not dataset_root.is_dir():
        print(f"错误: 数据集根目录 '{dataset_root}' 不存在或不是一个目录。")
        sys.exit(1)

    scene_paths = [p for p in dataset_root.iterdir() if p.is_dir()]
    print(f"找到 {len(scene_paths)} 个潜在的场景目录。")

    for scene_path in tqdm(scene_paths, desc="Processing Scenes"):
        scene_name = scene_path.name
        print(f"\n--- 正在处理场景: {scene_name} ---")

        # --- 1. 定位文件和目录 ---
        colmap_points_path = scene_path / "sparse" / "0" / "points3D.txt"
        if not colmap_points_path.exists():
             colmap_points_path = scene_path / "sparse" / "0" / "points3D.bin"

        gt_dir = scene_path / args.gt_dir_name

        if not colmap_points_path.exists():
            print(f"  警告: 找不到COLMAP点云, 跳过此场景。")
            continue
        if not gt_dir.is_dir():
            print(f"  警告: 找不到真值目录 '{args.gt_dir_name}', 跳过此场景。")
            continue

        # --- 2. 加载点云 (使用新的合并函数) ---
        try:
            if colmap_points_path.suffix == ".txt":
                colmap_points = read_points3D_text(colmap_points_path)
            else:
                colmap_points = read_points3D_binary(colmap_points_path)
            
            pcd_colmap = o3d.geometry.PointCloud()
            pcd_colmap.points = o3d.utility.Vector3dVector(colmap_points)

            # 调用新的合并函数
            pcd_gt = load_and_merge_gt_plys(gt_dir)
            
            if not pcd_colmap.has_points() or pcd_gt is None or not pcd_gt.has_points():
                print("  警告: 加载的点云之一为空, 跳过。")
                continue

        except Exception as e:
            print(f"  错误: 加载点云时出错: {e}, 跳过。")
            continue

        # --- 3. 执行ICP对齐 (与之前相同) ---
        voxel_size = args.voxel_size
        pcd_colmap_down = pcd_colmap.voxel_down_sample(voxel_size)
        pcd_gt_down = pcd_gt.voxel_down_sample(voxel_size)

        threshold = voxel_size * 1.5
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_gt_down, pcd_colmap_down, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

        transformation_matrix = reg_p2p.transformation
        
        all_transformations[scene_name] = transformation_matrix.tolist()
        print(f"  场景 '{scene_name}' 对齐完成。")

        if args.visualize:
            pcd_colmap.paint_uniform_color([0, 0, 1])
            pcd_gt.transform(transformation_matrix).paint_uniform_color([0, 1, 0])
            print("  正在显示对齐结果...关闭窗口以继续。")
            o3d.visualization.draw_geometries([pcd_colmap, pcd_gt], window_name=f"对齐结果: {scene_name}")

    # --- 4. 保存所有变换矩阵 (与之前相同) ---
    if not all_transformations:
        print("\n错误：未能成功处理任何场景。请检查你的目录结构和文件路径。")
        sys.exit(1)

    print(f"\n--- 所有场景处理完毕, 正在将 {len(all_transformations)} 个变换矩阵保存到 '{output_file}' ---")
    with open(output_file, 'w') as f:
        json.dump(all_transformations, f, indent=4)

    print("保存成功！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量对齐ETH3D场景的真值点云到COLMAP坐标系 (V2 - 自动合并多个scan.ply)。")
    parser.add_argument("dataset_root", type=str, 
                        help="包含所有ETH3D场景文件夹的根目录。")
    parser.add_argument("--output_file", type=str, default="eth3d_alignments_v2.json",
                        help="用于保存所有变换矩阵的输出JSON文件名。")
    # 修改了参数，现在指定目录名而不是完整文件路径
    parser.add_argument("--gt_dir_name", type=str, default="dslr_scan_eval",
                        help="包含真值PLY扫描文件的目录名。")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="用于ICP降采样的体素大小。")
    parser.add_argument("--visualize", action="store_true",
                        help="如果设置此项, 将会逐个可视化每个场景的对齐结果。")
    
    args = parser.parse_args()
    main(args)