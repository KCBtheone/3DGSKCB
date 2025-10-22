import open3d as o3d
import numpy as np
from pathlib import Path
import sys
import argparse
import xml.etree.ElementTree as ET

# 复用之前的PLY合并函数
def load_and_merge_gt_plys(gt_dir_path: Path):
    if not gt_dir_path.is_dir(): return None
    ply_files = sorted(list(gt_dir_path.glob("scan*.ply")))
    mlp_path = gt_dir_path / "scan_alignment.mlp"
    if not ply_files: return None
    transforms = {}
    if mlp_path.exists():
        try:
            tree = ET.parse(mlp_path)
            root = tree.getroot()
            for mesh_elem in root.findall(".//Mesh"):
                filename = mesh_elem.get("filename")
                matrix_elem = mesh_elem.find("MLMatrix")
                if filename and matrix_elem is not None:
                    values = [float(v) for v in matrix_elem.text.strip().split()]
                    if len(values) == 16: transforms[filename] = np.array(values).reshape(4, 4)
        except Exception as e:
            print(f"  警告: 解析 '{mlp_path}' 文件失败: {e}")
    merged_pcd = o3d.geometry.PointCloud()
    for ply_path in ply_files:
        pcd_part = o3d.io.read_point_cloud(str(ply_path))
        if ply_path.name in transforms: pcd_part.transform(transforms[ply_path.name])
        merged_pcd += pcd_part
    return merged_pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将一个场景的多个scan.ply文件合并成一个单一的真值PLY文件。")
    parser.add_argument("scene_path", type=str, help="单个场景的路径 (例如: data/pipes)。")
    args = parser.parse_args()

    scene_path = Path(args.scene_path)
    gt_dir = scene_path / "dslr_scan_eval"
    output_path = scene_path / f"{scene_path.name}_gt_merged.ply"

    print(f"正在从 '{gt_dir}' 加载并合并PLY文件...")
    merged_pcd = load_and_merge_gt_plys(gt_dir)

    if merged_pcd and merged_pcd.has_points():
        print(f"合并成功！正在将结果保存到: {output_path}")
        o3d.io.write_point_cloud(str(output_path), merged_pcd)
        print("保存完毕。")
    else:
        print(f"错误：未能从 '{gt_dir}' 加载或合并任何点云。")