# preprocess_and_merge_gt.py
import open3d as o3d
import sys
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="合并多个真值点云文件，并进行体素下采样，为对齐做准备。")
    parser.add_argument("input_pcds", type=str, nargs='+', 
                        help="一个或多个输入的真值点云文件路径 (例如: scan1.ply scan2.ply)。")
    parser.add_argument("--output_pcd_path", type=str, required=True, 
                        help="处理后输出的单一、已降采样的点云文件路径。")
    parser.add_argument("--voxel_size", type=float, default=0.05, 
                        help="用于降采样的体素大小(米)。")
    args = parser.parse_args()

    print("─"*80)
    print("🚀 开始合并与预处理真值点云...")
    
    pcds_to_merge = []
    for pcd_path in args.input_pcds:
        print(f"  - 正在加载: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.has_points():
            pcds_to_merge.append(pcd)
        else:
            print(f"  ⚠️ 警告: 文件 {pcd_path} 为空或无法加载，已跳过。")

    if not pcds_to_merge:
        print("❌ 错误: 未能加载任何有效的点云文件。")
        sys.exit(1)

    print("\n  - 正在合并点云...")
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcds_to_merge:
        merged_pcd += pcd
    
    print(f"    -> 合并后总点数: {len(merged_pcd.points)}")

    print(f"\n  - 正在进行体素下采样 (voxel_size = {args.voxel_size})...")
    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size=args.voxel_size)
    print(f"    -> 降采样后总点数: {len(downsampled_pcd.points)}")

    print(f"\n  - 正在保存处理后的点云至: {args.output_pcd_path}")
    o3d.io.write_point_cloud(args.output_pcd_path, downsampled_pcd)

    print("\n✅ 预处理完成！现在可以在对齐脚本中使用这个新生成的点云文件了。")
    print("─"*80)

if __name__ == "__main__":
    main()