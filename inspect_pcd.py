# inspect_pcd.py
import open3d as o3d
import numpy as np
import sys
from argparse import ArgumentParser

def inspect_point_cloud(pcd_path):
    """
    加载一个点云文件并打印其关键诊断信息。
    """
    print("─"*80)
    print(f"🕵️  正在检查点云文件: {pcd_path}")
    
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            print("  ❌ 错误: 点云为空或无法加载。")
            print("─"*80)
            return
    except Exception as e:
        print(f"  ❌ 错误: 加载点云失败: {e}")
        print("─"*80)
        return

    # 获取点云的基本信息
    num_points = len(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    center = bbox.get_center()
    extent = bbox.get_extent() # 边界框的尺寸 (长、宽、高)

    print("\n[📊 核心诊断信息]")
    print(f"  - 点的总数: {num_points}")
    print(f"  - 坐标中心点 (Center): [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  - 边界框尺寸 (Extent): [长:{extent[0]:.4f}, 宽:{extent[1]:.4f}, 高:{extent[2]:.4f}]")
    print(f"  - 最小坐标 (Min Bound): [{min_bound[0]:.4f}, {min_bound[1]:.4f}, {min_bound[2]:.4f}]")
    print(f"  - 最大坐标 (Max Bound): [{max_bound[0]:.4f}, {max_bound[1]:.4f}, {max_bound[2]:.4f}]")
    
    print("\n[💡 分析建议]")
    print("  - 比较两个点云的'坐标中心点'和'边界框尺寸'。")
    print("  - 如果'边界框尺寸'相差几个数量级 (例如，一个为5.0，另一个为5000.0)，说明存在严重的【尺度差异】。")
    print("  - 如果'边界框尺寸'相似，但'坐标中心点'相距很远，说明存在【平移差异】。")
    print("─"*80)

if __name__ == "__main__":
    parser = ArgumentParser(description="检查并打印点云文件的基本信息以诊断对齐问题。")
    parser.add_argument("pcd_file", type=str, help="要检查的点云文件路径 (.ply)")
    args = parser.parse_args()
    
    inspect_point_cloud(args.pcd_file)