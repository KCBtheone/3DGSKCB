import os
import argparse
import re
from plyfile import PlyData

def searchForMaxIteration(folder):
    """从utils/system_utils.py中借鉴，用于查找最大迭代次数"""
    saved_iters = [int(re.search(r"iteration_(\d+)", f).group(1)) for f in os.listdir(folder) if "iteration_" in f]
    if not saved_iters:
        return None
    return max(saved_iters)

def find_ply_file(exp_path):
    """
    在给定的实验路径中查找最合适的.ply文件。
    优先查找 best.ply，如果找不到，则查找最新迭代次数的 point_cloud.ply。
    """
    point_cloud_dir = os.path.join(exp_path, "point_cloud")
    if not os.path.isdir(point_cloud_dir):
        return None, f"目录不存在: {point_cloud_dir}"

    # 优先查找 best.ply
    best_ply_path = os.path.join(point_cloud_dir, "best", "point_cloud.ply")
    if os.path.exists(best_ply_path):
        return best_ply_path, "best"

    # 如果没有 best.ply，查找最新迭代
    iteration = searchForMaxIteration(point_cloud_dir)
    if iteration is not None:
        iter_ply_path = os.path.join(point_cloud_dir, f"iteration_{iteration}", "point_cloud.ply")
        if os.path.exists(iter_ply_path):
            return iter_ply_path, f"iteration_{iteration}"

    return None, "未找到 'best' 或任何 'iteration_XXX' 结果"

def inspect_ply(ply_path):
    """
    读取.ply文件并返回其'vertex'元素的属性列表。
    """
    try:
        ply_data = PlyData.read(ply_path)
        if 'vertex' not in ply_data:
            return None, "文件中不包含 'vertex' 元素"
        
        vertex_element = ply_data['vertex']
        property_names = [prop.name for prop in vertex_element.properties]
        return property_names, f"共 {len(property_names)} 个属性"
    except Exception as e:
        return None, f"读取文件时出错: {e}"

def main(root_dir):
    print(f"🕵️  开始检查实验目录: {root_dir}\n")

    if not os.path.isdir(root_dir):
        print(f"❌ 错误: 路径 '{root_dir}' 不是一个有效的目录。")
        return

    # 假设顶级目录直接就是场景目录，或者包含多个场景目录
    scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not scenes: # 如果顶级目录就是场景目录本身
        scenes = [os.path.basename(root_dir)]
        root_dir = os.path.dirname(root_dir)

    for scene_name in scenes:
        scene_path = os.path.join(root_dir, scene_name)
        print(f"--- 场景: {scene_name} ---")
        
        experiments = sorted([d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))])
        if not experiments:
            print("  -> 未找到任何实验目录。")
            continue

        for exp_name in experiments:
            exp_path = os.path.join(scene_path, exp_name)
            print(f"  - 实验: {exp_name}")
            
            ply_path, status = find_ply_file(exp_path)
            
            if ply_path:
                print(f"    - 正在检查: {status}")
                properties, prop_status = inspect_ply(ply_path)
                if properties:
                    print(f"    - 属性列表 ({prop_status}):")
                    # 为了美观，每8个属性换一行打印
                    for i in range(0, len(properties), 8):
                        print("      " + ", ".join(properties[i:i+8]))
                else:
                    print(f"    - ❌ 检查失败: {prop_status}")
            else:
                print(f"    - ❌ 文件查找失败: {status}")
        print("-" * (len(scene_name) + 6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查3DGS实验输出的.ply文件格式。")
    parser.add_argument("experiment_root", type=str, help="包含所有场景和实验结果的顶级目录，例如 'output/V6_MODULE_SHOWDOWN_FINAL'。")
    args = parser.parse_args()
    
    main(args.experiment_root)