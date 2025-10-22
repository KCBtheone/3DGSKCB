# 文件名: post_process_mvs.py (Faiss-GPU 终极版)
import numpy as np
import open3d as o3d
import os
import json
import argparse
from tqdm import tqdm
import torch

# 尝试导入 Faiss，如果失败则提供清晰的安装指引
try:
    import faiss
except ImportError:
    print("❌ 错误: 未找到 'faiss' 库。")
    print("   请在您的 conda 环境中运行以下命令进行安装:")
    print("   conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=YOUR_CUDA_VERSION")
    print("   (请将 YOUR_CUDA_VERSION 替换为您的 CUDA 版本, 例如 11.3, 11.7, 12.1)")
    exit()

def point_cloud_to_udf_grid_faiss_gpu(point_cloud, grid_resolution=256):
    """
    【Faiss-GPU 终极版】
    使用 Faiss 库在 GPU 上执行高效的 M*log(N) 级别的最近邻搜索。
    这是目前速度最快的工业级解决方案。
    """
    if not torch.cuda.is_available():
        print("❌ 错误: 未找到 CUDA 设备。此脚本需要 GPU 运行。")
        return None, None
    print(f"✅ 检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")

    points_np = np.asarray(point_cloud.points).astype(np.float32)
    if len(points_np) == 0:
        print("❌ 错误: 点云中没有点，无法继续。")
        return None, None

    # 1. 【核心】构建 Faiss GPU 索引
    print("  -> 使用 Faiss 构建 GPU 索引...")
    dimension = points_np.shape[1]  # 应该是 3 (X, Y, Z)
    
    # a. 创建一个量化器/分区器
    # nlist 是分区的数量，是影响速度和精度的关键参数。
    # 一个好的经验法则是 sqrt(N)，其中 N 是点的总数。
    nlist = int(np.sqrt(len(points_np)))
    print(f"  -> Faiss 分区数 (nlist): {nlist}")
    quantizer = faiss.IndexFlatL2(dimension)
    
    # b. 创建一个倒排文件索引 (IVF)，这是实现 log(N) 性能的关键
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # c. 将索引移动到 GPU 上
    gpu_resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index) # 0 是 GPU ID

    # 2. 【核心】训练索引并添加数据
    print(f"  -> 训练 Faiss 索引...")
    gpu_index.train(points_np)
    print(f"  -> 将 {len(points_np)} 个点添加到 GPU 索引...")
    gpu_index.add(points_np)

    # 3. 【核心】设置搜索参数
    # nprobe 是搜索时要访问的分区数量。值越大越精确，但越慢。
    # 16 是一个在速度和精度之间很好的平衡点。
    gpu_index.nprobe = 16
    print(f"  -> Faiss 搜索分区数 (nprobe): {gpu_index.nprobe}")

    # 4. 创建查询网格点 (与之前相同)
    min_bound = point_cloud.get_min_bound()
    max_bound = point_cloud.get_max_bound()
    center = (min_bound + max_bound) / 2
    extent = (max_bound - min_bound) * 1.1
    min_bound_expanded = center - extent / 2
    max_bound_expanded = center + extent / 2
    x = np.linspace(min_bound_expanded[0], max_bound_expanded[0], grid_resolution, dtype=np.float32)
    y = np.linspace(min_bound_expanded[1], max_bound_expanded[1], grid_resolution, dtype=np.float32)
    z = np.linspace(min_bound_expanded[2], max_bound_expanded[2], grid_resolution, dtype=np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    grid_points_np = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)

    # 5. 【核心】在 GPU 上分批执行搜索
    print(f"  -> 开始在 GPU 上使用 Faiss 执行最近邻搜索...")
    batch_size = 262144 # Faiss 内存效率很高，可以使用较大的批次
    all_distances = []
    
    for i in tqdm(range(0, len(grid_points_np), batch_size), desc="Faiss 查询进度"):
        batch = grid_points_np[i:i+batch_size]
        
        # 核心搜索函数！ k=1 表示寻找最近的1个邻居
        distances, _ = gpu_index.search(batch, k=1)
        
        # Faiss 返回的是L2距离的平方，需要开根号得到真实距离
        all_distances.append(np.sqrt(distances))

    # 6. 合并结果
    print("  -> 计算完成，正在合并结果...")
    distances_combined = np.concatenate(all_distances)
    udf_grid = distances_combined.reshape(grid_resolution, grid_resolution, grid_resolution)

    grid_info = {
        "min_bound": min_bound_expanded.tolist(),
        "max_bound": max_bound_expanded.tolist(),
        "resolution": grid_resolution,
    }
    
    return udf_grid, grid_info


def process_scene(scene_path, resolution, voxel_size):
    print(f"\n--- 正在处理场景: {os.path.basename(scene_path)} ---")
    point_cloud_path = os.path.join(scene_path, "dense", "fused.ply")
    output_info_path = os.path.join(scene_path, "udf_grid_info.json")
    output_grid_path = os.path.join(scene_path, "udf_grid.npy")

    if not os.path.exists(point_cloud_path):
        print(f"❌ 错误: 未找到稠密点云 '{point_cloud_path}'。")
        return

    print(f"  -> 正在加载点云: {point_cloud_path}")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    
    if not pcd.has_points():
        print(f"❌ 错误: '{point_cloud_path}' 为空点云。")
        return

    if voxel_size > 0.0:
        print(f"  -> 原始点数: {len(pcd.points)}")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  -> 体素降采样后点数 (voxel_size={voxel_size}): {len(pcd.points)}")

    # 调用全新的 Faiss-GPU 加速函数
    udf_grid, grid_info = point_cloud_to_udf_grid_faiss_gpu(pcd, grid_resolution=resolution)
    
    if udf_grid is None:
        print(f"❌ 场景处理失败。")
        return

    print(f"  -> 正在保存 UDF 网格和元数据...")
    np.save(output_grid_path, udf_grid.astype(np.float32))
    with open(output_info_path, 'w') as f:
        json.dump(grid_info, f, indent=4)
        
    print(f"✅ 场景处理完成！结果已保存至 '{scene_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="【Faiss-GPU 终极版】将 COLMAP 的稠密点云转换为无符号距离场网格。")
    parser.add_argument("--dataset_path", type=str, required=True, help="单个场景的数据集根目录路径。")
    parser.add_argument("--resolution", type=int, default=256, help="生成的3D距离场网格的分辨率。")
    parser.add_argument("--voxel_size", type=float, default=0.015, 
                        help="用于点云降采样的体素大小(米)。设为 0.0 则不进行降采样。")
    args = parser.parse_args()
    
    process_scene(args.dataset_path, args.resolution, args.voxel_size)