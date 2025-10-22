import open3d as o3d
import numpy as np
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
from scipy.spatial.transform import Rotation
import multiprocessing
from tqdm import tqdm
import argparse

# --- COLMAP 数据读取器 ---
class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

class Image:
    def __init__(self, id, qvec, tvec, camera_id, name):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name

    def qvec2rotmat(self):
        q = self.qvec
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            if line.startswith("#"):
                continue
            elems = line.strip().split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array([float(elem) for elem in elems[4:]])
            cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras

def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        lines = fid.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("#"):
                i += 1
                continue
            elems = line.strip().split()
            image_id = int(elems[0])
            qvec = np.array([float(q) for q in elems[1:5]])
            tvec = np.array([float(t) for t in elems[5:8]])
            camera_id = int(elems[8])
            image_name = elems[9]
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name)
            i += 2
    return images

# --- 真值点云加载与合并函数 ---
def load_and_merge_gt_plys(gt_dir_path: Path):
    if not gt_dir_path.is_dir():
        return None
    ply_files = sorted(list(gt_dir_path.glob("scan*.ply")))
    mlp_path = gt_dir_path / "scan_alignment.mlp"
    if not ply_files:
        return None
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
                    if len(values) == 16:
                        transforms[filename] = np.array(values).reshape(4, 4)
        except Exception as e:
            print(f"  警告: 解析 '{mlp_path}' 文件失败: {e}")
    
    merged_pcd = o3d.geometry.PointCloud()
    print("  - 正在合并以下 .ply 文件:")
    for ply_path in ply_files:
        print(f"    - {ply_path.name}")
        pcd_part = o3d.io.read_point_cloud(str(ply_path))
        if ply_path.name in transforms:
            pcd_part.transform(transforms[ply_path.name])
        merged_pcd += pcd_part
    return merged_pcd

# --- 为单个图像生成并保存法线图的函数 ---
def process_and_save_normal_map(image, camera, xyz_world, normals_world, output_dir):
    R_cw = image.qvec2rotmat()
    t_cw = image.tvec.reshape(3, 1)

    fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    W, H = camera.width, camera.height

    xyz_c = R_cw @ xyz_world.T + t_cw
    normals_c = R_cw @ normals_world.T

    z_c = xyz_c[2, :]
    uv_homogeneous = K @ xyz_c
    uv = uv_homogeneous[:2, :] / (uv_homogeneous[2, :] + 1e-6)
    u, v = uv[0, :], uv[1, :]

    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_c > 0.1)
    u_proj, v_proj = u[valid_mask].astype(int), v[valid_mask].astype(int)

    normal_map = np.zeros((H, W, 3), dtype=np.float32)
    normal_map[v_proj, u_proj] = normals_c.T[valid_mask]

    normal_map_uint8 = ((normal_map / 2.0 + 0.5) * 255).astype(np.uint8)
    mask_sparse = (np.linalg.norm(normal_map, axis=2) > 0).astype(np.uint8)

    inpainted_uint8 = cv2.inpaint(normal_map_uint8, 1 - mask_sparse, 5, cv2.INPAINT_NS)

    normal_map_dense = (inpainted_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0

    norms = np.linalg.norm(normal_map_dense, axis=2, keepdims=True)
    normal_map_dense /= (norms + 1e-6)

    img_basename = Path(image.name).stem
    normal_to_save_uint8 = ((normal_map_dense / 2.0 + 0.5) * 255).astype(np.uint8)
    normal_to_save_bgr = cv2.cvtColor(normal_to_save_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{img_basename}_normal.png"), normal_to_save_bgr)
    return image.name

# --- 用于解包参数的包装函数 ---
def process_and_save_normal_map_wrapper(args):
    return process_and_save_normal_map(*args)

# --- 主函数 ---
def main(args):
    scene_path = Path(args.scene_path)
    scene_name = scene_path.name
    alignment_file = Path(args.alignment_file)
    
    print(f"--- 开始处理场景: {scene_name} ---")

    # --- 1. 加载对齐矩阵 ---
    print(f"\n[步骤 1/7] 正在从 '{alignment_file}' 加载对齐矩阵...")
    if not alignment_file.exists():
        print(f"错误: 对齐文件 '{alignment_file}' 不存在。")
        sys.exit(1)
    with open(alignment_file, 'r') as f:
        alignments = json.load(f)
    if scene_name not in alignments:
        print(f"错误: 在对齐文件中找不到场景 '{scene_name}' 的变换矩阵。")
        sys.exit(1)
    transformation_matrix = np.array(alignments[scene_name])
    print("✓ 对齐矩阵加载成功。")

    # --- 2. 加载并合并真值点云 ---
    gt_dir = scene_path / "dslr_scan_eval"
    print(f"\n[步骤 2/7] 正在从 '{gt_dir}' 加载并合并真值点云...")
    pcd_gt = load_and_merge_gt_plys(gt_dir)
    if pcd_gt is None or not pcd_gt.has_points():
        print(f"错误: 加载真值点云失败。请检查 '{gt_dir}' 目录。")
        sys.exit(1)
    print(f"  - 合并后点云包含 {len(pcd_gt.points):,} 个点。")
    print("✓ 真值点云加载并合并成功。")
    
    # --- 3. 对齐点云 ---
    print("\n[步骤 3/7] 正在应用变换矩阵对齐点云...")
    pcd_gt.transform(transformation_matrix)
    print("✓ 点云对齐成功。")

    # --- 4. 体素降采样 ---
    print(f"\n[步骤 4/7] 正在进行体素降采样...")
    if args.voxel_size > 0:
        print(f"  - Voxel size 设置为: {args.voxel_size}")
        pcd_gt = pcd_gt.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"  - 降采样后点云数量: {len(pcd_gt.points):,} 个点。")
        print("✓ 降采样完成。")
    else:
        print("  - 已跳过降采样 (voxel_size <= 0)。")

    # --- 5. 为点云计算法线 ---
    print("\n[步骤 5/7] 正在为点云计算法线... (此步骤可能耗时较长，请耐心等待)")
    pcd_gt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=30))
    pcd_gt.orient_normals_consistent_tangent_plane(100)
    print("✓ 法线计算完成。")
    
    xyz_world = np.asarray(pcd_gt.points)
    normals_world = np.asarray(pcd_gt.normals)

    # --- 6. 加载COLMAP相机数据 ---
    colmap_sparse_dir = scene_path / "sparse" / "0"
    print(f"\n[步骤 6/7] 正在从 '{colmap_sparse_dir}' 加载COLMAP相机和图像数据...")
    try:
        cameras = read_cameras_text(colmap_sparse_dir / "cameras.txt")
        images = read_images_text(colmap_sparse_dir / "images.txt")
    except FileNotFoundError:
        print(f"错误: 找不到 COLMAP 稀疏重建文件。")
        sys.exit(1)
    print("✓ COLMAP 数据加载成功。")
        
    # --- 7. 并行生成法线图 ---
    output_dir = scene_path / args.output_dir_name
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n[步骤 7/7] 正在并行生成法线图 (将保存到: '{output_dir}')")

    tasks = []
    for image in images.values():
        camera = cameras[image.camera_id]
        tasks.append((image, camera, xyz_world, normals_world, output_dir))

    # 使用手动指定的核心数，如果未指定则自动检测
    if args.num_cores > 0:
        num_cores_to_use = args.num_cores
        print(f"  - 使用手动指定的 {num_cores_to_use} 个CPU核心进行处理...")
    else:
        num_cores_to_use = multiprocessing.cpu_count()
        print(f"  - 自动检测到 {num_cores_to_use} 个CPU核心，开始并行处理...")
    
    with multiprocessing.Pool(processes=num_cores_to_use) as pool:
        results = []
        pbar = tqdm(pool.imap_unordered(process_and_save_normal_map_wrapper, tasks), total=len(tasks))
        for result in pbar:
            pbar.set_description(f"已处理 {result}")
            results.append(result)

    print(f"\n--- 成功! 所有法线图已生成在 '{output_dir}' ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从对齐后的真值点云并行生成高质量法线图，并显示详细进度。")
    parser.add_argument("scene_path", type=str,
                        help="单个场景的路径 (例如: data/courtyard)。")
    parser.add_argument("--alignment_file", type=str, default="eth3d_alignments_v2.json",
                        help="包含所有场景对齐变换矩阵的JSON文件。")
    parser.add_argument("--output_dir_name", type=str, default="gt_normals",
                        help="在场景文件夹内用于保存输出法线图的目录名。")
    parser.add_argument("--normal_radius", type=float, default=0.1,
                        help="Open3D中用于法线估计的邻域半径。")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                        help="用于点云降采样的体素大小（单位：米）。设置为0则不进行降采样。")
    parser.add_argument("--num_cores", type=int, default=0,
                        help="手动指定用于并行处理的CPU核心数。设置为0则自动检测。")
    
    args = parser.parse_args()
    main(args)