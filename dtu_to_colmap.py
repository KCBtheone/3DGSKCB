import numpy as np
import os
import sys

# 检查 scipy 是否安装，如果未安装则提前报错
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("\n❌ 错误: 依赖库 `scipy` 未安装。")
    print("请运行 `pip install scipy` 或 `conda install scipy` 进行安装。")
    sys.exit(1)

# ===================================================================================
#      DTU to COLMAP 格式转换脚本 (v2.1 - 批量处理版)
# ===================================================================================
# 功能:
# - 自动处理一个预设列表中的所有 DTU scan 场景。
# - 读取 DTU 特有的 `_cam.txt` 文件并转换为 COLMAP 格式。
# - 生成 `cameras.txt`, `images.txt`, 和 `points3D.txt`。
#
# 改进 (v2.1):
# - 将要处理的 scan 列表直接写在配置区，方便批量执行。
# - 增加了更详细的启动信息和结束总结。
# ===================================================================================


def read_dtu_cam_file(filepath):
    """读取单个 DTU 相机文件并解析内外参。"""
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        return intrinsics, extrinsics
    except Exception as e:
        print(f"    ❌ 错误: 读取或解析文件 {filepath} 失败: {e}")
        return None, None


def pose_world_to_colmap(extrinsics):
    """将 DTU 的 World-to-Camera 位姿转换为 COLMAP 的 Camera-to-World 位姿。"""
    R_world_cam = extrinsics[:3, :3]
    t_world_cam = extrinsics[:3, 3]
    R_cam_world = R_world_cam.T
    t_cam_world = -np.dot(R_cam_world, t_world_cam)
    quat_xyzw = R.from_matrix(R_cam_world).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz, t_cam_world


def convert_dtu_to_colmap(scan_path):
    """为指定的 DTU 扫描场景生成 COLMAP 格式的稀疏重建文件。"""
    print("\n" + "="*80)
    print(f"🚀 开始处理场景: {scan_path}")
    print("="*80)

    cams_path = os.path.join(scan_path, 'cams')
    images_path = os.path.join(scan_path, 'images')
    colmap_sparse_path = os.path.join(scan_path, 'sparse/0')

    print(f"[检查] 正在检查输入目录...")
    if not os.path.isdir(cams_path) or not os.path.isdir(images_path):
        print(f"❌ 严重错误: 找不到 'cams' 或 'images' 目录，跳过此场景。")
        return False

    os.makedirs(colmap_sparse_path, exist_ok=True)
    print(f"  ✅ 确保输出目录存在: {colmap_sparse_path}")

    cam_files = sorted([f for f in os.listdir(cams_path) if f.endswith('_cam.txt')])
    image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if len(cam_files) == 0:
        print(f"❌ 严重错误: 在 {cams_path} 中没有找到相机文件，跳过此场景。")
        return False
    print(f"  ✅ 发现 {len(cam_files)} 个相机文件和 {len(image_files)} 个图像文件。")

    # --- 生成 cameras.txt ---
    print("\n[步骤 1/3] 正在生成 cameras.txt...")
    width, height = 1600, 1200 # DTU 默认分辨率
    intrinsics, _ = read_dtu_cam_file(os.path.join(cams_path, cam_files[0]))
    if intrinsics is None: return False
    
    cameras_txt_path = os.path.join(colmap_sparse_path, 'cameras.txt')
    with open(cameras_txt_path, 'w') as f:
        line = f"1 SIMPLE_PINHOLE {width} {height} {intrinsics[0, 0]} {intrinsics[0, 2]} {intrinsics[1, 2]}\n"
        f.write(line)
    print(f"  ✅ 成功生成文件: {cameras_txt_path}")

    # --- 生成 images.txt ---
    print("\n[步骤 2/3] 正在生成 images.txt...")
    images_txt_path = os.path.join(colmap_sparse_path, 'images.txt')
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (cam_file, image_file) in enumerate(zip(cam_files, image_files)):
            _, extrinsics = read_dtu_cam_file(os.path.join(cams_path, cam_file))
            if extrinsics is None: continue
            q, t = pose_world_to_colmap(extrinsics)
            line = f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {image_file}\n\n"
            f.write(line)
    print(f"  ✅ 成功生成文件: {images_txt_path}")

    # --- 生成空的 points3D.txt ---
    print("\n[步骤 3/3] 正在生成空的 points3D.txt...")
    points3d_txt_path = os.path.join(colmap_sparse_path, 'points3D.txt')
    with open(points3d_txt_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    print(f"  ✅ 成功生成文件: {points3d_txt_path}")
    
    print(f"\n🎉 场景 {os.path.basename(scan_path)} 处理完成！")
    return True


if __name__ == "__main__":
    # --- [ 用户配置区 ] ---
    # 1. 确认 DTU 数据集根目录
    DTU_DATA_ROOT = './data/dtu'

    # 2. 这是根据您图片提取的所有 scan 编号，已按数字大小排好序
    SCANS_TO_PROCESS = [
        1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 
        62, 75, 77, 110, 114, 118
    ]
    # -----------------------

    print("==========================================================")
    print("      开始执行 DTU to COLMAP 批量数据转换流程")
    print(f"      目标根目录: {DTU_DATA_ROOT}")
    print(f"      将要处理 {len(SCANS_TO_PROCESS)} 个场景: {SCANS_TO_PROCESS}")
    print("==========================================================")

    if not os.path.isdir(DTU_DATA_ROOT):
        print(f"\n❌ 错误: DTU 数据集根目录不存在: {DTU_DATA_ROOT}")
        sys.exit(1)

    successful_scans = []
    failed_scans = []

    for scan_id in SCANS_TO_PROCESS:
        scan_name = f'scan{scan_id}'
        scan_path = os.path.join(DTU_DATA_ROOT, scan_name)
        
        if os.path.isdir(scan_path):
            if convert_dtu_to_colmap(scan_path):
                successful_scans.append(scan_id)
            else:
                failed_scans.append(scan_id)
        else:
            print(f"\n⚠️ 警告: 找不到场景目录，跳过: {scan_path}")
            failed_scans.append(scan_id)

    print("\n\n" + "#"*80)
    print("            批量处理流程执行完毕！")
    print("#"*80)
    print(f"\n✅ 成功处理场景数量: {len(successful_scans)}")
    if successful_scans:
        print(f"   列表: {successful_scans}")
    print(f"\n❌ 失败/跳过场景数量: {len(failed_scans)}")
    if failed_scans:
        print(f"   列表: {failed_scans}")
    
    print("\n下一步: 请使用 `colmap model_converter -h` 来确定正确的转换命令。")