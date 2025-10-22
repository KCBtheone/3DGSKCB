import json
import numpy as np
import os
import sys
from PIL import Image

# 检查依赖库是否安装
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("\n❌ 错误: 依赖库 `scipy` 未安装。")
    print("请运行 `pip install scipy` 或 `conda install scipy` 进行安装。")
    sys.exit(1)

# ===================================================================================
#      NeRF Synthetic to COLMAP 格式转换脚本 (v1.0)
# ===================================================================================
# 功能:
# - 自动扫描指定 NeRF Synthetic 数据集目录下的所有场景 (chair, drums, etc.)。
# - 读取 'transforms_train.json' 文件。
# - 将 NeRF 的相机参数 (FOV, c2w 矩阵) 转换为 COLMAP .txt 格式。
# - 自动处理 NeRF 和 COLMAP 之间的坐标系差异。
#
# 使用方法:
# 1. 修改底部的 `--- [ 用户配置区 ] ---` 中的 `NERF_SYNTHETIC_ROOT` 路径。
# 2. 运行 `python nerf_to_colmap.py`。
# ===================================================================================

def convert_nerf_scene_to_colmap(scene_path):
    """
    为单个 NeRF Synthetic 场景生成 COLMAP 格式的稀疏重建文件。
    """
    print("\n" + "="*80)
    print(f"🚀 开始处理场景: {scene_path}")
    print("="*80)

    # --- 1. 路径和文件检查 ---
    json_path = os.path.join(scene_path, 'transforms_train.json')
    colmap_sparse_path = os.path.join(scene_path, 'sparse/0')

    print(f"[检查] 正在检查输入文件...")
    if not os.path.exists(json_path):
        print(f"❌ 严重错误: 找不到 'transforms_train.json' 文件: {json_path}")
        print("  -> 请确保这是一个标准的 NeRF Synthetic 数据集场景。跳过此场景。")
        return False

    print(f"  ✅ 找到 'transforms_train.json'。")
    os.makedirs(colmap_sparse_path, exist_ok=True)
    print(f"  ✅ 确保输出目录存在: {colmap_sparse_path}")

    # --- 2. 加载 JSON 数据 ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 严重错误: 解析 JSON 文件失败: {e}")
        return False

    # --- 3. 生成 cameras.txt ---
    print("\n[步骤 1/3] 正在生成 cameras.txt...")
    
    # 从 JSON 获取相机视场角
    camera_angle_x = data.get('camera_angle_x')
    if camera_angle_x is None:
        print("❌ 严重错误: JSON 文件中缺少 'camera_angle_x'。")
        return False

    # 获取图像尺寸
    try:
        first_frame = data['frames'][0]
        image_relative_path = first_frame['file_path']
        # 兼容 './train/r_0' 和 'train/r_0' 两种格式
        image_filename = f"{image_relative_path.split('/')[-1]}.png"
        image_dir = os.path.dirname(image_relative_path)
        image_path = os.path.join(scene_path, image_dir, image_filename)
        
        with Image.open(image_path) as img:
            width, height = img.size
        print(f"  - 从图像 '{image_path}' 获取尺寸: {width}x{height}")
    except Exception as e:
        print(f"⚠️ 警告: 无法自动获取图像尺寸: {e}。将使用默认值 800x800。")
        width, height = 800, 800

    # 计算焦距
    focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)
    cx = width / 2.0
    cy = height / 2.0
    print(f"  - 计算得到的内参: fx={focal_length:.4f}, cx={cx:.2f}, cy={cy:.2f}")

    cameras_txt_path = os.path.join(colmap_sparse_path, 'cameras.txt')
    with open(cameras_txt_path, 'w') as f:
        line = f"1 PINHOLE {width} {height} {focal_length} {focal_length} {cx} {cy}\n"
        f.write(line)
    print(f"  ✅ 成功生成文件: {cameras_txt_path}")

    # --- 4. 生成 images.txt ---
    print("\n[步骤 2/3] 正在生成 images.txt...")
    images_txt_path = os.path.join(colmap_sparse_path, 'images.txt')
    
    # NeRF (OpenCV/Blender) 和 COLMAP 之间的坐标系转换矩阵
    # NeRF: [right, up, backwards] -> COLMAP: [right, down, forwards]
    # 需要绕 x 轴旋转 180 度
    nerf_to_colmap_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        frames = sorted(data['frames'], key=lambda x: x['file_path'])
        for i, frame in enumerate(frames):
            image_id = i + 1
            
            # 获取 NeRF 的 c2w 矩阵
            c2w_nerf = np.array(frame['transform_matrix'])
            
            # 应用坐标系转换
            c2w_colmap = c2w_nerf @ nerf_to_colmap_transform
            
            # 提取旋转和平移
            R_mat = c2w_colmap[:3, :3]
            t_vec = c2w_colmap[:3, 3]
            
            # 转换为四元数 (qw, qx, qy, qz)
            quat_xyzw = R.from_matrix(R_mat).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            # 获取图像文件名
            image_relative_path = frame['file_path']
            image_filename = f"{os.path.basename(image_relative_path)}.png"
            image_dir_name = os.path.basename(os.path.dirname(image_relative_path))
            full_image_name = os.path.join(image_dir_name, image_filename)

            # 写入文件
            qw, qx, qy, qz = quat_wxyz
            tx, ty, tz = t_vec
            line = f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {full_image_name}\n\n"
            f.write(line)

    print(f"  ✅ 成功为 {len(frames)} 张图像生成位姿信息。")
    print(f"  ✅ 成功生成文件: {images_txt_path}")

    # --- 5. 生成空的 points3D.txt ---
    print("\n[步骤 3/3] 正在生成空的 points3D.txt...")
    points3d_txt_path = os.path.join(colmap_sparse_path, 'points3D.txt')
    with open(points3d_txt_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    print(f"  ✅ 成功生成文件: {points3d_txt_path}")
    
    print(f"\n🎉 场景 {os.path.basename(scene_path)} 处理完成！")
    print(f"🎉 COLMAP (.txt) 文件已生成于: {colmap_sparse_path}")
    return True

if __name__ == "__main__":
    # --- [ 用户配置区 ] ---
    # 1. 请将此路径修改为您的 nerf_synthetic 数据集所在的根目录
    NERF_SYNTHETIC_ROOT = './data/nerf_synthetic'
    # -----------------------

    print("==========================================================")
    print("      开始执行 NeRF Synthetic to COLMAP 数据转换流程")
    print(f"      目标根目录: {NERF_SYNTHETIC_ROOT}")
    print("==========================================================")

    if not os.path.isdir(NERF_SYNTHETIC_ROOT):
        print(f"\n❌ 错误: 数据集根目录不存在: {NERF_SYNTHETIC_ROOT}")
        print("请检查 `NERF_SYNTHETIC_ROOT` 变量是否设置正确。")
        sys.exit(1)

    # 自动查找所有场景文件夹
    try:
        scenes_to_process = sorted([d for d in os.listdir(NERF_SYNTHETIC_ROOT) if os.path.isdir(os.path.join(NERF_SYNTHETIC_ROOT, d))])
        print(f"      发现 {len(scenes_to_process)} 个场景: {scenes_to_process}\n")
    except FileNotFoundError:
        scenes_to_process = []
    
    if not scenes_to_process:
        print("⚠️ 警告: 在指定目录下没有找到任何场景文件夹。")
        sys.exit(0)

    successful_scenes = []
    failed_scenes = []

    for scene_name in scenes_to_process:
        scene_path = os.path.join(NERF_SYNTHETIC_ROOT, scene_name)
        if convert_nerf_scene_to_colmap(scene_path):
            successful_scenes.append(scene_name)
        else:
            failed_scenes.append(scene_name)

    print("\n\n" + "#"*80)
    print("            批量处理流程执行完毕！")
    print("#"*80)
    print(f"\n✅ 成功处理场景数量: {len(successful_scenes)}")
    if successful_scenes:
        print(f"   列表: {successful_scenes}")
    print(f"\n❌ 失败/跳过场景数量: {len(failed_scenes)}")
    if failed_scenes:
        print(f"   列表: {failed_scenes}")
    
    print("\n下一步: 您现在可以尝试对这些场景运行您原来的 `run` 脚本了。")
    print("如果 `run` 脚本中的 `colmap model_converter` 步骤失败，")
    print("请使用我们之前讨论过的 `prepare_dtu_scene.py` 脚本中的转换逻辑来手动生成 .bin 文件。")