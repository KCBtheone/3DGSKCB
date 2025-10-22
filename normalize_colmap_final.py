import numpy as np
import os
import argparse
import shutil

# 直接从您的代码库中导入 colmap_loader 模块
from scene import colmap_loader

# --- 用于写入二进制COLMAP文件的辅助函数 (与之前版本相同) ---
def write_points3D_binary(path, points3D_xyz, points3D_rgb):
    with open(path, "wb") as fid:
        num_points = len(points3D_xyz)
        fid.write(np.uint64(num_points).tobytes())
        for i in range(num_points):
            fid.write(np.uint64(i + 1).tobytes())
            fid.write(points3D_xyz[i].astype(np.float64).tobytes())
            fid.write(points3D_rgb[i].astype(np.uint8).tobytes())
            fid.write(np.float64(0.0).tobytes())
            fid.write(np.uint64(0).tobytes())

def write_images_binary(path, images):
    with open(path, "wb") as fid:
        fid.write(np.uint64(len(images)).tobytes())
        for img_id, img in sorted(images.items()):
            fid.write(np.int32(img.id).tobytes())
            fid.write(img.qvec.astype(np.float64).tobytes())
            fid.write(img.tvec.astype(np.float64).tobytes())
            fid.write(np.int32(img.camera_id).tobytes())
            name_bytes = img.name.encode('utf-8')
            fid.write(name_bytes)
            fid.write(b'\x00') 
            fid.write(np.uint64(0).tobytes())

def normalize_and_compensate_scene(source_dir: str):
    """
    读取COLMAP场景，进行正确的点云中心归一化，并主动补偿
    Scene加载器中不当的NeRF++式中心化，最终生成一个可直接用于训练的
    、坐标系正确的COLMAP模型。
    """
    print(f"--- [COLMAP场景归一化与补偿工具 (最终版)] ---")
    print(f"源场景目录: {source_dir}")

    sparse_dir = os.path.join(source_dir, "sparse/0")
    if not os.path.exists(sparse_dir) and os.path.exists(os.path.join(source_dir, "sparse/0_original_huge")):
        print("检测到已备份的原始数据，将使用 'sparse/0_original_huge' 作为输入。")
        sparse_dir = os.path.join(source_dir, "sparse/0_original_huge")
    
    if not os.path.exists(sparse_dir):
        print(f"❌ [错误] 找不到 'sparse/0' 或 'sparse/0_original_huge' 目录: {sparse_dir}")
        return

    # 1. 加载原始数据
    print("正在加载原始COLMAP模型...")
    points_path = os.path.join(sparse_dir, "points3D.bin")
    images_path = os.path.join(sparse_dir, "images.bin")
    cameras_path = os.path.join(sparse_dir, "cameras.bin")

    try:
        points3D_xyz, points3D_rgb, _ = colmap_loader.read_points3D_binary(points_path)
        images = colmap_loader.read_extrinsics_binary(images_path)
        print(f"✅ 加载成功: {len(points3D_xyz)} 个点, {len(images)} 张图像。")
    except Exception as e:
        print(f"❌ [错误] 加载COLMAP文件时出错: {e}"); return

    # 2. 计算正确的归一化参数 (基于点云中心)
    print("正在计算正确的归一化参数 (基于点云)...")
    point_cloud_center = points3D_xyz.mean(axis=0)
    distances = np.linalg.norm(points3D_xyz - point_cloud_center, axis=1)
    normalization_radius = np.percentile(distances, 95)
    print(f"   - 真实点云中心: {point_cloud_center}")
    print(f"   - 归一化半径: {normalization_radius}")

    # 3. 模拟 Scene 加载器的错误行为，计算出那个错误的平移量
    print("正在模拟 Scene 加载器的错误行为以计算补偿量...")
    cam_centers_original = []
    for img in images.values():
        R = colmap_loader.qvec2rotmat(img.qvec)
        C = -np.dot(R.T, img.tvec)
        cam_centers_original.append(C.reshape(3, 1))
    
    nerfpp_center = np.mean(np.hstack(cam_centers_original), axis=1)
    print(f"   - NeRF++ (错误的) 相机中心: {nerfpp_center}")

    # 4. 对点云和相机应用“正确归一化”+“主动补偿”
    print("正在应用归一化与补偿...")
    
    # 归一化点云 (只使用正确的点云中心)
    points3D_xyz_final = (points3D_xyz - point_cloud_center) / normalization_radius

    # 归一化并补偿相机
    images_final = {}
    for img_id, img in images.items():
        R = colmap_loader.qvec2rotmat(img.qvec)
        C = -np.dot(R.T, img.tvec)
        
        # 核心逻辑：
        # Scene加载器最终会计算相机中心为： (C + translate) / radius
        # 其中 translate = -nerfpp_center, radius 是基于相机距离计算的，与我们的 normalization_radius 不同但数量级相似
        # 我们的目标是让最终结果约等于 (C - point_cloud_center) / normalization_radius
        #
        # 我们的新相机中心 C_new, 在加载后会变成 (C_new - nerfpp_center) / radius
        # 我们希望 (C_new - nerfpp_center) / radius ≈ (C - point_cloud_center) / normalization_radius
        # 忽略 radius 的微小差异，我们得到 C_new ≈ C - point_cloud_center + nerfpp_center
        #
        # 所以，我们先对 C 应用这个变换，然后再归一化
        
        C_compensated = C - point_cloud_center + nerfpp_center
        
        # 现在，我们将这个“预补偿”过的相机中心，用我们正确的参数进行归一化
        C_final = (C_compensated - nerfpp_center) / normalization_radius
        
        # 重新计算最终的 tvec
        tvec_final = -np.dot(R, C_final)
        
        images_final[img_id] = img._replace(tvec=tvec_final)
        
    print("✅ 变换完成。")

    # 5. 将最终结果写入新模型
    output_dir = os.path.join(os.path.dirname(sparse_dir), "0_compensated")
    print(f"正在将最终模型写入新目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    write_points3D_binary(os.path.join(output_dir, "points3D.bin"), points3D_xyz_final, points3D_rgb)
    write_images_binary(os.path.join(output_dir, "images.bin"), images_final)
    shutil.copy(cameras_path, os.path.join(output_dir, "cameras.bin"))
    
    print("\n--- [🎉 完成] ---")
    print("已生成经过归一化和补偿的场景！")
    print("\n下一步操作建议:")
    print("1. 备份并替换您原始的 'sparse/0' 文件夹:")
    print(f"   (如果存在) mv {source_dir}/sparse/0 {source_dir}/sparse/0_backup")
    print(f"   mv {output_dir} {source_dir}/sparse/0")
    print("2. 使用 'debug_coordinate_system.py' 脚本对新生成的 'sparse/0' 进行最终验证，")
    print("   这次相机中心应该接近于它在归一化空间中的真实位置，而不是 [0,0,0]。")
    print("3. 如果验证通过，即可开始训练。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="归一化COLMAP场景并补偿NeRF++中心化错误。")
    parser.add_argument("source_dir", type=str, help="场景根目录的路径 (例如, 'data/courtyard')。")
    args = parser.parse_args()
    
    if not os.path.isdir(args.source_dir):
        print(f"❌ [错误] 提供的路径不是一个有效的目录: {args.source_dir}")
    else:
        normalize_and_compensate_scene(args.source_dir)