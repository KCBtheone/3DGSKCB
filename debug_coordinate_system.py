import argparse
import numpy as np
import torch
import os

# 从您的项目中导入所有必要的类和函数
from scene import Scene
from scene.gaussian_model import GaussianModel
# =================================================================================
# >>> [ 🚀 核心修复 ] <<<
# 导入正确的、在您文件中存在的函数名：readColmapScene
# =================================================================================
from scene.dataset_readers import readColmapScene

def print_matrix(name, matrix):
    """一个用于格式化打印矩阵的辅助函数。"""
    print(f"--- Matrix: {name} ---")
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    print(f"Shape: {matrix.shape}, Dtype: {matrix.dtype}")
    np.set_printoptions(precision=4, suppress=True)
    print(matrix)
    print("-" * (len(name) + 14))
    np.set_printoptions()

def run_coordinate_investigation_v2(cli_args):
    """
    深入数据加载流程，打印出每一步的坐标系变换矩阵。
    此版本与您的新版 dataset_readers.py 完全兼容。
    """
    print("=======================================================================")
    print("🚀 开始执行坐标系变换流程深度诊断 (v2)...")
    print(f"   分析场景: {cli_args.source_path}")
    print("=======================================================================")

    try:
        # --- 步骤 1: 直接调用 readColmapScene ---
        print("\n--- [ 阶段 1: 调用 readColmapScene 加载原始 COLMAP 数据 ] ---")
        scene_info = readColmapScene(path=cli_args.source_path, images=cli_args.images, eval=cli_args.eval)
        print("✅ readColmapScene 执行完毕。")

        if not scene_info.train_cameras:
            print("❌ 错误: 未能加载任何训练相机信息！"); return
        
        sample_cam_info = scene_info.train_cameras[0]
        print(f"\n--- [ 阶段 2: 分析样本相机 '{sample_cam_info.image_name}' 的 CamInfo 对象 ] ---")
        print_matrix("R (旋转矩阵, 来自 CamInfo)", sample_cam_info.R)
        print_matrix("T (平移向量, 来自 CamInfo)", sample_cam_info.T)
        
        # --- 步骤 2: 完整初始化 Scene 对象以获取最终渲染矩阵 ---
        print("\n--- [ 阶段 3: 完整初始化 Scene 对象以获取最终渲染矩阵 ] ---")

        class Args:
            def __init__(self):
                self.source_path = cli_args.source_path; self.model_path = cli_args.model_path if cli_args.model_path else "./output/coord_debug"
                self.images = cli_args.images; self.resolution = cli_args.resolution
                self.white_background = False; self.sh_degree = 3; self.eval = cli_args.eval; self.data_device = "cuda"
                self.convert_SHs_python = False; self.compute_cov3D_python = False
        
        args = Args()
        os.makedirs(args.model_path, exist_ok=True)
        gaussians = GaussianModel(sh_degree=args.sh_degree)
        
        # 您的 Scene.__init__ 会再次调用 readColmapScene，但没关系，我们的目标是获取最终的 Camera 对象
        scene = Scene(args, gaussians, shuffle=False)
        print("✅ Scene 对象初始化完毕。")
        
        if not scene.getTrainCameras():
            print("❌ 错误: Scene 对象中没有训练相机！"); return
            
        final_sample_cam = scene.getTrainCameras()[0]
        
        print(f"\n--- [ 阶段 4: 分析最终 Camera 对象的渲染矩阵 ] ---")
        
        print_matrix("world_view_transform (W2C 矩阵)", final_sample_cam.world_view_transform)
        print_matrix("projection_matrix (投影矩阵)", final_sample_cam.projection_matrix)
        print_matrix("full_proj_transform (完整变换矩阵)", final_sample_cam.full_proj_transform)

        # --- 步骤 3: 分析与解读 ---
        print("\n--- [ 阶段 5: 坐标系分析与解读 ] ---")
        
        w2c = final_sample_cam.world_view_transform.detach().cpu().numpy()
        c2w = np.linalg.inv(w2c)
        camera_center = c2w[:3, 3]
        print(f"根据 W2C 矩阵计算出的相机中心 (世界坐标): {np.array2string(camera_center, precision=4, suppress_small=True)}")

        R_w2c = w2c[:3, :3]
        if abs(R_w2c[1, 1]) < 0.2 and abs(R_w2c[2, 2]) < 0.2 and abs(abs(R_w2c[1, 2])) > 0.8:
            print("🚨 [高风险警告] 检测到 W2C 矩阵中可能存在 Y-Z 轴翻转！")
            print("   这通常是 NeRF++ (由内朝外) 坐标系变换的特征。对于 ETH3D 这类前向拍摄场景，")
            print("   这很可能是一个错误的变换，并会导致训练失败。")
        else:
            print("✅ W2C 矩阵的旋转部分看起来是常规的，未检测到明显的 Y-Z 轴翻转。")
        
        print("\n=======================================================================")
        print("🕵️ 坐标系侦探工作完成。")
        print("=======================================================================")

    except Exception as e:
        print(f"❌ 在诊断过程中发生错误！"); print(f"   错误信息: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS坐标系变换流程深度诊断工具 (v2)。")
    parser.add_argument("-s", "--source_path", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("-r", "--resolution", default=-1, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("-m", "--model_path", default="", type=str)
    
    cli_args = parser.parse_args()
    run_coordinate_investigation_v2(cli_args)