# render_test_from_ply.py
# 一个专门从 .ply 文件加载模型并渲染测试集视角的脚本

import os
import sys
import torch
import re
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import math

# --- 步骤 1: 确保项目路径在sys.path中并导入核心模块 ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from scene import Scene
    from arguments import ModelParams, PipelineParams
    from utils.general_utils import safe_state
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    print("[SUCCESS] 核心模块导入成功。")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}\n请确保此脚本位于 3D Gaussian Splatting 项目的根目录。")
    sys.exit(1)

# ==================== 配置加载函数 (已验证) ====================
def load_config(cfg_path: str) -> dict:
    try:
        import pickle
        with open(cfg_path, 'rb') as f: return vars(pickle.load(f))
    except Exception:
        try:
            with open(cfg_path, 'r') as f: content = f.read().strip()
            if content.startswith("Namespace(") and content.endswith(")"): content = content[10:-1]
            pattern = re.compile(r"(\w+)\s*=\s*('([^']*)'|\"([^\"]*)\"|\[.*?\]|[\w.-]+|True|False|None)")
            matches = pattern.findall(content)
            cfg_dict = {}
            for key, val_group, str_val1, str_val2 in matches:
                val_str = val_group
                if (val_str.startswith("'") and val_str.endswith("'")) or (val_str.startswith('"') and val_str.endswith('"')): cfg_dict[key] = val_str[1:-1]
                elif val_str == 'True': cfg_dict[key] = True
                elif val_str == 'False': cfg_dict[key] = False
                elif val_str == 'None': cfg_dict[key] = None
                elif val_str.startswith('[') and val_str.endswith(']'):
                    try: cfg_dict[key] = eval(val_str)
                    except: cfg_dict[key] = val_str
                else:
                    try:
                        if '.' in val_str: cfg_dict[key] = float(val_str)
                        else: cfg_dict[key] = int(val_str)
                    except ValueError: cfg_dict[key] = val_str
            return cfg_dict
        except Exception as e: raise IOError(f"无法解析 '{os.path.basename(cfg_path)}': {e}")

# ==================== 核心渲染函数 ====================
@torch.no_grad()
def render_test_set(args):
    """
    加载模型并渲染测试集的所有视角。
    """
    print("🚀 开始从 PLY 文件渲染测试集...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" - 使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. 加载配置
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"❌ 错误: 在'{args.model_path}'中找不到 'cfg_args' 配置文件。"); return
    
    parser = ArgumentParser()
    model_params_def = ModelParams(parser)
    pipe_params_def = PipelineParams(parser)
    args_defaults = parser.parse_args([])
    saved_cfg_dict = load_config(cfg_path)
    for k, v in saved_cfg_dict.items():
        if hasattr(args_defaults, k): setattr(args_defaults, k, v)

    args_defaults.source_path = args.source_path
    args_defaults.model_path = args.model_path
    model_params = model_params_def.extract(args_defaults)
    pipe_params = pipe_params_def.extract(args_defaults)
    
    # 2. 从 PLY 文件加载模型
    print(f" - 正在从 PLY 文件加载模型: {args.ply_path}")
    if not os.path.exists(args.ply_path):
        print(f"❌ 错误: PLY 文件不存在: {args.ply_path}"); return
    
    gaussians_from_ply = GaussianModel(model_params.sh_degree)
    gaussians_from_ply.load_ply(args.ply_path)
    print(f"   -> 从 PLY 加载成功: {gaussians_from_ply.get_xyz.shape[0]} 个高斯点。")

    # 3. 创建 Scene 并注入模型
    print(f" - 正在从 '{model_params.source_path}' 加载场景和相机...")
    # 关键：确保 eval=True，这样 Colmap 加载器才会正确分割训练/测试集
    model_params.eval = True 
    scene = Scene(model_params, GaussianModel(model_params.sh_degree), load_iteration=None, shuffle=False)
    scene.gaussians = gaussians_from_ply
    print("   -> 已成功将 PLY 加载的模型注入 Scene 对象。")
    
    # ‼️‼️‼️ 核心修改：调用 getTestCameras() ‼️‼️‼️
    cameras = scene.getTestCameras()
    if not cameras:
        print("❌ 错误: 未能加载任何测试相机。请检查你的数据源是否包含测试集分割。")
        return
    print(f"   -> 加载成功: {len(cameras)} 个测试相机。")

    # 4. 准备并执行渲染
    os.makedirs(args.output_path, exist_ok=True)
    background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device=device)

    print(f" - 开始渲染 {len(cameras)} 个视角，结果保存至 '{args.output_path}'...")
    for camera in tqdm(cameras, desc="渲染测试集"):
        camera.to_device(device)
        
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height), image_width=int(camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy, bg=background, scale_modifier=1.0,
            viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=scene.gaussians.active_sh_degree, campos=camera.camera_center,
            prefiltered=False, debug=getattr(pipe_params, 'debug', False),
            antialiasing=getattr(pipe_params, 'antialiasing', False)
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D, opacities, scales, rotations, shs = (
            scene.gaussians.get_xyz, scene.gaussians.get_opacity, scene.gaussians.get_scaling,
            scene.gaussians.get_rotation, scene.gaussians.get_features
        )
        
        rasterizer_outputs = rasterizer(
            means3D=means3D, means2D=torch.zeros_like(means3D, requires_grad=True, device=device)+0,
            shs=shs, colors_precomp=None, opacities=opacities, scales=scales,
            rotations=rotations, cov3D_precomp=None
        )
        rendered_image = rasterizer_outputs[0]
        
        # 保存图像
        img_tensor = rendered_image.clamp(0.0, 1.0)
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(img_np)
        pil_img.save(os.path.join(args.output_path, f"{os.path.splitext(camera.image_name)[0]}.png"))

    print("\n✅ 测试集渲染全部完成！")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = ArgumentParser(description="从 .ply 文件加载3DGS模型并渲染其 **测试集** 视角。")
    parser.add_argument("model_path", type=str, help="指向模型实验目录的路径 (用于加载 cfg_args)。")
    parser.add_argument("ply_path", type=str, help="指向要渲染的 point_cloud.ply 文件的完整路径。")
    parser.add_argument("output_path", type=str, help="保存渲染图像的输出文件夹。")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="【必需】指向原始场景数据源的路径 (如 COLMAP 目录)。")
    args = parser.parse_args()

    def camera_to_device(self, device):
        for attr in ['world_view_transform', 'full_proj_transform', 'camera_center']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
    Camera.to_device = camera_to_device
    
    render_test_set(args)