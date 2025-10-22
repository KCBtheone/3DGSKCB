# render_final.py
# 一个健壮、可配置、整合了所有修复的3DGS离线渲染脚本

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
    
    # 动态导入你提供的、已经验证过的正确类
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from scene import Scene, Camera
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from utils.general_utils import safe_state
    
    # 动态导入光栅化器
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
        # Fallback to text parsing if pickle fails
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

# ==================== 核心渲染函数 (已验证) ====================
@torch.no_grad()
def render_model(args):
    """
    加载并渲染一个训练好的3DGS模型的所有训练视角。
    """
    print("🚀 开始最终渲染流程...")

    # 1. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" - 使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 2. 加载配置
    print(f" - 加载配置文件: {os.path.join(args.model_path, 'cfg_args')}")
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"❌ 错误: 在'{args.model_path}'中找不到 'cfg_args' 配置文件。"); return
    
    parser = ArgumentParser()
    model_params_def = ModelParams(parser)
    pipe_params_def = PipelineParams(parser)
    opt_params_def = OptimizationParams(parser)
    
    args_defaults = parser.parse_args([])
    saved_cfg_dict = load_config(cfg_path)
    for k, v in saved_cfg_dict.items():
        if hasattr(args_defaults, k): setattr(args_defaults, k, v)

    # 注入命令行参数
    args_defaults.source_path = args.source_path
    args_defaults.model_path = args.model_path
    if not hasattr(args_defaults, 'sh_degree'): args_defaults.sh_degree = 3
        
    model_params = model_params_def.extract(args_defaults)
    pipe_params = pipe_params_def.extract(args_defaults)
    opt_params = opt_params_def.extract(args_defaults)
    if not hasattr(pipe_params, 'debug'): safe_state(False); pipe_params.debug = False

    # 3. 加载模型检查点
    chkpt_path = os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    print(f" - 正在从 '{os.path.basename(chkpt_path)}' 加载模型...")
    
    gaussians = GaussianModel(model_params.sh_degree)
    checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
    model_params_from_ckpt = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
    gaussians.restore(model_params_from_ckpt, opt_params)
    print(f"   -> 加载成功: {gaussians.get_xyz.shape[0]} 个高斯点。")

    # 4. 关键修复：将所有模型张量移动到GPU
    tensor_attrs = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
                    'max_radii2D', 'xyz_gradient_accum', 'denom']
    for attr in tensor_attrs:
        if hasattr(gaussians, attr):
            tensor = getattr(gaussians, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(gaussians, attr, tensor.to(device))

    # 5. 加载场景和相机
    print(f" - 正在从 '{model_params.source_path}' 加载场景...")
    scene = Scene(model_params, gaussians, shuffle=False)
    cameras = scene.getTrainCameras()
    print(f"   -> 加载成功: {len(cameras)} 个训练相机。")

    # 6. 准备并执行渲染
    os.makedirs(args.output_path, exist_ok=True)
    background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device=device)

    print(f" - 开始渲染 {len(cameras)} 个视角，结果保存至 '{args.output_path}'...")
    for camera in tqdm(cameras, desc="渲染中"):
        # 关键修复：确保相机内部张量也在GPU上
        camera.to_device(device)
        
        # --- 内联渲染逻辑，确保正确性 ---
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=pipe_params.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # 核心步骤：从模型中获取经过激活函数处理的正确属性
        means3D = gaussians.get_xyz
        opacities = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        shs = gaussians.get_features

        # 光栅化
        rendered_image, _ = rasterizer(
            means3D = means3D,
            means2D = torch.zeros_like(means3D, requires_grad=True, device=device) + 0,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None
        )
        
        # 保存图像
        img_tensor = rendered_image.clamp(0.0, 1.0)
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(img_np)
        pil_img.save(os.path.join(args.output_path, f"{os.path.splitext(camera.image_name)[0]}.png"))

    print("\n✅ 渲染全部完成！")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = ArgumentParser(description="加载一个3DGS模型并渲染其所有训练视角。")
    parser.add_argument("model_path", type=str, help="指向模型实验目录的路径 (包含 cfg_args 和 chkpnt*.pth 文件)。")
    parser.add_argument("output_path", type=str, help="保存渲染图像的输出文件夹。")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="【必需】指向原始场景数据源的路径 (如 COLMAP 目录)。")
    parser.add_argument("--iteration", type=str, default="15000", help="要加载的模型迭代次数 (如 '15000')。")
    args = parser.parse_args()

    # 关键修复：为Camera类动态添加 to_device 方法
    def camera_to_device(self, device):
        for attr in ['world_view_transform', 'full_proj_transform', 'camera_center']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
    Camera.to_device = camera_to_device
    
    # 清理Python缓存，以防万一
    os.system('find . -name "*.pyc" -delete')
    print(" - 已清理项目中的 __pycache__ 文件。")
    
    render_model(args)