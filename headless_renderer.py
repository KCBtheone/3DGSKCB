#
# Robust Headless Renderer for 3D Gaussian Splatting (v2)
# Fixes AttributeError by manually moving all tensors for non-Module GaussianModel
#

import os
import sys
import torch
import re
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

# --- 确保项目路径在sys.path中 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from arguments import ModelParams, PipelineParams, OptimizationParams 
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.system_utils import searchForMaxIteration
    from utils.general_utils import safe_state
except ImportError as e:
    print(f"❌ 导入模块失败，请确保此脚本位于gaussian-splatting项目根目录: {e}")
    sys.exit(1)

# ==================== 1. 配置加载函数 (不变) ====================
def load_config(cfg_path: str) -> dict:
    # ... (这部分代码与之前完全相同，此处省略以保持简洁) ...
    try:
        import pickle
        with open(cfg_path, 'rb') as f: args_namespace = pickle.load(f)
        return vars(args_namespace)
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
        except Exception as e: raise IOError(f"无法将 '{os.path.basename(cfg_path)}' 解析为任何已知格式。错误: {e}")


# ==================== 2. 渲染核心函数 (最终修复版) ====================
@torch.no_grad()
def render_and_save(model_path: str, output_path: str, source_path: str, iteration_name: str = "best"):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 渲染目标设备: {device}")
    print(f"开始处理模型: {model_path}")

    # --- 1. 加载模型配置 (不变) ---
    parser = ArgumentParser(description="渲染脚本参数加载器")
    model_params_def = ModelParams(parser)
    pipe_params_def = PipelineParams(parser)
    opt_params_def = OptimizationParams(parser)
    args_defaults = parser.parse_args([])

    cfg_path = os.path.join(model_path, "cfg_args")
    saved_cfg_dict = load_config(cfg_path)
    for k, v in saved_cfg_dict.items():
        if hasattr(args_defaults, k): setattr(args_defaults, k, v)
    
    args_defaults.source_path = source_path
    args_defaults.model_path = model_path
    if not hasattr(args_defaults, 'sh_degree'): args_defaults.sh_degree = 3
        
    model_params = model_params_def.extract(args_defaults)
    pipe_params = pipe_params_def.extract(args_defaults)
    opt_params = opt_params_def.extract(args_defaults)
    if not hasattr(pipe_params, 'debug'): safe_state(False); pipe_params.debug = False

    # --- 2. 从 .pth 检查点加载模型 ---
    load_iteration = iteration_name
    if load_iteration == "best":
        load_iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    
    chkpt_file = f"chkpnt{load_iteration}.pth"
    chkpt_path = os.path.join(model_path, chkpt_file)
    
    gaussians = GaussianModel(model_params.sh_degree)
    
    print(f"    -> [INFO] 正在从 '{chkpt_file}' 加载...")
    checkpoint = torch.load(chkpt_path, map_location="cpu")

    model_params_from_ckpt = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
    gaussians.restore(model_params_from_ckpt, opt_params) 

    # <<< ‼️‼️ 最终核心修复：手动将模型所有张量移动到正确设备 ‼️‼️ >>>
    print("    -> [INFO] 手动将所有模型张量移动到目标设备...")
    
    # 定义所有需要转移的张量属性名
    tensor_attributes = [
        '_xyz', '_features_dc', '_features_rest', 
        '_scaling', '_rotation', '_opacity',
        'max_radii2D', 'xyz_gradient_accum', 'denom'
    ]
    
    for attr_name in tensor_attributes:
        if hasattr(gaussians, attr_name):
            tensor = getattr(gaussians, attr_name)
            if isinstance(tensor, torch.Tensor):
                # 统一处理 Parameter 和 Tensor
                if isinstance(tensor, torch.nn.Parameter):
                    setattr(gaussians, attr_name, torch.nn.Parameter(tensor.to(device)))
                else:
                    setattr(gaussians, attr_name, tensor.to(device))
    
    print(f"    -> [成功] 模型加载成功，包含 {gaussians.get_xyz.shape[0]} 个点。")
    print(f"    -> [INFO] 模型已完全转移到 {device}.")

    # --- 3. 加载场景和训练相机 ---
    print("  -> 步骤3: 加载场景和训练相机...")
    scene = Scene(model_params, gaussians, shuffle=False)
    train_cameras = scene.getTrainCameras()
    print(f"    -> [成功] 成功加载 {len(train_cameras)} 个训练相机。")
    
    # --- 4. 准备渲染 ---
    print("  -> 步骤4: 准备渲染...")
    os.makedirs(output_path, exist_ok=True)
    background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device=device)

    # --- 5. 循环渲染并保存 ---
    print("  -> 步骤5: 开始渲染并保存图像...")
    for camera in tqdm(train_cameras, desc="渲染训练视角"):
        # 确保相机内部的张量也在GPU上
        camera.to_device(device)

        render_pkg = render(camera, gaussians, pipe_params, background)
        rendered_image_tensor = render_pkg["render"].clamp(0.0, 1.0)
        
        rendered_np = (rendered_image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(rendered_np)

        image_name = os.path.splitext(camera.image_name)[0]
        output_filename = os.path.join(output_path, f"{image_name}.png")
        pil_image.save(output_filename)

    print(f"\n✅ 渲染完成！所有 {len(train_cameras)} 张训练视角图像已保存至 '{output_path}'。")

if __name__ == "__main__":
    parser = ArgumentParser(description="加载一个3DGS模型并渲染其所有训练视角。")
    parser.add_argument("model_path", type=str, help="指向模型实验目录的路径。")
    parser.add_argument("output_path", type=str, help="保存渲染图像的输出文件夹路径。")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="【必需】指向原始场景数据源的路径。")
    parser.add_argument("--iteration", type=str, default="15000", help="要加载的模型迭代次数 (例如 '15000')。")
    parser.add_argument("--resolution", type=int, default=-1, help="渲染分辨率缩放比例。-1 表示使用训练时的分辨率。")
    
    args = parser.parse_args()

    # 为了让Camera类能正确移动到设备，添加一个辅助方法
    from scene.cameras import Camera
    def camera_to_device(self, device):
        for attr in ['world_view_transform', 'full_proj_transform', 'camera_center']:
            if hasattr(self, attr):
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(self, attr, tensor.to(device))
    Camera.to_device = camera_to_device
    
    render_and_save(args.model_path, args.output_path, args.source_path, args.iteration)