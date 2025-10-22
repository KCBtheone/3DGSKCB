# render_debug_v3.py
# 增加了对 scaling 和 opacity 数值状态的检查

import os
import sys
import torch
import re
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import time

# --- 步骤 0: 确保项目路径在sys.path中 ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"[DEBUG] 脚本启动，已将项目根目录 '{project_root}' 添加到 sys.path")
    
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.system_utils import searchForMaxIteration
    from utils.general_utils import safe_state
    from scene.cameras import Camera
    print("[SUCCESS] 核心模块导入成功！环境设置正确。")
except ImportError as e:
    print(f"❌ [致命错误] 导入核心模块失败: {e}")
    print("   -> 请确保此脚本位于 3D Gaussian Splatting 项目的根目录。")
    sys.exit(1)

# ==================== 配置加载函数 (不变) ====================
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
        except Exception as e: raise IOError(f"无法将 '{os.path.basename(cfg_path)}' 解析为任何已知格式。错误: {e}")

# ==================== 核心渲染函数 (带详细调试) ====================
@torch.no_grad()
def render_and_save_debug(model_path: str, output_path: str, source_path: str, iteration: str):
    print("\n" + "="*80)
    print("🚀 开始执行终极调试渲染脚本 (v3 - 检查数据状态) 🚀")
    print("="*80 + "\n")

    # --- 步骤 1: 环境和设备检查 (不变) ---
    print("--- [步骤 1/7] 环境和设备检查 ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] CUDA 可用！将使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("-" * 40)
    
    # --- 步骤 2: 加载模型配置 (不变) ---
    print("\n--- [步骤 2/7] 加载模型配置文件 (cfg_args) ---")
    # ... (代码与 v2 完全相同, 此处省略以保持简洁)
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"❌ [致命错误] 配置文件 'cfg_args' 不存在！"); return
    parser = ArgumentParser()
    model_params_def = ModelParams(parser)
    pipe_params_def = PipelineParams(parser)
    opt_params_def = OptimizationParams(parser)
    args_defaults = parser.parse_args([])
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
    print("[SUCCESS] 模型参数和渲染管线参数已成功构建。")
    print("-" * 40)
    
    # --- 步骤 3: 加载模型检查点 (不变) ---
    print(f"\n--- [步骤 3/7] 加载模型检查点 (iteration: {iteration}) ---")
    chkpt_file = f"chkpnt{iteration}.pth"
    chkpt_path = os.path.join(model_path, chkpt_file)
    if not os.path.exists(chkpt_path):
        print(f"❌ [致命错误] 检查点文件不存在: {chkpt_path}"); return
    gaussians = GaussianModel(model_params.sh_degree)
    checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
    model_params_from_ckpt = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
    gaussians.restore(model_params_from_ckpt, opt_params)
    print(f"[SUCCESS] 模型加载成功，包含 {gaussians.get_xyz.shape[0]} 个高斯点。")
    print("-" * 40)

    # --- ‼️‼️ 新增步骤 4: 检查关键张量的数值状态 ‼️‼️ ---
    print("\n--- [新增步骤 4/7] 检查加载后张量的数值状态 (在CPU上) ---")
    scaling_tensor = gaussians._scaling
    opacity_tensor = gaussians._opacity
    
    print("[INFO] 这是在任何激活函数（如exp/sigmoid）应用之前，直接从 .pth 文件恢复的值。")
    
    print("\n--- 检查 Scaling (_scaling) ---")
    print(f"   - 类型: {scaling_tensor.dtype}, 设备: {scaling_tensor.device}")
    print(f"   - 形状: {scaling_tensor.shape}")
    print(f"   - 最小值 (Min): {scaling_tensor.min().item():.6f}")
    print(f"   - 最大值 (Max): {scaling_tensor.max().item():.6f}")
    print(f"   - 均值 (Mean): {scaling_tensor.mean().item():.6f}")
    print(f"   -> [诊断] 正常情况下，这些值应该是较小的负数 (例如，均值在 -3 到 -6 之间)。如果它们是大的正数或非常接近0，可能存在问题。")

    print("\n--- 检查 Opacity (_opacity) ---")
    print(f"   - 类型: {opacity_tensor.dtype}, 设备: {opacity_tensor.device}")
    print(f"   - 形状: {opacity_tensor.shape}")
    print(f"   - 最小值 (Min): {opacity_tensor.min().item():.6f}")
    print(f"   - 最大值 (Max): {opacity_tensor.max().item():.6f}")
    print(f"   - 均值 (Mean): {opacity_tensor.mean().item():.6f}")
    print(f"   -> [诊断] 正常情况下，这些值应该在 0 附近。如果均值非常大或非常小，可能存在问题。")
    print("-" * 40)
    
    # --- 步骤 5: 将模型所有张量移动到GPU (不变) ---
    print(f"\n--- [步骤 5/7] 将模型张量手动移动到设备: {device} ---")
    tensor_attr_names = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
                         'max_radii2D', 'xyz_gradient_accum', 'denom']
    for attr_name in tensor_attr_names:
        if hasattr(gaussians, attr_name):
            tensor = getattr(gaussians, attr_name)
            if isinstance(tensor, torch.Tensor):
                setattr(gaussians, attr_name, tensor.to(device))
    print("[SUCCESS] 所有必需的模型张量已检查并尝试移动到目标设备。")
    print("-" * 40)

    # --- 步骤 6: 加载场景和相机 (不变) ---
    print("\n--- [步骤 6/7] 加载场景和相机 ---")
    print(f"[INFO] 使用数据源路径: {model_params.source_path}")
    scene = Scene(model_params, gaussians, shuffle=False)
    train_cameras = scene.getTrainCameras()
    if not train_cameras:
        print(f"❌ [致命错误] 未能加载任何训练相机！"); return
    print(f"[SUCCESS] 成功加载 {len(train_cameras)} 个训练相机。")
    print("-" * 40)
    
    # --- 步骤 7: 循环渲染并保存 (不变) ---
    print("\n--- [步骤 7/7] 开始渲染循环 ---")
    os.makedirs(output_path, exist_ok=True)
    background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device=device)
    
    for idx, camera in enumerate(tqdm(train_cameras, desc="渲染训练视角")):
        camera.to_device(device)
        render_pkg = render(camera, gaussians, pipe_params, background)
        # ... (后续保存逻辑不变)
        rendered_image_tensor = render_pkg["render"].clamp(0.0, 1.0)
        rendered_np = (rendered_image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(rendered_np)
        image_name = os.path.splitext(camera.image_name)[0]
        output_filename = os.path.join(output_path, f"{image_name}.png")
        pil_image.save(output_filename)

    print("\n" + "="*80)
    print(f"✅ 渲染完成！所有 {len(train_cameras)} 张图像已保存至 '{output_path}'。")
    print("="*80 + "\n")


# ==================== 脚本主入口 (不变) ====================
if __name__ == "__main__":
    HARDCODED_MODEL_PATH    = "/root/autodl-tmp/gaussian-splatting/kicker_v2_new_start"
    HARDCODED_OUTPUT_PATH   = "/root/autodl-tmp/gaussian-splatting/kicker_v2_new_start/render_output_final_check_v3"
    HARDCODED_SOURCE_PATH   = "/root/autodl-tmp/gaussian-splatting/data/kicker"
    HARDCODED_ITERATION     = "15000"
    
    # (我稍微改了下输出路径，避免覆盖之前的结果)

    # 为Camera类动态添加 to_device 方法
    def camera_to_device(self, device):
        for attr in ['world_view_transform', 'full_proj_transform', 'camera_center']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
    Camera.to_device = camera_to_device
    
    render_and_save_debug(
        model_path=HARDCODED_MODEL_PATH,
        output_path=HARDCODED_OUTPUT_PATH,
        source_path=HARDCODED_SOURCE_PATH,
        iteration=HARDCODED_ITERATION
    )