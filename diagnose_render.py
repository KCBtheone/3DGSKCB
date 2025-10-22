# diagnose_render.py (v3 - Corrected attribute access)
import os
import sys
import torch
import re
from argparse import ArgumentParser
from PIL import Image

# --- 确保项目路径在sys.path中 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.general_utils import safe_state
except ImportError as e:
    print(f"❌ 导入模块失败，请确保此脚本位于gaussian-splatting项目根目录: {e}")
    sys.exit(1)

def load_config(cfg_path: str) -> dict:
    """健壮地加载 cfg_args 文件，兼容多种格式。"""
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

@torch.no_grad()
def diagnose_and_render(model_path: str, output_path: str, source_path: str):
    """
    加载模型，在渲染前深度诊断其内部状态，然后尝试渲染第一帧。
    """
    print(f"🚀 开始处理模型: {model_path}")

    # --- 1. 加载模型配置 ---
    print("  -> 步骤1: 加载模型配置...")
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"  -> ❌ 错误: 在'{model_path}'中找不到 'cfg_args' 配置文件。"); return
    
    parser = ArgumentParser(description="渲染脚本参数加载器")
    model_params_def, pipe_params_def, opt_params_def = ModelParams(parser), PipelineParams(parser), OptimizationParams(parser)
    args_defaults = parser.parse_args([])

    try:
        saved_cfg_dict = load_config(cfg_path)
        for k, v in saved_cfg_dict.items():
            if hasattr(args_defaults, k): setattr(args_defaults, k, v)
    except Exception as e:
        print(f"  -> ❌ 错误: 解析配置文件 '{cfg_path}' 失败: {e}"); return
    
    args_defaults.source_path = source_path
    args_defaults.model_path = model_path
    if not hasattr(args_defaults, 'sh_degree'): args_defaults.sh_degree = 3
        
    model_params, pipe_params, opt_params = model_params_def.extract(args_defaults), pipe_params_def.extract(args_defaults), opt_params_def.extract(args_defaults)
    
    if not hasattr(pipe_params, 'debug'):
        safe_state(False); pipe_params.debug = False

    # --- 2. 加载高斯模型 ---
    print("  -> 步骤2: 从 .pth 检查点加载完整的模型状态...")
    chkpt_file = "chkpnt30000.pth"
    chkpt_path = os.path.join(model_path, chkpt_file)
    if not os.path.exists(chkpt_path):
        print(f"  -> ❌ 错误: 检查点文件不存在: {chkpt_path}"); return
    
    gaussians = GaussianModel(model_params.sh_degree)
    print(f"    -> [INFO] 正在从 '{chkpt_file}' 加载...")
    checkpoint = torch.load(chkpt_path, map_location="cpu")

    model_params_from_ckpt = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
    gaussians.restore(model_params_from_ckpt, opt_params)
    print(f"    -> [成功] 模型加载成功，包含 {gaussians.get_xyz.shape[0]} 个点。")

    # --- 3. 深度模型诊断 ---
    print("\n" + "="*30 + " 模型状态诊断报告 " + "="*30)
    
    # ============================ 本次核心修复 ============================
    # 定义已知的张量属性名称列表
    tensor_attr_names = [
        '_xyz', '_features_dc', '_features_rest',
        '_scaling', '_rotation', '_opacity'
    ]
    
    # 正确地将模型的所有张量属性移动到GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  诊断设备: {device}")
    for attr_name in tensor_attr_names:
        if hasattr(gaussians, attr_name):
            tensor = getattr(gaussians, attr_name)
            if isinstance(tensor, torch.Tensor):
                setattr(gaussians, attr_name, tensor.to(device))
    # ====================================================================

    try:
        # 检查 NaN (非数值)
        for attr_name in tensor_attr_names:
             if hasattr(gaussians, attr_name):
                tensor = getattr(gaussians, attr_name)
                if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                    print(f"  [🔴 致命问题] 属性 '{attr_name}' 中包含 NaN 值！")
        
        # 透明度分析
        opacities_raw = gaussians.get_opacity
        opacities_activated = torch.sigmoid(opacities_raw)
        print("\n--- 透明度 (Opacity) ---")
        print(f"  激活后 (0到1范围):")
        print(f"    均值: {opacities_activated.mean().item():.6f} | 标准差: {opacities_activated.std().item():.6f}")
        print(f"    最小值: {opacities_activated.min().item():.6f} | 最大值: {opacities_activated.max().item():.6f}")
        print(f"  (原始 logits):")
        print(f"    均值: {opacities_raw.mean().item():.4f} | 最小值: {opacities_raw.min().item():.4f} | 最大值: {opacities_raw.max().item():.4f}")

        # 缩放分析
        scales_raw = gaussians.get_scaling
        scales_activated = gaussians.scaling_activation(scales_raw)
        print("\n--- 缩放 (Scale) ---")
        print(f"  激活后 (真实世界尺寸):")
        print(f"    均值 (x,y,z): {scales_activated.mean(dim=0).cpu().numpy()}")
        print(f"    标准差 (x,y,z): {scales_activated.std(dim=0).cpu().numpy()}")
        print(f"    最小值: {scales_activated.min().item():.6f} | 最大值: {scales_activated.max().item():.6f}")
        print(f"  (原始 logits):")
        print(f"    均值: {scales_raw.mean().item():.4f} | 最小值: {scales_raw.min().item():.4f} | 最大值: {scales_raw.max().item():.4f}")

        # 位置分析
        xyz = gaussians.get_xyz
        print("\n--- 位置 (XYZ) ---")
        print(f"  场景中心点 (均值): {xyz.mean(dim=0).cpu().numpy()}")
        print(f"  场景范围 (最小值): {xyz.min(dim=0).values.cpu().numpy()}")
        print(f"  场景范围 (最大值): {xyz.max(dim=0).values.cpu().numpy()}")
        
    except Exception as e:
        print(f"  [🔴 致命问题] 在分析模型状态时发生错误: {e}")

    print("="*82 + "\n")


    # --- 4. 加载场景和相机 ---
    print("  -> 步骤4: 加载场景和训练相机...")
    scene = Scene(model_params, gaussians, shuffle=False)
    train_cameras = scene.getTrainCameras()
    if not train_cameras:
        print(f"  -> ❌ 错误: 未能加载任何训练相机。"); return
    print(f"    -> [成功] 成功加载 {len(train_cameras)} 个训练相机。")
    
    # --- 5. 渲染第一帧 ---
    print("\n  -> 步骤5: 尝试渲染第一帧图像...")
    os.makedirs(output_path, exist_ok=True)
    camera_to_render = train_cameras[0]
    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    try:
        render_pkg = render(camera_to_render, gaussians, pipe_params, background)
        rendered_image_tensor = render_pkg["render"].clamp(0.0, 1.0)
        
        rendered_np = (rendered_image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(rendered_np)

        output_filename = os.path.join(output_path, "diagnostic_render.png")
        pil_image.save(output_filename)
        print(f"✅ 诊断性渲染完成！图像已保存至 '{output_filename}'。")

    except Exception as e:
        import traceback
        print(f"  [🔴 致命问题] 在执行渲染时发生错误:")
        traceback.print_exc()


if __name__ == "__main__":
    parser = ArgumentParser(description="诊断并渲染3DGS模型的第一帧。")
    parser.add_argument("model_path")
    parser.add_argument("output_path")
    parser.add_argument("-s", "--source_path", required=True)
    args = parser.parse_args()
    diagnose_and_render(args.model_path, args.output_path, args.source_path)