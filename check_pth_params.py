import torch
import os
import sys

# 确保项目路径在sys.path中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # 假设您的项目结构，需要引入 GaussianModel
    from scene.gaussian_model import GaussianModel 
except ImportError:
    print("❌ 错误: 无法导入 GaussianModel。请确保您的工作目录位于 3D Gaussian Splatting 项目的根目录下。")
    sys.exit(1)

# ======================= 配置部分 =======================
# 替换为您要检查的 .pth 文件路径
CHECKPOINT_PATH = "/root/autodl-tmp/gaussian-splatting/kicker_v2_new_start/chkpnt15000.pth"
# =======================================================


def check_checkpoint(path):
    print(f"==================================================")
    print(f"🚀 正在检查文件: {os.path.basename(path)}")
    print(f"路径: {path}")
    
    if not os.path.exists(path):
        print(f"❌ 错误: 检查点文件不存在。请检查路径是否正确。")
        return

    try:
        # 1. 加载检查点
        checkpoint = torch.load(path, map_location="cpu")
        
        # 2. 从检查点中提取模型参数（通常是元组或字典中的第一个元素）
        if isinstance(checkpoint, tuple):
            print(f"✅ 检查点格式: 包含迭代次数 ({checkpoint[1]})。提取模型参数...")
            model_params_data = checkpoint[0]
        else:
            print(f"✅ 检查点格式: 仅包含模型参数。")
            model_params_data = checkpoint

        # 检查模型参数数据是否是预期的元组格式
        if not isinstance(model_params_data, tuple) or len(model_params_data) < 7:
             print("❌ 错误: 模型参数数据结构异常。可能是自定义保存格式。")
             return

        # 3. 解析核心张量（从 model_params_data 元组中提取）
        # model_params_data 结构通常是 (active_sh_degree, _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity, ...)
        # 注意：这里我们只需要张量部分
        _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity = model_params_data[1:7]
        
        # 4. 打印统计信息
        num_points = _xyz.shape[0]
        print(f"\n====================== 参数统计 =======================")
        print(f"🌟 点数 (Gaussians): {num_points}")
        
        # 检查 XYZ (坐标)
        xyz_min, xyz_max = _xyz.min().item(), _xyz.max().item()
        xyz_mean = _xyz.mean().item()
        print(f"\n📍 _xyz (坐标):")
        print(f"   - Min/Max: {xyz_min:.4f} / {xyz_max:.4f}")
        print(f"   - Mean: {xyz_mean:.4f}")

        # 检查 _scaling (缩放的Logits)
        scaling_min, scaling_max = _scaling.min().item(), _scaling.max().item()
        scaling_mean = _scaling.mean().item()
        print(f"\n📏 _scaling (Logits):")
        print(f"   - Min/Max: {scaling_min:.4f} / {scaling_max:.4f}")
        print(f"   - Mean: {scaling_mean:.4f}")
        
        # 激活后的 Scaling (实际大小)
        actual_scale = torch.exp(_scaling)
        actual_scale_min, actual_scale_max = actual_scale.min().item(), actual_scale.max().item()
        print(f"   - 实际 Scale (exp(_s)): {actual_scale_min:.6f} / {actual_scale_max:.4f}")

        # 检查 _opacity (透明度的Logits)
        opacity_min, opacity_max = _opacity.min().item(), _opacity.max().item()
        print(f"\n👻 _opacity (Logits):")
        print(f"   - Min/Max: {opacity_min:.4f} / {opacity_max:.4f}")

        # 检查 _rotation (四元数)
        rotation_min, rotation_max = _rotation.min().item(), _rotation.max().item()
        print(f"\n🔄 _rotation (Logits):")
        print(f"   - Min/Max: {rotation_min:.4f} / {rotation_max:.4f}")
        
        print(f"==================================================")

        # 5. 关键警告判断
        if actual_scale_max > 10.0 or actual_scale_min < 1e-7:
            print("\n🚨 【⚠️ 严重警告：Scale 范围异常 ⚠️】")
            print("模型中存在**尺寸过大**或**尺寸过小**的离群点。")
            print("极大的 Scale 值 (Max > 10) 意味着高斯点巨大，是渲染模糊的直接原因。")
            print("这 99% 证实了之前的诊断：几何约束（`cameras_extent=0`）失效，或`scaling_lr`过大导致了数值爆炸（尽管您已经降低了学习率，但爆炸可能发生在训练早期）。")
        else:
            print("\n✅ 【Scale 范围正常】")
            print("如果 Scale 范围正常，但渲染仍然模糊，请检查离线渲染脚本中的**相机参数**（如FoV、分辨率或世界-视图变换矩阵）是否与训练时使用的**完全一致**。")
        
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_checkpoint(CHECKPOINT_PATH)