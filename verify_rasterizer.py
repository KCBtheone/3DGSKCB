import torch
import math
import time

# --- 1. 导入和环境检查 ---
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print("✅ [成功] 自定义CUDA模块 'diff_gaussian_rasterization' 导入成功。")
except ImportError as e:
    print("❌ [失败] 无法导入 'diff_gaussian_rasterization'。")
    print(f"   错误信息: {e}")
    exit()

if not torch.cuda.is_available():
    print("❌ [失败] 未检测到可用的CUDA设备。")
    exit()

print(f"   - PyTorch 版本: {torch.__version__}")
print(f"   - PyTorch CUDA 版本: {torch.version.cuda}")
print("---")

# --- 2. 辅助函数 ---
def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2)); tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear; bottom = -top; right = tanHalfFovX * znear; left = -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left); P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left); P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(zfar + znear) / (zfar - znear); P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    P[3, 2] = -1.0
    return P

# --- 3. 核心测试函数 ---
def run_environment_verification():
    print("\n--- 开始执行最终的环境和梯度验证 ---")
    
    # a. 虚拟场景
    device = "cuda"; image_height, image_width = 256, 256
    view_matrix = torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,-3.5],[0.,0.,0.,1.]], device=device)
    fov_rad = math.pi / 4.0
    proj_matrix = get_projection_matrix(0.1, 100.0, fov_rad, fov_rad).to(device)
    full_proj_transform = (proj_matrix @ torch.inverse(view_matrix)).T
    view_matrix_for_rasterizer = view_matrix.T

    # b. 虚拟高斯球 (需要计算梯度)
    means3D = torch.tensor([[0.0, 0.0, 0.0]], device=device, requires_grad=True)
    scales = torch.tensor([[0.2, 0.2, 0.2]], device=device, requires_grad=True)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
    opacities = torch.tensor([[0.9]], device=device, requires_grad=True)
    shs = torch.tensor([[[0.8, 0.1, 0.1]]], device=device, requires_grad=True)

    print("✅ [成功] 虚拟场景和待测高斯球设置完毕。")

    try:
        # =================================================================================
        # >>> [ 🚀 核心修正 1 ] <<<
        # 根据最新的错误信息，添加必需的 `antialiasing` 参数。
        # =================================================================================
        raster_settings = GaussianRasterizationSettings(
            image_height=image_height, image_width=image_width,
            tanfovx=math.tan(fov_rad / 2), tanfovy=math.tan(fov_rad / 2),
            bg=torch.tensor([0.1, 0.1, 0.1], device=device, dtype=torch.float32),
            scale_modifier=1.0, viewmatrix=view_matrix_for_rasterizer,
            projmatrix=full_proj_transform, sh_degree=0,
            campos=torch.inverse(view_matrix)[:3, 3], prefiltered=False,
            debug=False,
            antialiasing=False # <-- 直接解决 "missing... argument: 'antialiasing'" 错误
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        print("✅ [成功] 光栅化器实例化成功。API参数已完美匹配！")
    except Exception as e:
        print(f"❌ [致命失败] 光栅化器实例化仍然失败: {e}")
        print("   如果此步骤仍然失败，说明环境存在比API不匹配更深层的问题。")
        return

    # d. 执行前向和后向传播
    try:
        # =================================================================================
        # >>> [ 🚀 核心修正 2 ] <<<
        # 捕获所有返回值，以兼容返回多个值的、较新版本的CUDA模块。
        # =================================================================================
        render_pkg = rasterizer(
            means3D=means3D, means2D=torch.zeros_like(means3D), shs=shs,
            opacities=opacities, scales=scales, rotations=rotations,
            cov3D_precomp=None
        )
        rendered_image = render_pkg[0]

        print(f"✅ [成功] 前向传播（渲染）完成。")
        
        loss = rendered_image.sum()
        loss.backward()
        print("✅ [成功] 后向传播（梯度计算）完成。")

    except Exception as e:
        print("❌ [致命失败] 在前向或后向传播中发生运行时错误！")
        print(f"   错误信息: {e}")
        print("   这强烈表明编译出的CUDA内核存在功能性问题。")
        return

    # e. 验证梯度结果
    print("\n--- 正在验证梯度... ---")
    grad_valid = True; grad = means3D.grad
    if grad is None:
        print("❌ [核心失败] 'means3D' 的梯度为 None！"); grad_valid = False
    else:
        print(f"✅ [成功] 'means3D' 的梯度已生成。"); print(f"   - 梯度值(示例): {grad.cpu().numpy()}")
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print("❌ [核心失败] 梯度中包含 NaN 或 Inf 值！"); grad_valid = False
        if torch.all(grad == 0):
            print("⚠️ [警告] 梯度全为零，异常。"); grad_valid = False

    # f. 输出最终诊断结论
    print("\n--- 最终诊断结论 ---")
    if grad_valid:
        print("✅ [通过] 恭喜！您的底层编译环境功能完好，梯度计算正常！")
        print("\n   下一步行动指令:")
        print("   ➡️ 立即停止怀疑编译问题。问题根源 **100% 在数据层面**。")
        print("   1. **【首要任务】检查场景尺度**: 您的 Mip-NeRF 360 和 ETH3D 场景是否被正确归一化到半径为1的球内？这是导致灾难性失败的最常见原因。请立刻编写并运行场景尺度检查脚本。")
        print("   2. **【次要任务】检查坐标系**: 如果尺度正常，请可视化加载后的相机位姿，确保它们正确地包围着点云，且朝向正确。")
    else:
        print("❌ [未通过] 底层梯度测试失败。")
        print("   问题根源就在您的 **编译环境 (第一阶段)**。")
        print("\n   强制性行动指令:")
        print("   ➡️ **执行“焦土策略”进行重建**。不要抱有侥幸心理，必须严格执行清理和重编译。")
        print("   1. **彻底清理**: `cd submodules/diff-gaussian-rasterization && rm -rf build *.so` (对两个子模块都执行)")
        print("   2. **强制同步**: `git submodule update --init --recursive --force`")
        print("   3. **全新编译**: `pip install -e ./submodules/diff-gaussian-rasterization`")
        print("   重复此过程，直到这个诊断脚本显示 [通过] 为止。")

if __name__ == "__main__":
    run_environment_verification()