#!/bin/bash

# ===================================================================================
#   Mip-Splatting 标准训练对比脚本 (v2.1 - 迭代次数修改为 20000)
#   >>> 已启用 train.py 内的 [7000次后每1000次测试] 动态逻辑 <<<
# ===================================================================================

# --- [ 1. 全局配置区 ] ---
PROJECT_DIR=$(pwd)
# 假设 mip-splatting 仓库就在 gaussian-splatting 旁边的同级目录
DATA_ROOT_DIR="$PROJECT_DIR/data"
MIP_SPLAT_DIR="$PROJECT_DIR/../mip-splatting" # 假设您克隆到了上级目录

MIP_EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/MIP_SPLAT_COMPARISON_RUNS_2"
mkdir -p "$MIP_EXPERIMENTS_ROOT_DIR"

# --- 仅包含 ETH3D 和 NeRF-360 场景 ---
MIP_SCENE_NAMES=(
    # --- ETH3D 场景 ---
    "electro" "delivery_area" "pipes" "courtyard" "facade"
    "kicker" "meadow" "office"
    
    # --- NeRF-360 场景 ---
    "nerf_360/bicycle" "nerf_360/bonsai" "nerf_360/counter" "nerf_360/garden"
    "nerf_360/kitchen" "nerf_360/room" "nerf_360/stump"
)

# 默认训练迭代次数
DEFAULT_ITERATIONS=20000

# --- [ 2. 核心执行函数 - Mip-Splatting ] ---
run_mip_splatting_experiment() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local scene_name_safe=${scene_name//\//_} # 将 "nerf_360/bicycle" 转换为 "nerf_360_bicycle"
    local model_path="$MIP_EXPERIMENTS_ROOT_DIR/$scene_name_safe/mip_splatting_default"

    echo; echo "--- [Mip-Splatting] 正在处理场景 [${scene_name}] ---"
    
    if [ ! -d "$MIP_SPLAT_DIR" ]; then
        echo "     -> ❌ 错误: 找不到 mip-splatting 目录: ${MIP_SPLAT_DIR}"
        echo "     -> 提示: 请确认您已将 mip-splatting 克隆到正确的位置。"
        return 1
    fi

    if [ -d "${model_path}" ]; then
        echo "     -> ✅ 结果已存在，跳过。";
        return 0
    fi
    echo "     -> 输出至: ${model_path}"
    
    # --- 数据集适配逻辑 (保持与原脚本一致的分辨率) ---
    local image_subdir="images"
    local resolution_scale=4 # 默认值
    local extra_args=()
    
    if [[ $scene_name == nerf_360* ]]; then
        echo "     -> 检测到 NeRF-360 场景。"
        resolution_scale=8
        if [ -d "$scene_path/images_8" ]; then image_subdir="images_8"
        elif [ -d "$scene_path/images_4" ]; then image_subdir="images_4"
        elif [ -d "$scene_path/images_2" ]; then image_subdir="images_2"
        else image_subdir="images"
        fi
        extra_args=(--convert_pointcloud_beta) # 360 场景的默认 Mip-Splatting 参数
    elif [ -d "$scene_path/images/dslr_images_undistorted" ]; then
        echo "     -> 检测到 ETH3D (DSLR) 场景。"
        image_subdir="images/dslr_images_undistorted"
        resolution_scale=4
    else # 默认 ETH3D
        echo "     -> 使用默认 ETH3D 配置。"
        image_subdir="images"
        resolution_scale=4
    fi
    echo "     -> 使用图像路径: '${image_subdir}', 分辨率缩放: ${resolution_scale}"
    
    # --- 运行 Mip-Splatting 训练 ---
    # !!! 关键修改: 删除了 --test_iterations "${DEFAULT_ITERATIONS}" !!!
    # 这样 train.py 就会使用其内部的默认值 [7000, 30000]，并在此基础上动态添加 1000 步测试。
    timeout 6h python "$MIP_SPLAT_DIR/train.py" \
        -s "${scene_path}" \
        --images "${image_subdir}" \
        --resolution "${resolution_scale}" \
        --iterations "${DEFAULT_ITERATIONS}" \
        --model_path "${model_path}" \
        --save_iterations "${DEFAULT_ITERATIONS}" \
        --eval \
        "${extra_args[@]}"

    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "     -> ✅ 成功完成。";
    elif [ ${exit_code} -eq 124 ]; then
        echo "     -> ❌ 超时！实验运行超过6小时。";
        touch "${model_path}_TIMED_OUT.log";
    else
        echo "     -> ❌ 失败！Python 脚本以错误码 ${exit_code} 退出。";
        touch "${model_path}_FAILED.log";
    fi
}

# --- [ 3. 主执行循环 ] ---
echo "🚀🚀🚀 开始 Mip-Splatting 与您原有实验的对比运行 (共 ${#MIP_SCENE_NAMES[@]} 个场景) 🚀🚀🚀"
# 切换到 mip-splatting 所在的目录，确保 Python 能够找到其模块
cd "$MIP_SPLAT_DIR" || exit

for scene in "${MIP_SCENE_NAMES[@]}"; do
    run_mip_splatting_experiment "$scene"
done

echo; echo "# ======================================================================"
echo "# 🎉🎉🎉 Mip-Splatting 对比流程执行完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "所有结果已保存至 '$MIP_EXPERIMENTS_ROOT_DIR' 文件夹。"