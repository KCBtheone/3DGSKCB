#!/bin/bash

# ===================================================================================
#      3DGS 法线损失 - 全场景基准测试脚本 (v8)
# ===================================================================================
#
# 目的: 在所有指定的 COLMAP 场景上，自动化地运行一套包含7个核心对比
#       实验的完整流程，以全面评估法线损失的效果。
#
# 特性:
# - 全自动: 遍历所有场景，并为每个场景运行7个实验。
# - 结构化输出: 结果保存在 `NORMAL_EXPERIMENTS/` 目录下，按场景分组。
# - 智能路径: 自动检测并适应不同场景的 `images` 子目录结构。
#
# ===================================================================================

# --- [ 1. 全局配置区 ] ---
set -e # 任何命令失败时立即停止
PROJECT_DIR=$(pwd)
DATA_ROOT_DIR="$PROJECT_DIR/data"
# 新的顶层输出目录
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/NORMAL_EXPERIMENTS"

# 在这里列出您希望运行的所有场景文件夹名称
SCENE_NAMES=(
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "kicker"
    "meadow"
    "office"
    "pipes"
)

# 通用训练参数
ITERATIONS=30000
RESOLUTION_SCALE=4
SAVE_ITERATIONS=30000 
TEST_ITERATIONS=30000

# --- [ 2. 核心执行函数 ] ---

run_scene_experiments() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local scene_output_dir="$EXPERIMENTS_ROOT_DIR/$scene_name"
    
    mkdir -p "$scene_output_dir"
    
    echo
    echo "######################################################################"
    echo "###   开始处理场景: [${scene_name}]"
    echo "######################################################################"

    # --- 智能检测图像子目录 ---
    local image_subdir="images" # 默认值
    if [ -d "$scene_path/images/dslr_images_undistorted" ]; then
        image_subdir="images/dslr_images_undistorted"
        echo "   -> 检测到特殊图像路径: ${image_subdir}"
    fi

    # 准备通用参数数组
    local common_args=(
        -s "${scene_path}"
        --images "${image_subdir}"
        --resolution "${RESOLUTION_SCALE}"
        --iterations "${ITERATIONS}"
        --save_iterations "${SAVE_ITERATIONS}"
        --test_iterations "${TEST_ITERATIONS}"
        --eval
    )
    
    # --- 实验 1: Baseline ---
    echo "--- [1/7] 正在运行: Baseline ---"
    python train.py "${common_args[@]}" --model_path "${scene_output_dir}/${scene_name}_exp1_base" --geometry_constraint_type none

    # --- 法线损失系列实验的默认参数 ---
    local lambda_default=0.10
    local geo_start_default=7000

    # --- 实验 2: 法线损失 (弱) ---
    local lambda=0.05
    echo "--- [2/7] 正在运行: 法线损失 (弱, λ=${lambda}) ---"
    python train.py "${common_args[@]}" --model_path "${scene_output_dir}/${scene_name}_exp2_normal_l${lambda//./p}" \
        --geometry_constraint_type normal --lambda_normals "${lambda}" --geometry_start_iter "${geo_start_default}"

    # --- 实验 3: 法线损失 (中) ---
    local lambda=0.10
    echo "--- [3/7] 正在运行: 法线损失 (中, λ=${lambda}) ---"
    python train.py "${common_args[@]}" --model_path "${scene_output_dir}/${scene_name}_exp3_normal_l${lambda//./p}" \
        --geometry_constraint_type normal --lambda_normals "${lambda}" --geometry_start_iter "${geo_start_default}"

    # --- 实验 4: 法线损失 (强) ---
    local lambda=0.20
    echo "--- [4/7] 正在运行: 法线损失 (强, λ=${lambda}) ---"
    python train.py "${common_args[@]}" --model_path "${scene_output_dir}/${scene_name}_exp4_normal_l${lambda//./p}" \
        --geometry_constraint_type normal --lambda_normals "${lambda}" --geometry_start_iter "${geo_start_default}"

    # --- 实验 5: 法线损失 (晚启动) ---
    local geo_start=12000
    echo "--- [5/7] 正在运行: 法线损失 (晚启动, iter=${geo_start}) ---"
    python train.py "${common_args[@]}" --model_path "${scene_output_dir}/${scene_name}_exp5_normal_late${geo_start}" \
        --geometry_constraint_type normal --lambda_normals "${lambda_default}" --geometry_start_iter "${geo_start}"

    # --- 实验 6: 法线损失 (早启动) ---
    local geo_start=3000
    echo "--- [6/7] 正在运行: 法线损失 (早启动, iter=${geo_start}) ---"
    python train.py "${common_args[@]}" --model_path "${scene_output_dir}/${scene_name}_exp6_normal_early${geo_start}" \
        --geometry_constraint_type normal --lambda_normals "${lambda_default}" --geometry_start_iter "${geo_start}"
        
    # --- 实验 7: 法线损失 (无 SSIM) ---
    local ssim_lambda=0.0
    echo "--- [7/7] 正在运行: 法线损失 (无 SSIM) ---"
    python train.py "${common_args[@]}" --model_path "${scene_output_dir}/${scene_name}_exp7_normal_no_ssim" \
        --geometry_constraint_type normal --lambda_normals "${lambda_default}" --geometry_start_iter "${geo_start_default}" \
        --lambda_dssim "${ssim_lambda}"

    echo "✅ 场景 [${scene_name}] 的所有7个实验已完成。"
}

# --- [ 3. 主执行循环 ] ---

echo "🚀🚀🚀 开始全场景法线损失基准测试 (共 ${#SCENE_NAMES[@]} 个场景) 🚀🚀🚀"
cd "$PROJECT_DIR"

for scene in "${SCENE_NAMES[@]}"; do
    run_scene_experiments "$scene"
done

echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 所有场景的基准测试流程执行完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "所有结果已保存至 '$EXPERIMENTS_ROOT_DIR' 文件夹。"