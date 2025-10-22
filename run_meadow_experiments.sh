#!/bin/bash

# --- [安全设置] ---
# 确保任何命令失败时，脚本都会立即停止执行
set -e

# --- [通用参数配置] ---
SCENE_PATH="data/meadow"
IMAGE_SUBDIR="images/dslr_images_undistorted"
RESOLUTION_SCALE=4
ITERATIONS=20000
SAVE_ITERATIONS=20000 
TEST_ITERATIONS=20000

# 准备通用参数数组
COMMON_ARGS=(
    -s "${SCENE_PATH}"
    --images "${IMAGE_SUBDIR}"
    --resolution "${RESOLUTION_SCALE}"
    --iterations "${ITERATIONS}"
    --save_iterations "${SAVE_ITERATIONS}"
    --test_iterations "${TEST_ITERATIONS}"
    --eval
)

# ==============================================================================
#  实验 1/10: Baseline
# ==============================================================================
echo ""
echo "================================================="
echo ">>> [1/10] Starting: Baseline (No Geometry)"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp1_base --geometry_constraint_type none

# ==============================================================================
#  系列实验: 不同强度的法线权重 (lambda_normals)
# ==============================================================================
GEO_START_ITER_DEFAULT=7000

# --- 实验 2/10: 弱引导 ---
LAMBDA=0.05
echo ""
echo "================================================="
echo ">>> [2/10] Starting: Normal Weighting (Weak, lambda=${LAMBDA})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp2_normal_l${LAMBDA//./p} \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA}" --geometry_start_iter "${GEO_START_ITER_DEFAULT}"

# --- 实验 3/10: 中等引导 (标准) ---
LAMBDA=0.10
echo ""
echo "================================================="
echo ">>> [3/10] Starting: Normal Weighting (Medium, lambda=${LAMBDA})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp3_normal_l${LAMBDA//./p} \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA}" --geometry_start_iter "${GEO_START_ITER_DEFAULT}"

# --- 实验 4/10: 强引导 ---
LAMBDA=0.20
echo ""
echo "================================================="
echo ">>> [4/10] Starting: Normal Weighting (Strong, lambda=${LAMBDA})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp4_normal_l${LAMBDA//./p} \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA}" --geometry_start_iter "${GEO_START_ITER_DEFAULT}"

# --- 实验 5/10: 激进引导 ---
LAMBDA=0.50
echo ""
echo "================================================="
echo ">>> [5/10] Starting: Normal Weighting (Aggressive, lambda=${LAMBDA})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp5_normal_l${LAMBDA//./p} \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA}" --geometry_start_iter "${GEO_START_ITER_DEFAULT}"

# ==============================================================================
#  系列实验: 不同的生效时间 (geometry_start_iter)
# ==============================================================================
LAMBDA_DEFAULT=0.10

# --- 实验 6/10: 延迟启动 ---
GEO_START=12000
echo ""
echo "================================================="
echo ">>> [6/10] Starting: Normal Weighting (Late Start, iter=${GEO_START})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp6_normal_late${GEO_START} \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA_DEFAULT}" --geometry_start_iter "${GEO_START}"

# --- 实验 7/10: 早期介入 ---
GEO_START=3000
echo ""
echo "================================================="
echo ">>> [7/10] Starting: Normal Weighting (Early Start, iter=${GEO_START})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp7_normal_early${GEO_START} \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA_DEFAULT}" --geometry_start_iter "${GEO_START}"

# --- 实验 8/10: 全程介入 ---
GEO_START=0
echo ""
echo "================================================="
echo ">>> [8/10] Starting: Normal Weighting (Full Time, iter=${GEO_START})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp8_normal_full${GEO_START} \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA_DEFAULT}" --geometry_start_iter "${GEO_START}"

# ==============================================================================
#  系列实验: 其他措施 (与 SSIM 结合)
# ==============================================================================

# --- 实验 9/10: 无 SSIM 干扰 ---
SSIM_LAMBDA=0.0
echo ""
echo "================================================="
echo ">>> [9/10] Starting: Normal Weighting (No SSIM, lambda_dssim=${SSIM_LAMBDA})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp9_normal_no_ssim \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA_DEFAULT}" --geometry_start_iter "${GEO_START_ITER_DEFAULT}" \
    --lambda_dssim "${SSIM_LAMBDA}"

# --- 实验 10/10: 高 SSIM 权重 ---
SSIM_LAMBDA=0.5
echo ""
echo "================================================="
echo ">>> [10/10] Starting: Normal Weighting (High SSIM, lambda_dssim=${SSIM_LAMBDA})"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_exp10_normal_high_ssim \
    --geometry_constraint_type normal --lambda_normals "${LAMBDA_DEFAULT}" --geometry_start_iter "${GEO_START_ITER_DEFAULT}" \
    --lambda_dssim "${SSIM_LAMBDA}"


# --- [完成] ---
echo ""
echo "✅ All 10 experiments completed successfully!"
echo "Check the 'output/' directory for results."