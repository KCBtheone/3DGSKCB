#!/bin/bash

# --- [安全设置] ---
set -e

# --- [通用参数配置] ---
SCENE_PATH="data/meadow"
IMAGE_SUBDIR="images/dslr_images_undistorted"
RESOLUTION_SCALE=4
ITERATIONS=20000
SAVE_ITERATIONS=20000 
TEST_ITERATIONS=20000
GEO_START_ITER_DEFAULT=7000 # 混合约束也应延迟启动

# 约束类型：Line (支持静态和动态)
CONSTRAINT_TYPE="line"

# 准备通用参数数组
COMMON_ARGS=(
    -s "${SCENE_PATH}"
    --images "${IMAGE_SUBDIR}"
    --resolution "${RESOLUTION_SCALE}"
    --iterations "${ITERATIONS}"
    --save_iterations "${SAVE_ITERATIONS}"
    --test_iterations "${TEST_ITERATIONS}"
    --geometry_constraint_type "${CONSTRAINT_TYPE}"
    --geometry_start_iter "${GEO_START_ITER_DEFAULT}"
    --eval
)


# --- [实验 1/4: 静态重型 (75% Static)] ---
ALPHA=0.15
LAMBDA_DYN=0.05
echo ""
echo "================================================="
echo ">>> [1/4] Starting HYBRID_STATIC_HEAVY Training"
echo ">>> Static(alpha)=${ALPHA}, Dynamic(lambda)=${LAMBDA_DYN}"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_hybrid_A${ALPHA//./p}_D${LAMBDA_DYN//./p} \
    --line_static_alpha "${ALPHA}" \
    --line_dynamic_lambda "${LAMBDA_DYN}"


# --- [实验 2/4: 平衡分配 (50% Balance)] ---
ALPHA=0.10
LAMBDA_DYN=0.10
echo ""
echo "================================================="
echo ">>> [2/4] Starting HYBRID_BALANCED Training"
echo ">>> Static(alpha)=${ALPHA}, Dynamic(lambda)=${LAMBDA_DYN}"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_hybrid_A${ALPHA//./p}_D${LAMBDA_DYN//./p} \
    --line_static_alpha "${ALPHA}" \
    --line_dynamic_lambda "${LAMBDA_DYN}"


# --- [实验 3/4: 动态重型 (75% Dynamic)] ---
ALPHA=0.05
LAMBDA_DYN=0.15
echo ""
echo "================================================="
echo ">>> [3/4] Starting HYBRID_DYNAMIC_HEAVY Training"
echo ">>> Static(alpha)=${ALPHA}, Dynamic(lambda)=${LAMBDA_DYN}"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_hybrid_A${ALPHA//./p}_D${LAMBDA_DYN//./p} \
    --line_static_alpha "${ALPHA}" \
    --line_dynamic_lambda "${LAMBDA_DYN}"


# --- [实验 4/4: 纯静态 (100% Static - 0% Dynamic)] ---
# 用于控制对比，验证动态权重的价值
ALPHA=0.20
LAMBDA_DYN=0.00
echo ""
echo "================================================="
echo ">>> [4/4] Starting HYBRID_PURE_STATIC Training"
echo ">>> Static(alpha)=${ALPHA}, Dynamic(lambda)=${LAMBDA_DYN}"
echo "================================================="
echo ""
python train.py "${COMMON_ARGS[@]}" --model_path output/meadow_hybrid_A${ALPHA//./p}_D${LAMBDA_DYN//./p} \
    --line_static_alpha "${ALPHA}" \
    --line_dynamic_lambda "${LAMBDA_DYN}"


# --- [完成] ---
echo ""
echo "✅ Hybrid Allocation experiments completed successfully!"