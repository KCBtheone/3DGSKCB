#!/bin/bash

# ===================================================================================
#               3DGS TUM RGB-D 深度图损失 (Depth Loss) 实验
# ===================================================================================

# --- [ 1. 核心训练参数 ] ---
TRAIN_ITERATIONS=30000
# 核心设置: 开启 Depth Loss，关闭 Line Loss
LAMBDA_LINE_EXP=0.0      # <-- 关闭 Line Loss
DEPTH_WEIGHT_INIT=0.05   # <-- 深度损失的初始权重
DEPTH_WEIGHT_FINAL=0.05  # <-- 深度损失的最终权重 (不随时间衰减)

# 定义保存点 (每 3000 步保存一次)
SAVE_CHECKPOINT_LIST=(3000 6000 9000 12000 15000 18000 21000 24000 27000 ${TRAIN_ITERATIONS})
ITERATION_ARGS="--checkpoint_iterations ${SAVE_CHECKPOINT_LIST[@]} --save_iterations ${SAVE_CHECKPOINT_LIST[@]}"

# 基础参数
BASE_ARGS="--iterations ${TRAIN_ITERATIONS} ${ITERATION_ARGS} \
           --depth_l1_weight_init ${DEPTH_WEIGHT_INIT} \
           --depth_l1_weight_final ${DEPTH_WEIGHT_FINAL} \
           --lambda_line ${LAMBDA_LINE_EXP}"

# --- [ 2. 数据集 (单场景) ] ---
DATASET_NAME="rgbd_dataset_freiburg1_desk"
DATA_PATH="./dataset/${DATASET_NAME}"

# --- [ 3. 运行训练 ] ---
TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUTPUT_DIR="./output/${DATASET_NAME}_DEPTH_L${DEPTH_WEIGHT_INIT}_${TIMESTAMP}"
    
echo "=========================================================================="
echo ">>>> 正在启动训练: ${DATASET_NAME} (TUM 深度图初始化 + 深度损失)"
echo ">>>> 深度损失权重: ${DEPTH_WEIGHT_INIT} (固定)"
echo ">>>> 输出将保存至: ${OUTPUT_DIR}"
echo "=========================================================================="

python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_DIR" \
    ${BASE_ARGS}

echo "🎉🎉🎉 深度损失实验启动完成。 🎉🎉🎉"
