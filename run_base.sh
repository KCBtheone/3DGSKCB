#!/bin/bash
set -e # 如果任何命令失败，立即停止脚本

# ===================================================================================
#      Mip-NeRF 360 场景 Baseline 批量测试脚本 (v7 - 终极稳定版)
# ===================================================================================
#  核心修复 (v7):
#  这是一套旨在彻底解决360度场景训练崩溃的“组合拳”。
#
#  1. `--random_background`: (保留) 这是稳定背景的基石，必须启用。
#
#  2. `--opacity_reset_interval 15000`: [最关键的改变] 大幅推迟透明度重置。
#     给模型整整15000次迭代的“安全期”，只专注于构建稳固的几何结构，
#     避免任何剧烈的“休克疗法”过早发生。
#
#  3. `--densify_grad_threshold 0.0004`: [提高致密化门槛]
#     让模型不要那么轻易地创建新点，强制它优先优化好现有的点。
#
#  4. `--scaling_lr 0.0025 --rotation_lr 0.0005`: [给优化器“镇静剂”]
#     大幅降低缩放和旋转的学习率，这是防止点被“流放”到无穷远、
#     导致“假加速”和几何崩溃的核心手段。
# ===================================================================================

# --- 1. 配置区 ---
OFFICIAL_GS_DIR="."
DATA_ROOT_DIR="./data"
OUTPUT_ROOT_DIR="./MIPNERF360_FINAL_STABLE_RUN" # 使用全新的输出目录

SCENE_NAMES=(
    "nerf_360/bicycle"
    "nerf_360/bonsai"
    "nerf_360/counter"
    "nerf_360/garden"
    "nerf_360/kitchen"
    "nerf_360/room"
    "nerf_360/stump"
)

ITERATIONS=30000
SAVE_ITERS="7000 ${ITERATIONS}"
TEST_SEQUENCE="7000 "
for i in $(seq 8000 1000 ${ITERATIONS}); do
    TEST_SEQUENCE+="${i} "
done
TEST_ITERS="${TEST_SEQUENCE}"

# --- 2. 启动前检查 ---
echo "================================================================"
echo ">>> Mip-NeRF 360 场景终极稳定版 Baseline 测试启动"
echo "================================================================"
PROJECT_ROOT=$(readlink -f "$(pwd)")
OFFICIAL_GS_DIR_ABS="${PROJECT_ROOT}/${OFFICIAL_GS_DIR}"
DATA_ROOT_DIR_ABS="${PROJECT_ROOT}/${DATA_ROOT_DIR}"
OUTPUT_ROOT_DIR_ABS="${PROJECT_ROOT}/${OUTPUT_ROOT_DIR}"
if [ ! -f "${OFFICIAL_GS_DIR_ABS}/train.py" ]; then echo "❌ 错误: 找不到 train.py: ${OFFICIAL_GS_DIR_ABS}"; exit 1; fi
echo "✅ 官方代码库路径: ${OFFICIAL_GS_DIR_ABS}"
if [ ! -d "$DATA_ROOT_DIR_ABS" ]; then echo "❌ 错误: 找不到数据根目录: ${DATA_ROOT_DIR_ABS}"; exit 1; fi
echo "✅ 数据根目录: ${DATA_ROOT_DIR_ABS}"
mkdir -p "$OUTPUT_ROOT_DIR_ABS"
echo "✅ 结果将保存至: ${OUTPUT_ROOT_DIR_ABS}"
echo "----------------------------------------------------------------"

# --- 3. 循环执行所有场景的测试 ---
for scene in "${SCENE_NAMES[@]}"; do
    scene_name_safe=${scene//\//_}
    echo
    echo "--- 开始处理场景: ${scene} ---"

    SCENE_DATA_PATH_ABS="${DATA_ROOT_DIR_ABS}/${scene}"
    SCENE_OUTPUT_PATH_ABS="${OUTPUT_ROOT_DIR_ABS}/${scene_name_safe}"

    if [ -d "$SCENE_OUTPUT_PATH_ABS" ]; then
        echo "      -> 结果目录已存在，跳过此场景。"
        continue
    fi
    echo "      -> 数据源: ${SCENE_DATA_PATH_ABS}"
    echo "      -> 输出至: ${SCENE_OUTPUT_PATH_ABS}"

    image_subdir=""
    resolution_param=""
    if [ -d "$SCENE_DATA_PATH_ABS/images_4" ]; then
        image_subdir="images_4"; resolution_param=1;
    elif [ -d "$SCENE_DATA_PATH_ABS/images_8" ]; then
        image_subdir="images_8"; resolution_param=1;
    else
        image_subdir="images"; resolution_param=4;
    fi
    echo "      -> 自动配置: --images '${image_subdir}', -r '${resolution_param}'"

    # ===================================================================================
    # >>> [ 🚀 核心执行命令 ] <<<
    # ===================================================================================
    python "${OFFICIAL_GS_DIR_ABS}/train.py" \
        -s "$SCENE_DATA_PATH_ABS" \
        -m "$SCENE_OUTPUT_PATH_ABS" \
        --images "${image_subdir}" \
        -r "${resolution_param}" \
        --iterations "$ITERATIONS" \
        --save_iterations ${SAVE_ITERS} \
        --test_iterations ${TEST_ITERS} \
        --eval \
        --random_background \
        --opacity_reset_interval 15000 \
        --densify_grad_threshold 0.0004 \
        --scaling_lr 0.0025 \
        --rotation_lr 0.0005

    if [ $? -eq 0 ]; then
        echo "      -> ✅ 场景 [${scene}] 训练成功完成。"
    else
        echo "      -> ❌ 场景 [${scene}] 训练失败！"
    fi
done

# --- 4. 结束 ---
echo
echo "================================================================"
echo ">>> 所有场景的测试已尝试运行完毕！"
echo ">>> 请检查 '${OUTPUT_ROOT_DIR_ABS}' 文件夹获取所有结果。"
echo "================================================================"