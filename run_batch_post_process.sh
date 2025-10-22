#!/bin/bash

# ===================================================================================
#      批量将稠密点云转换为无符号距离场 (UDF)
# ===================================================================================
#
# 目的: 自动为指定场景调用 post_process_mvs.py 脚本，
#       将 fused.ply 点云文件转换为 udf_grid.npy 距离场。
# 特性: 增加了可选的点云降采样功能，以大幅缩短处理时间。
#
# ===================================================================================

set -e
PROJECT_DIR=$(pwd)
DATA_ROOT_DIR="$PROJECT_DIR/data"
POST_PROCESS_SCRIPT="$PROJECT_DIR/post_process_mvs.py"

# --- [ 核心配置：处理与之前相同的场景 ] ---
SCENE_NAMES=(
    "delivery_area"
    "meadow"
)

# --- [ UDF 网格分辨率 ] ---
# 256 是一个很好的平衡点，在精度和内存占用之间取得了平衡。
GRID_RESOLUTION=256

# --- [ 核心优化：点云降采样配置 ] ---
# voxel_size (单位：米): 值越大，降采样越狠，点越少，处理越快。
#
# - 设为 0.0: 不进行降采样，与原始行为一致，质量最高但速度最慢。
# - 推荐值: 对于像 delivery_area 和 meadow 这样的大规模室外场景，
#           一个 1cm 到 2cm (0.01 到 0.02) 的体素大小是一个很好的起点。
#           它可以在保留绝大部分宏观几何结构的同时，去除大量冗余的点，
#           从而将处理时间从数小时缩短到一小时以内。
#
# 我们选择一个相对保守的值 0.015 (1.5厘米) 作为平衡点。
VOXEL_SIZE=0.015

echo "🚀🚀🚀 开始为指定场景 [${SCENE_NAMES[*]}] 生成无符号距离场... 🚀🚀🚀"
echo "-> 使用的降采样体素大小 (Voxel Size): ${VOXEL_SIZE} 米"
echo "-> 生成的UDF网格分辨率: ${GRID_RESOLUTION}"

for scene in "${SCENE_NAMES[@]}"; do
    scene_path="$DATA_ROOT_DIR/$scene"

    # 调用 Python 脚本处理每个场景，并传入降采样参数
    python "$POST_PROCESS_SCRIPT" \
        --dataset_path "$scene_path" \
        --resolution "$GRID_RESOLUTION" \
        --voxel_size "$VOXEL_SIZE"
done

echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 所有指定场景的距离场已生成完毕！ 🎉🎉🎉"
echo "# ======================================================================"