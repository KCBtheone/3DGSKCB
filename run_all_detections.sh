#!/bin/bash

# ===================================================================================
#      批量生成 lines.json - V3 (使用 LSD + 长度滤波)
# ===================================================================================
#
# 目的: 使用更鲁棒的 LSD 算法为所有场景重新生成 lines.json，
#       并通过长度后处理滤波，移除由纹理产生的过短直线。
#
# ===================================================================================

# --- [ 1. 配置区 ] ---
PROJECT_DIR=$(pwd)
DATA_ROOT_DIR="$PROJECT_DIR/data"
# 确保这个脚本能找到我们最新的 detect_lines.py (V5)
DETECT_SCRIPT="$PROJECT_DIR/detect_lines.py"

# 在这里列出您希望处理的所有 ETH3D 场景文件夹名称
SCENE_NAMES=(
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "kicker"
    "meadow"
    "office"
    "pipes"
    "playground"
    "relief"
    "relief_2"
)

# --- [ 核心参数区 ] ---
#
#   这是后处理滤波的最小长度阈值（单位：像素）。
#   所有检测出的、长度小于此值的直线都将被丢弃。
#   40 是一个比较合理、通用的起始值。
MIN_FILTER_LENGTH=60

set -e # 遇到错误立即停止

# --- [ 2. 执行区 ] ---

echo "🚀 开始使用 LSD + 长度滤波(min_len=${MIN_FILTER_LENGTH}) 为所有场景批量生成 lines.json 文件..."
cd "$PROJECT_DIR"

for scene in "${SCENE_NAMES[@]}"; do
    scene_path="$DATA_ROOT_DIR/$scene"

    if [ ! -d "$scene_path" ]; then
        echo "⚠️ 警告: 未找到场景目录 '$scene_path'，跳过此场景。"
        continue
    fi
    
    # 为了确保使用新参数重新生成，我们先删除旧文件
    if [ -f "$scene_path/lines.json" ]; then
        echo "ℹ️ [${scene}] 的旧 lines.json 已存在，将删除以使用新参数重新生成。"
        rm "$scene_path/lines.json"
    fi
    if [ -d "$scene_path/lines_visualization" ]; then
        rm -r "$scene_path/lines_visualization"
    fi

    echo
    echo "--- 正在处理场景: [${scene}] ---"

    # 调用脚本，并明确指定使用 lsd 检测器 和 新的长度滤波参数
    python "$DETECT_SCRIPT" \
        --dataset_path "$scene_path" \
        --detector "lsd" \
        --min_filter_length "$MIN_FILTER_LENGTH" \
        --visualize
done

echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 全部场景的 lines.json 使用 LSD+滤波 重新生成完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "请务必检查每个场景下的 'lines_visualization' 文件夹，确认滤波效果。"