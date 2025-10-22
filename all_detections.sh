#!/bin/bash

# ===================================================================================
#      批量生成 lines.json - 针对所有 ETH3D 场景
# ===================================================================================
#
# 目的: 自动化地为所有指定的场景文件夹运行 `detect_lines.py`，
#       生成用于后续训练的 `lines.json` 文件。
#
# 特性:
# - 全自动: 遍历所有场景。
# - 高效: 自动跳过已经生成过 `lines.json` 的场景。
#
# ===================================================================================

# --- [ 1. 配置区 ] ---
PROJECT_DIR=$(pwd)
# 数据集根目录
DATA_ROOT_DIR="$PROJECT_DIR/data"
# Python 检测脚本的路径
HOUGH_SCRIPT="$PROJECT_DIR/detect_lines.py"

# 在这里列出您下载的所有 ETH3D 场景的文件夹名称
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

# 霍夫变换参数 (与您之前的实验保持一致)
HOUGH_THRESHOLD=80
MIN_LINE_LENGTH=40
MAX_LINE_GAP=10

set -e # 遇到错误立即停止

# --- [ 2. 执行区 ] ---

echo "🚀 开始为所有场景批量生成 lines.json 文件..."
cd "$PROJECT_DIR"

# 遍历所有场景
for scene in "${SCENE_NAMES[@]}"; do
    scene_path="$DATA_ROOT_DIR/$scene"

    # 检查场景路径是否存在
    if [ ! -d "$scene_path" ]; then
        echo "⚠️ 警告: 未找到场景目录 '$scene_path'，跳过此场景。"
        continue
    fi
    
    # 【高效】检查 lines.json 是否已存在，如果存在则跳过
    if [ -f "$scene_path/lines.json" ]; then
        echo "✅ [${scene}] 的 lines.json 文件已存在，跳过。"
        continue
    fi

    echo
    echo "--- 正在处理场景: [${scene}] ---"

    python "$HOUGH_SCRIPT" \
        --dataset_path "$scene_path" \
        --image_dir "images" \
        --visualize \
        --hough_threshold $HOUGH_THRESHOLD \
        --min_length $MIN_LINE_LENGTH \
        --max_gap $MAX_LINE_GAP
done

echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 全部场景的 lines.json 生成完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "现在您可以运行 'run_final_benchmark.sh' 来启动大规模训练了。"