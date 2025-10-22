#!/bin/bash
# ==============================================================================
#           批量几何先验生成脚本 (法线 & 置信度)
# ==============================================================================
#
# 功能:
# 1. 遍历指定的场景列表。
# 2. 调用多核Python脚本，一次性为所有有效场景并行生成 'geometry_priors'。
# 3. 自动检查每个场景的数据完整性 (必需的.txt文件)。
#
#===============================================================================

# --- [ 1. 配置区 ] ---

# 项目根目录
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"

# 包含所有场景数据的根目录
DATA_ROOT_DIR="$PROJECT_DIR/data"

# [重要] 指向您新创建的多核Python脚本
# 确保这个文件名与你保存的文件名一致
PRIOR_GENERATOR_SCRIPT="$PROJECT_DIR/generate_priors_parallel.py"

# [可配置] 设置并行处理时使用的CPU核心数
NUM_WORKERS=8 # 您可以根据您的机器配置修改这个值

# --- [ 2. 需要处理的场景列表 ] ---
# (与您之前的脚本保持一致)
SCENE_NAMES=(
    "kicker"
    "courtyard"
    "delivery_area"
    "dtu"
    "electro"
    "facade"
    "meadow"
    "office"
    "pipes"
    "playground"
    "relief"
    "relief_2"
)

# --- [ 3. 主执行逻辑 ] ---
echo "🚀 开始为 ${#SCENE_NAMES[@]} 个场景批量生成几何先验..."
echo "============================================================"

# 检查Python生成器脚本是否存在
if [ ! -f "$PRIOR_GENERATOR_SCRIPT" ]; then
    echo "❌ 错误: 找不到先验生成脚本: $PRIOR_GENERATOR_SCRIPT"
    echo "请确保路径正确，并且你已经将多核Python脚本保存在该位置。"
    exit 1
fi

# 构建一个包含所有有效场景完整路径的列表
declare -a scene_paths_to_process

for scene in "${SCENE_NAMES[@]}"; do
    scene_path="$DATA_ROOT_DIR/$scene"
    if [ -d "$scene_path" ]; then
        # 预检查：确保COLMAP的TXT文件存在，避免Python脚本报错
        if [ -f "$scene_path/sparse/0/cameras.txt" ] && \
           [ -f "$scene_path/sparse/0/images.txt" ] && \
           [ -f "$scene_path/sparse/0/points3D.txt" ]; then
            echo "-> [✔️ ${scene}] 路径和文件有效，将被处理。"
            scene_paths_to_process+=("$scene_path")
        else
            echo "-> [⚠️ ${scene}] 警告: 场景缺少必要的.txt文件(cameras/images/points3D)，已跳过。"
        fi
    else
        echo "-> [⚠️ ${scene}] 警告: 找不到场景目录 [${scene_path}]，已跳过。"
    fi
done

# 检查是否有任何有效的场景可以处理
if [ ${#scene_paths_to_process[@]} -eq 0 ]; then
    echo "❌ 错误: 没有找到任何可以处理的有效场景。请检查 SCENE_NAMES 和 DATA_ROOT_DIR 配置。"
    exit 1
fi

echo "------------------------------------------------------------"
echo "将使用 ${NUM_WORKERS} 个CPU核心处理以下 ${#scene_paths_to_process[@]} 个场景:"
printf " - %s\n" "${scene_paths_to_process[@]}"
echo "------------------------------------------------------------"

# --- [ 4. 执行命令 ] ---
# 调用多核Python脚本，并把所有场景路径一次性作为参数传给它
python "$PRIOR_GENERATOR_SCRIPT" --num_workers ${NUM_WORKERS} "${scene_paths_to_process[@]}"

# 检查最终执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉🎉🎉 所有场景的几何先验数据准备完毕！ 🎉🎉🎉"
else
    echo ""
    echo "🔥🔥🔥 数据准备过程中发生错误，请检查上面的日志输出。 🔥🔥🔥"
fi