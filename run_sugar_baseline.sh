#!/bin/bash

# ==============================================================================
#           全新、简化的 SuGaR 基准测试运行脚本 (v1.0)
#
#  这个脚本只做一件事：为官方的 `train_full_pipeline.py` 脚本准备参数
#  并正确地调用它。我们不再修改任何 Python 源代码。
#
#  使用方法:
#  1. 将您的数据集放在 `data/` 目录下 (例如 `data/electro`)
#  2. 运行此脚本: ./run_sugar_baseline.sh
# ==============================================================================

# --- 检查环境 ---
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "sugar" ]]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! 错误: 请先激活 SuGaR 的 Conda 环境                  !!!"
    echo "!!! -> conda activate sugar                               !!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi

# --- 配置区 ---
# 项目根目录 (脚本所在的目录)
PROJECT_DIR=$(pwd)

# 数据集根目录
DATA_ROOT="$PROJECT_DIR/data"

# 输出结果的根目录
OUTPUT_ROOT="$PROJECT_DIR/output/sugar_baseline"

# 定义要测试的场景列表
# 注意：路径是相对于 DATA_ROOT 的
SCENE_NAMES=(
    "electro"
    "delivery_area"
    "pipes"
    "courtyard"
    "facade"
    "kicker"
    "meadow"
    "office"
    # "nerf_360/bicycle" # 如果您的数据结构是这样，也请取消注释
)

# --- 主循环 ---
echo "🚀🚀🚀 开始 SuGaR 基准测试 (共 ${#SCENE_NAMES[@]} 个场景) 🚀🚀🚀"

for scene_name in "${SCENE_NAMES[@]}"; do
    
    SCENE_PATH="$DATA_ROOT/$scene_name"
    OUTPUT_PATH="$OUTPUT_ROOT/$scene_name"

    # 检查场景数据是否存在
    if [ ! -d "$SCENE_PATH" ]; then
        echo "⚠️ 警告: 找不到场景数据: $SCENE_PATH，跳过..."
        continue
    fi

    # 如果输出目录已存在，则跳过
    if [ -d "$OUTPUT_PATH" ]; then
        echo "✅ 结果已存在于 $OUTPUT_PATH，跳过场景: $scene_name"
        continue
    fi
    
    echo ""
    echo "######################################################################"
    echo "### 正在处理场景: [$scene_name]"
    echo "######################################################################"
    
    # --- 动态确定图像目录和分辨率 ---
    # 默认使用 ETH3D 的设置
    IMAGE_DIR="images/dslr_images_undistorted"
    RESOLUTION=4
    
    # 检查是否是 NeRF-360 场景
    if [[ $scene_name == nerf_360* ]]; then
        echo "-> 检测到 NeRF-360 场景。"
        IMAGE_DIR="images_8" # 优先使用 images_8
        RESOLUTION=8
    else
        echo "-> 检测到 ETH3D 或类似场景。"
    fi

    # --- 构建并执行命令 ---
    # 这是调用 SuGaR 官方脚本的正确方式。
    # -s: 场景路径
    # -m: 模型输出路径 (官方脚本用 -m 表示 model_path)
    # -r: 正则化类型 (这是 SuGaR 的核心参数之一)
    # --images: 图像子目录
    # --resolution: 降采样率
    
    # 注意：官方的 train_full_pipeline.py 使用 -m 作为输出目录
    # 并且它内部会处理所有步骤，包括 vanilla 3DGS 训练
    
    COMMAND="python train_full_pipeline.py \
        -s $SCENE_PATH \
        -m $OUTPUT_PATH \
        -r dn_consistency \
        --images $IMAGE_DIR \
        --resolution $RESOLUTION \
        --eval True"

    echo "将要执行的命令:"
    echo "$COMMAND"
    echo "----------------------------------------------------------------------"
    
    # 执行命令
    $COMMAND

    # 检查退出码
    if [ $? -eq 0 ]; then
        echo "✅ 场景 [$scene_name] 处理成功。"
    else
        echo "❌ 场景 [$scene_name] 处理失败。"
    fi

done

echo ""
echo "🎉🎉🎉 所有场景处理完毕! 🎉🎉🎉"