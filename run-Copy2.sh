#!/bin/bash

# ===================================================================================
#               3DGS with Hough Line Loss - Automated Workflow (v2)
# ===================================================================================
# v2 更新: 修复了由 '~' 引起的路径扩展问题，使用 $HOME 替代。
#
# 该脚本将执行以下三个步骤：
# 1. 将 Blender/NeRF 格式的数据转换为 3DGS 使用的 COLMAP 格式。
# 2. 在转换后的图像上运行霍夫直线检测，生成 lines.json 文件。
# 3. 启动 3DGS 训练，并启用自定义的直线损失函数。
# ===================================================================================

# --- [ 配置区: 请根据您的实际路径和需求修改这里的变量 ] ---

# 设置项目的主目录 (即 gaussian-splatting 仓库的根目录)
# 使用 $HOME 变量来正确指向主目录，而不是'~'
PROJECT_DIR="$HOME/autodl-tmp/gaussian-splatting"

# 原始数据路径
INPUT_DATA_DIR="$HOME/autodl-tmp/blender_output_1/py_controlled_render/scene_A_data"

# 转换后数据的存放根目录
# 我们将在 $PROJECT_DIR 下创建一个名为 'data' 的文件夹来存放所有处理过的数据集
OUTPUT_BASE_DIR="$PROJECT_DIR/data"

# 转换后数据集的最终路径 (脚本会自动创建)
CONVERTED_DATA_DIR="$OUTPUT_BASE_DIR/scene_A_data"

# 脚本路径
CONVERSION_SCRIPT="$PROJECT_DIR/prepare_data_aligned.py"
HOUGH_SCRIPT="$PROJECT_DIR/detect_lines.py"

# 训练参数
TRAIN_ITERATIONS=30000       # 先用 7000 次迭代快速查看效果
LAMBDA_LINE=0.01            # 直线损失的权重 (关键调优参数)
MODEL_OUTPUT_DIR="output/scene_A_with_lines" # 训练模型的输出路径

# --- [ 脚本执行区 ] ---

# 步骤 0: 确保我们从项目根目录开始执行
# 这有助于保证所有相对路径的正确性
cd "$PROJECT_DIR"

# 使脚本在遇到任何错误时立即退出
set -e

# 步骤 1: 数据格式转换
echo "======================================================================"
echo "STEP 1: 正在将 NeRF 数据转换为 COLMAP 格式..."
echo "输入: $INPUT_DATA_DIR"
echo "输出: $CONVERTED_DATA_DIR"
echo "======================================================================"

# 虽然脚本设计用于对齐A/B场景，但我们可以通过将同一路径传给 ref_dir 和 other_dirs 来处理单个场景。
python "$CONVERSION_SCRIPT" \
    --ref_dir "$INPUT_DATA_DIR" \
    --other_dirs "$INPUT_DATA_DIR" \
    --output_base "$OUTPUT_BASE_DIR"

echo "✅ 数据转换完成。"
echo


# 步骤 2: 霍夫直线检测
echo "======================================================================"
echo "STEP 2: 正在转换后的图像上运行霍夫直线检测..."
echo "目标数据集: $CONVERTED_DATA_DIR"
echo "======================================================================"

# 在转换后的数据集上运行检测，并生成可视化结果以供检查
python "$HOUGH_SCRIPT" \
    --dataset_path "$CONVERTED_DATA_DIR" \
    --image_dir "images" \
    --visualize

echo "✅ 霍夫直线检测完成。结果已保存到 $CONVERTED_DATA_DIR/lines.json"
echo "💡 您可以检查 '$CONVERTED_DATA_DIR/lines_visualization' 文件夹中的图片来评估检测效果。"
echo


# 步骤 3: 启动模型训练
echo "======================================================================"
echo "STEP 3: 启动 3DGS 训练 (已启用直线损失)..."
echo "训练数据: $CONVERTED_DATA_DIR"
echo "直线损失权重 (lambda_line): $LAMBDA_LINE"
echo "迭代次数: $TRAIN_ITERATIONS"
echo "======================================================================"

python train.py \
    -s "$CONVERTED_DATA_DIR" \
    -m "$MODEL_OUTPUT_DIR" \
    --iterations "$TRAIN_ITERATIONS" \
    --lambda_line "$LAMBDA_LINE"

echo "🎉 全部流程执行完毕！训练完成。"