#!/bin/bash

# ===================================================================================
#      3DGS 直线损失工作流 v3.1 (集成原生 COLMAP 处理 - Xvfb 修复版)
# ===================================================================================
#
# v3.1 更新日志:
#   - [核心修复] 为所有 `xvfb-run` 命令添加了 `-a` 参数，使其能够自动寻找
#     可用的显示服务器端口，解决了在多用户或Docker环境中常见的 Xvfb
#     启动失败问题。
#
# 工作流程:
#   1. [COLMAP 重建]: 直接对原始图像文件夹运行 COLMAP，生成稀疏模型。
#   2. [霍夫直线检测]: 在重建后的图像上运行检测，生成 lines.json。
#   3. [3DGS 训练]: 使用 COLMAP 输出和直线数据启动训练。
# ===================================================================================

# --- [ 1. 配置区: 请根据您的实际路径和需求修改 ] ---

# 设置项目的主目录 (即 gaussian-splatting 仓库的根目录)
PROJECT_DIR=$(realpath "$(dirname "$0")")

# 包含原始图像的文件夹路径
RAW_IMAGE_DIR="$HOME/autodl-tmp/time5"

# COLMAP 和 3DGS 训练数据的存放根目录
DATA_BASE_DIR="$PROJECT_DIR/data"

# 经过 COLMAP 处理后的场景数据最终路径
COLMAP_SCENE_PATH="$DATA_BASE_DIR/scene_from_time5"

# 脚本路径
HOUGH_SCRIPT="$PROJECT_DIR/detect_lines.py"

# 训练参数
TRAIN_ITERATIONS=30000
LAMBDA_LINE=0.01
MODEL_OUTPUT_DIR="output/time5_with_lines_$(date +%Y%m%d_%H%M)"

# --- [ 2. 环境与前置检查 ] ---

cd "$PROJECT_DIR"
echo "🚀 当前工作目录: $(pwd)"
set -e

if ! command -v colmap &> /dev/null; then
    echo "❌ 错误: 'colmap' 命令未找到。请确保 COLMAP 已正确安装并已添加到系统 PATH 中。" >&2
    exit 1
fi
if ! command -v xvfb-run &> /dev/null; then
    echo "⚠️ 警告: 'xvfb-run' 未找到，尝试自动安装..."
    sudo apt-get update && sudo apt-get install -y xvfb
    if ! command -v xvfb-run &> /dev/null; then
        echo "❌ 错误: 自动安装 'xvfb' 失败。请手动安装 (sudo apt-get install xvfb)。" >&2
        exit 1
    fi
fi
echo "✅ 环境检查通过。"

# --- [ 3. 脚本执行区 ] ---

# 步骤 1: COLMAP 三维重建
echo "======================================================================"
echo "STEP 1: 正在从原始图像进行 COLMAP 三维重建..."
echo "    - 输入图像: $RAW_IMAGE_DIR"
echo "    - 输出路径: $COLMAP_SCENE_PATH"
echo "======================================================================"

DB_PATH="$COLMAP_SCENE_PATH/database.db"
SPARSE_DIR="$COLMAP_SCENE_PATH/sparse"

rm -rf "$COLMAP_SCENE_PATH"
mkdir -p "$COLMAP_SCENE_PATH"
mkdir -p "$SPARSE_DIR"

ln -sfn "$RAW_IMAGE_DIR" "$COLMAP_SCENE_PATH/images"
IMAGE_DIR_FOR_COLMAP="$COLMAP_SCENE_PATH/images"

# 1.1 特征提取 (强制使用 GPU)
echo "    -> (1/3) 正在提取特征 (使用 GPU)..."
# --- [ 修复点 ] 添加 -a 参数 ---
xvfb-run -a colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR_FOR_COLMAP" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

# 1.2 特征匹配 (使用 CPU 保持稳定性)
echo "    -> (2/3) 正在匹配特征 (使用 CPU)..."
# --- [ 修复点 ] 添加 -a 参数 ---
xvfb-run -a colmap exhaustive_matcher \
    --database_path "$DB_PATH" \
    --SiftMatching.use_gpu 0

# 1.3 三维重建 (Mapper)
echo "    -> (3/3) 正在进行稀疏重建 (CPU)..."
# --- [ 修复点 ] 添加 -a 参数 ---
xvfb-run -a colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR_FOR_COLMAP" \
    --output_path "$SPARSE_DIR"

if [ ! -d "$SPARSE_DIR/0" ]; then
    echo "❌ 错误: COLMAP 未能成功生成稀疏模型 'sparse/0'。流程中止。" >&2
    exit 1
fi

echo "✅ COLMAP 重建完成。"
echo

# 步骤 2: 霍夫直线检测
echo "======================================================================"
echo "STEP 2: 正在重建后的图像上运行霍夫直线检测..."
echo "    - 目标数据集: $COLMAP_SCENE_PATH"
echo "======================================================================"

python "$HOUGH_SCRIPT" \
    --dataset_path "$COLMAP_SCENE_PATH" \
    --image_dir "images" \
    --visualize

echo "✅ 霍夫直线检测完成。结果已保存到 $COLMAP_SCENE_PATH/lines.json"
echo "💡 您可以检查 '$COLMAP_SCENE_PATH/lines_visualization' 文件夹中的图片来评估检测效果。"
echo

# 步骤 3: 启动 3DGS 训练
echo "======================================================================"
echo "STEP 3: 启动 3DGS 训练 (已启用直线损失)..."
echo "    - 训练数据: $COLMAP_SCENE_PATH"
echo "    - 直线损失权重 (lambda_line): $LAMBDA_LINE"
echo "    - 迭代次数: $TRAIN_ITERATIONS"
echo "    - 模型输出至: $MODEL_OUTPUT_DIR"
echo "======================================================================"

python train.py \
    -s "$COLMAP_SCENE_PATH" \
    -m "$MODEL_OUTPUT_DIR" \
    --iterations "$TRAIN_ITERATIONS" \
    --lambda_line "$LAMBDA_LINE"

echo -e "\n🎉🎉🎉 全部流程执行完毕！训练完成。 🎉🎉🎉"