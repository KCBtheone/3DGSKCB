#!/bin/bash

# ===================================================================================
#      3DGS 几何先验 - ETH3D 全场景最终基准测试脚本 (v7)
# ===================================================================================
#
# 目的: 在 ETH3D 数据集的所有场景上，自动化地运行三种核心实验：
#       1. 纯动态权重 (Pure Dynamic)
#       2. 混合权重 (Hybrid)
#       3. 基线 (Baseline)
#
# 特性:
# - 全自动: 依次处理所有指定场景。
# - 节省空间: 只保存第 30000 次迭代的最终模型 (.pth 和 .ply)。
# - 结构化输出: 所有结果保存在新的 `ETH3D_FINAL_RUNS` 目录中。
#
# ===================================================================================

# --- [ 1. 全局配置区 ] ---
PROJECT_DIR=$(pwd)
# 新的顶层输出目录
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/ETH3D_FINAL_RUNS"
# 数据集根目录
DATA_ROOT_DIR="$PROJECT_DIR/data"

# 在这里列出您下载的所有 ETH3D 场景的文件夹名称
SCENE_NAMES=(
    "meadow"
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "kicker"
    "office"
    "pipes"
    "playground"
    "relief"
    "relief_2"
)

# 训练参数
RESOLUTION_SCALE=4
TRAIN_ITERATIONS=30000
# 【重要】只保存第 30000 次的结果
SAVE_CHECKPOINT_LIST=(${TRAIN_ITERATIONS})
ITERATION_ARGS="--checkpoint_iterations ${SAVE_CHECKPOINT_LIST[@]} --save_iterations ${SAVE_CHECKPOINT_LIST[@]}"

# 权重损失的固定超参数
SIGMA=5.0
BLUR_RADIUS=2

set -e # 遇到错误立即停止
mkdir -p "$EXPERIMENTS_ROOT_DIR"

# --- [ 2. 核心函数 ] ---

# 函数: 运行一个指定的实验
run_experiment() {
    local scene_name=$1
    local exp_type=$2 # 'baseline', 'hybrid', 'pure_dynamic'
    local alpha_val=$3
    local lambda_dyn=$4
    
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    # 创建场景专属的子目录
    local scene_output_dir="$EXPERIMENTS_ROOT_DIR/$scene_name"
    mkdir -p "$scene_output_dir"
    
    local model_output_dir="$scene_output_dir/${scene_name}_${exp_type}"

    if [ -d "$model_output_dir" ]; then
        echo "✅ [${scene_name}] 的 [${exp_type}] 实验结果已存在，跳过。"
        return
    fi
    
    # Baseline 需要隐藏 lines.json
    if [ "$exp_type" == "baseline" ] && [ -f "$scene_path/lines.json" ]; then
        echo "   - (注意: 为 Baseline 实验临时隐藏 lines.json)"
        mv "$scene_path/lines.json" "$scene_path/lines.json.bak"
    fi

    echo
    echo "# ======================================================================"
    echo "# 任务: 运行 [${scene_name}] -> [${exp_type}]"
    echo "#    - alpha=${alpha_val}, lambda_dynamic=${lambda_dyn}"
    echo "#    - 输出至: $model_output_dir"
    echo "# ======================================================================"

    # 仅在需要时 (非Baseline) 创建配置文件
    if [ "$exp_type" != "baseline" ]; then
        echo "{\"alpha\": ${alpha_val}, \"sigma\": ${SIGMA}}" > wl_config.json
    fi

    python train.py \
        -s "$scene_path" \
        -m "$model_output_dir" \
        --iterations "$TRAIN_ITERATIONS" \
        --lambda_line 0.0 \
        -r $RESOLUTION_SCALE \
        --lambda_dynamic_weight "$lambda_dyn" \
        --dynamic_weight_blur_radius "$BLUR_RADIUS" \
        ${ITERATION_ARGS}
    
    # 清理
    if [ -f "wl_config.json" ]; then
        rm wl_config.json
    fi
    if [ -f "$scene_path/lines.json.bak" ]; then
        echo "   - (注意: 已恢复 lines.json)"
        mv "$scene_path/lines.json.bak" "$scene_path/lines.json"
    fi
    
    echo "✅ [${scene_name}] 的 [${exp_type}] 实验完成。"
}

# --- [ 3. 实验执行区 ] ---

echo
echo "🚀 开始执行 ETH3D 全场景最终基准测试..."
cd "$PROJECT_DIR"

# 遍历所有场景
for scene in "${SCENE_NAMES[@]}"; do
    scene_path="$DATA_ROOT_DIR/$scene"

    # 检查场景路径是否存在
    if [ ! -d "$scene_path" ]; then
        echo "⚠️ 警告: 未找到场景目录 '$scene_path'，跳过此场景。"
        continue
    fi
    
    # 检查线条文件是否存在 (对于需要它的实验)
    if [ ! -f "$scene_path/lines.json" ]; then
        echo "⚠️ 警告: 未找到 '$scene_path/lines.json'。动态和混合实验将无法运行。请先为所有场景生成线条文件。"
        # 只运行 Baseline
        run_experiment "$scene" "baseline" 0.0 0.0
        continue
    fi

    echo
    echo "--- 开始处理场景: [${scene}] ---"

    # 1. 运行纯动态实验
    run_experiment "$scene" "pure_dynamic" 0.0 0.15
    
    # 2. 运行混合权重实验
    run_experiment "$scene" "hybrid" 0.05 0.05

    # 3. 运行基线实验
    run_experiment "$scene" "baseline" 0.0 0.0
    
    echo "--- 场景 [${scene}] 处理完毕 ---"
done

echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 全部场景的基准测试流程执行完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "所有结果已保存至 '$EXPERIMENTS_ROOT_DIR' 文件夹。"