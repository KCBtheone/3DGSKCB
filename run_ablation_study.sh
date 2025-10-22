#!/bin/bash

# ===================================================================================
#      3DGS 静态权重损失 - 系统性消融研究脚本 (v4)
# ===================================================================================
#
# 目的: 对比不同 alpha 超参数下的 Weighted Loss 效果，并与 Baseline 对比。
#
# 新特性:
# 1. 结构化输出: 所有结果保存在新的 `WL_EXPERIMENTS` 文件夹中。
# 2. 参数化循环: 自动为每个指定的 alpha 值运行实验。
# 3. 智能 Baseline: 自动为每个场景运行一次 Baseline (如果尚不存在)。
#
# ===================================================================================

# --- [ 1. 全局配置区 ] ---
PROJECT_DIR=$(pwd)

# 新的顶层输出目录
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/WL_EXPERIMENTS"

# 数据集路径 (请确保这些路径正确)
COURTYARD_PATH="$PROJECT_DIR/data/courtyard"
DELIVERY_AREA_PATH="$PROJECT_DIR/data/delivery_area"

# 训练参数
RESOLUTION_SCALE=4
TRAIN_ITERATIONS=30000
SAVE_CHECKPOINT_LIST=(9000 ${TRAIN_ITERATIONS})
ITERATION_ARGS="--checkpoint_iterations ${SAVE_CHECKPOINT_LIST[@]} --save_iterations ${SAVE_CHECKPOINT_LIST[@]}"

# 权重损失的固定超参数
SIGMA=5.0

# --- [ !! 核心实验参数区 !! ] ---
# 在这里列出您想测试的所有 ALPHA 值。脚本将为每个值运行一次实验。
# 建议从低到高探索。
ALPHA_VALUES=(0.05 0.25 0.5 0.8)

# 在这里列出您想运行实验的场景路径。
SCENE_PATHS=("$COURTYARD_PATH" "$DELIVERY_AREA_PATH")

set -e # 遇到错误立即停止
mkdir -p "$EXPERIMENTS_ROOT_DIR" # 创建顶层输出目录

# --- [ 2. 核心函数定义区 ] ---

# 函数: 运行 Baseline 实验
# 它会检查 Baseline 是否已存在，避免重复运行
run_baseline_if_needed() {
    local scene_path=$1
    local scene_name=$(basename "$scene_path")
    local model_output_dir="$EXPERIMENTS_ROOT_DIR/${scene_name}_baseline_r${RESOLUTION_SCALE}"
    
    if [ -d "$model_output_dir" ]; then
        echo "✅ [${scene_name}] 的 [Baseline] 结果已存在，跳过。"
        return
    fi
    
    echo
    echo "# ======================================================================"
    echo "# 任务: 运行 [${scene_name}] 的 [Baseline] 实验 (首次)"
    echo "#    - 输出至: $model_output_dir"
    echo "# ======================================================================"

    if [ -f "$scene_path/lines.json" ]; then
        echo "   - (注意: 临时隐藏 lines.json 以确保纯净的 Baseline 对比)"
        mv "$scene_path/lines.json" "$scene_path/lines.json.bak"
    fi

    python train.py \
        -s "$scene_path" \
        -m "$model_output_dir" \
        --iterations "$TRAIN_ITERATIONS" \
        --lambda_line 0.0 \
        -r $RESOLUTION_SCALE \
        ${ITERATION_ARGS}

    if [ -f "$scene_path/lines.json.bak" ]; then
        echo "   - (注意: 已恢复 lines.json)"
        mv "$scene_path/lines.json.bak" "$scene_path/lines.json"
    fi
    echo "✅ [${scene_name}] 的 [Baseline] 实验完成。"
}

# 函数: 运行 Weighted Loss 实验
run_weighted_loss() {
    local scene_path=$1
    local alpha_val=$2
    local scene_name=$(basename "$scene_path")
    local model_output_dir="$EXPERIMENTS_ROOT_DIR/${scene_name}_WL_A${alpha_val}_S${SIGMA}_r${RESOLUTION_SCALE}"

    if [ -d "$model_output_dir" ]; then
        echo "✅ [${scene_name}] 在 alpha=${alpha_val} 下的 [Weighted Loss] 结果已存在，跳过。"
        return
    fi

    # 确保线条文件存在 (注意: 需要您提前为所有场景生成好 lines.json)
    if [ ! -f "$scene_path/lines.json" ]; then
        echo "❌ 错误: 未找到 '$scene_path/lines.json'。请先运行霍夫直线检测。" >&2
        exit 1
    fi
    
    echo
    echo "# ======================================================================"
    echo "# 任务: 运行 [${scene_name}] 的 [Weighted Loss] 实验 (alpha = ${alpha_val})"
    echo "#    - 输出至: $model_output_dir"
    echo "# ======================================================================"

    # !!! 关键: 我们需要一种方式将 alpha 和 sigma 传递给 train.py !!!
    # 为了避免修改 train.py 的命令行参数，我们采用一个临时文件的方法。
    # 我们将把 alpha 和 sigma 写入 cameras.py 可以读取的一个临时配置文件。
    #
    # 修改 cameras.py, 在 _create_loss_weight_map_from_df 函数的开头加入以下代码:
    #
    # import json
    # import os
    # config_path = 'wl_config.json'
    # alpha = 1.0 # 默认值
    # sigma = 5.0 # 默认值
    # if os.path.exists(config_path):
    #     with open(config_path, 'r') as f:
    #         config = json.load(f)
    #         alpha = config.get('alpha', alpha)
    #         sigma = config.get('sigma', sigma)
    
    echo "{\"alpha\": ${alpha_val}, \"sigma\": ${SIGMA}}" > wl_config.json
    echo "   - (注意: 已创建临时配置文件 wl_config.json)"

    python train.py \
        -s "$scene_path" \
        -m "$model_output_dir" \
        --iterations "$TRAIN_ITERATIONS" \
        --lambda_line 0.0 \
        -r $RESOLUTION_SCALE \
        ${ITERATION_ARGS}
    
    # 清理临时文件
    rm wl_config.json
    echo "   - (注意: 已删除临时配置文件)"
    
    echo "✅ [${scene_name}] 在 alpha=${alpha_val} 下的 [Weighted Loss] 实验完成。"
}


# --- [ 3. 实验执行区 ] ---

echo
echo "🚀 开始执行系统性消融研究..."
cd "$PROJECT_DIR"

for scene in "${SCENE_PATHS[@]}"; do
    scene_name=$(basename "$scene")
    echo
    echo "--- 开始处理场景: [${scene_name}] ---"
    
    # 1. 确保该场景的 Baseline 已经运行
    run_baseline_if_needed "$scene"
    
    # 2. 遍历所有指定的 alpha 值，运行 Weighted Loss 实验
    for alpha in "${ALPHA_VALUES[@]}"; do
        run_weighted_loss "$scene" "$alpha"
    done
    
    echo "--- 场景 [${scene_name}] 处理完毕 ---"
done

echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 全部指定的消融研究流程执行完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "所有结果已保存至 '$EXPERIMENTS_ROOT_DIR' 文件夹。"