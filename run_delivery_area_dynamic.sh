#!/bin/bash

# ===================================================================================
#      3DGS 动态/混合权重 - delivery_area 场景深度探索 (v6)
# ===================================================================================
#
# 目的: 在 delivery_area 场景上，系统性地探索不同参数组合下的
#       动态及混合权重损失的效果。
#
# 输出: 所有结果都将保存到 WL_EXPERIMENTS 文件夹，方便与静态实验统一对比。
#
# ===================================================================================

# --- [ 1. 全局配置区 ] ---
PROJECT_DIR=$(pwd)
# 【重要】确保这个输出目录与您之前的静态实验一致
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/WL_EXPERIMENTS"

# 数据集路径
DELIVERY_AREA_PATH="$PROJECT_DIR/data/delivery_area"

# 训练参数
RESOLUTION_SCALE=4
TRAIN_ITERATIONS=30000
SAVE_CHECKPOINT_LIST=(9000 ${TRAIN_ITERATIONS})
ITERATION_ARGS="--checkpoint_iterations ${SAVE_CHECKPOINT_LIST[@]} --save_iterations ${SAVE_CHECKPOINT_LIST[@]}"

# 权重损失的固定超参数
SIGMA=5.0
BLUR_RADIUS=2

set -e
mkdir -p "$EXPERIMENTS_ROOT_DIR"

# --- [ 2. 核心函数 ] ---

# 函数: 运行一个指定的实验组合
run_experiment() {
    local scene_path=$1
    local alpha_val=$2
    local lambda_dyn=$3
    
    local scene_name=$(basename "$scene_path")
    # 文件夹命名格式: scene_WL_A<alpha>_D<lambda_dyn>_...
    local model_output_dir="$EXPERIMENTS_ROOT_DIR/${scene_name}_WL_A${alpha_val}_D${lambda_dyn}_S${SIGMA}_r${RESOLUTION_SCALE}"

    if [ -d "$model_output_dir" ]; then
        echo "✅ [${scene_name}] A=${alpha_val}, D=${lambda_dyn} 的结果已存在，跳过。"
        return
    fi

    echo
    echo "# ======================================================================"
    echo "# 任务: 运行 [${scene_name}] | alpha=${alpha_val}, lambda_dynamic=${lambda_dyn}"
    echo "#    - 输出至: $model_output_dir"
    echo "# ======================================================================"

    # 通过临时文件传递静态权重参数
    echo "{\"alpha\": ${alpha_val}, \"sigma\": ${SIGMA}}" > wl_config.json
    echo "   - (注意: 已创建临时配置文件 wl_config.json)"

    python train.py \
        -s "$scene_path" \
        -m "$model_output_dir" \
        --iterations "$TRAIN_ITERATIONS" \
        --lambda_line 0.0 \
        -r $RESOLUTION_SCALE \
        --lambda_dynamic_weight "$lambda_dyn" \
        --dynamic_weight_blur_radius "$BLUR_RADIUS" \
        ${ITERATION_ARGS}
    
    # 清理临时文件
    rm wl_config.json
    echo "   - (注意: 已删除临时配置文件)"
    
    echo "✅ [${scene_name}] A=${alpha_val}, D=${lambda_dyn} 实验完成。"
}

# --- [ 3. 实验执行区 ] ---

echo
echo "🚀 开始执行 delivery_area 场景的动态权重深度探索..."
cd "$PROJECT_DIR"

# --- [ 实验组定义 ] ---
# 我们将在这里调用 run_experiment 函数来执行上面推荐的五组实验

# 1. 纯动态 - 弱
run_experiment "$DELIVERY_AREA_PATH" 0.0 0.05

# 2. 纯动态 - 强
run_experiment "$DELIVERY_AREA_PATH" 0.0 0.15

# 3. 最佳静态 + 弱动态 (协同团队)
run_experiment "$DELIVERY_AREA_PATH" 0.05 0.05

# 4. 弱静态 + 中等动态 (动态主导)
run_experiment "$DELIVERY_AREA_PATH" 0.02 0.1

# 5. 中等静态 + 中等动态 (均衡推动)
run_experiment "$DELIVERY_AREA_PATH" 0.1 0.1


echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 全部指定的动态实验流程执行完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "所有结果已保存至 '$EXPERIMENTS_ROOT_DIR' 文件夹。"
echo "您可以更新 analyze_ablation_study.py 脚本来分析这些新结果。"