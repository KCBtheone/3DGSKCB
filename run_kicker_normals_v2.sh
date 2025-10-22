#!/bin/bash

# ===================================================================================
#      3DGS Kicker 场景法线约束专项优化实验 (v2.1 - 已修正图像路径)
#
# 目标: 寻找一组法线约束参数，使其 PSNR 比 exp5 (23.99 dB) 高出 1.0 dB 以上。
# 策略: 1. 使用已验证的最佳学习率 (alternative_lr)。
#        2. 延迟法线约束的启动时机 (geometry_start_iter=7000)。
#        3. 探索一组新的 alpha_normals 值。
# ===================================================================================

# --- [ 1. 终止信号陷阱 (保持不变) ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM

cleanup_and_exit() {
    echo ""
    echo "############################################################"
    echo "###   检测到 Ctrl+C！正在强制终止所有子进程...   ###"
    echo "############################################################"
    kill -9 -$$
}

# --- [ 2. 全局配置区 ] ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$PROJECT_DIR/data"
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/selective_sweep_normals_v2" 

# --- [ 3. 实验参数 ] ---
DEFAULT_ITERATIONS=15000
DEFAULT_RESOLUTION=8
KICKER_SCENE="kicker"

# --- [ 4. KICKER 新实验矩阵配置区 ] ---
# 固定使用效果最好的 alternative_lr
BEST_SCALING_LR=0.005
BEST_ROTATION_LR=0.0005

# 探索新的 alpha 值
NEW_ALPHA_VALUES=("0.1" "0.3" "0.8")

# 延迟法线约束的启动迭代
GEOMETRY_START_ITERATION=7000

# --- [ 5. 保存和测试的迭代次数 ] ---
SAVE_AND_CHECKPOINT_ITERS="7000 ${DEFAULT_ITERATIONS}"
TEST_ITERS="7000 ${DEFAULT_ITERATIONS}"

# =================================================================================

# --- [ 6. 辅助函数 (从原脚本复制，保持不变) ] ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [${exp_name}] for scene [${scene_name}] ---"
    if [ -d "${model_path}" ]; then echo "       -> 结果已存在，跳过。"; return; fi
    echo "       -> 输出至: ${model_path}"; mkdir -p "${model_path}"
    
    python "${PROJECT_DIR}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    local exit_code=${PIPESTATUS[0]}
    
    if [ ${exit_code} -eq 0 ]; then 
        echo "       -> ✅ 成功完成。"; 
    else 
        echo "       -> ❌ 失败！(错误码 ${exit_code})。标记失败并继续下一个实验。"; 
        touch "${model_path}/_FAILED.log";
    fi
}

# ---------------------------------------------------------------------------------
# --- [ 7. 主执行循环 ] ---
# ---------------------------------------------------------------------------------
echo "🚀🚀🚀 开始运行 Kicker 法线约束专项优化实验 🚀🚀🚀"
cd "$PROJECT_DIR"

# 基础参数
scene_path="$DATA_ROOT_DIR/$KICKER_SCENE"
scene_output_dir="$EXPERIMENTS_ROOT_DIR/$KICKER_SCENE"

# =======================================================================
# --- [ 关键修正 ] ---
# 动态检查并设置正确的图像子目录，恢复原始脚本的逻辑
local image_subdir="images"
if [ -d "$scene_path/images/dslr_images_undistorted" ]; then
    echo "     -> 检测到 'dslr_images_undistorted' 目录，将使用该目录。"
    image_subdir="images/dslr_images_undistorted"
else
    echo "     -> 未检测到 'dslr_images_undistorted' 目录，将使用默认的 'images' 目录。"
fi
# =======================================================================

base_args=(-s "${scene_path}" --images "${image_subdir}" --resolution "${DEFAULT_RESOLUTION}" --iterations "${DEFAULT_ITERATIONS}" --eval --save_iterations ${SAVE_AND_CHECKPOINT_ITERS} --checkpoint_iterations ${SAVE_AND_CHECKPOINT_ITERS} --test_iterations ${TEST_ITERS})

# 添加固定的最佳学习率
base_args+=(--scaling_lr "${BEST_SCALING_LR}" --rotation_lr "${BEST_ROTATION_LR}")

# 开始运行新的实验矩阵
# 从 exp9 开始计数，避免与之前的实验混淆
exp_counter=9
for alpha in "${NEW_ALPHA_VALUES[@]}"; do
    alpha_str="alpha_$(echo $alpha | tr '.' 'p')"
    start_iter_str="start_$(($GEOMETRY_START_ITERATION / 1000))k"
    
    # 实验命名更具描述性
    exp_name_str="K${exp_counter}/11: ${alpha_str}, ${start_iter_str}"
    model_path_str="${scene_output_dir}/exp${exp_counter}_${alpha_str}_${start_iter_str}"
    
    # 复制基础参数并添加本轮实验的特定参数
    current_args=("${base_args[@]}")
    current_args+=(--model_path "${model_path_str}")
    current_args+=(--geometry_constraint_type normal --alpha_normals "${alpha}" --geometry_start_iter ${GEOMETRY_START_ITERATION})
    
    run_single_experiment "$KICKER_SCENE" "${exp_name_str}" "${model_path_str}" "${current_args[@]}"
    
    exp_counter=$((exp_counter + 1))
done

echo; echo "######################################################################"
echo "### 🎉🎉🎉 新的专项优化实验运行完毕！请使用汇总脚本查看结果。 ###"
echo "######################################################################"