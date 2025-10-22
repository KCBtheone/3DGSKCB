#!/bin/bash

# ===================================================================================
#      Relief 场景综合测试 (v1.1 - 已修正约束组合)
#
# 目标: 在 relief 场景上，系统性地评估不同信度模式与几何约束组合的效果。
# ===================================================================================

# --- [ 1. 终止信号陷阱 (保持不变) ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###   检测到 Ctrl+C！正在强制终止所有子进程...   ###" && kill -9 -$$
}

# --- [ 2. 全局配置区 ] ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$PROJECT_DIR/data"
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/relief_comprehensive_test"

# --- [ 3. 固定实验配置 ] ---
TARGET_SCENE="relief"
ITERATIONS=20000
RESOLUTION=8

# 基础学习率配置
BASE_SCALING_LR=0.005
BASE_ROTATION_LR=0.001

# 几何约束启动时机
NORMAL_START_ITER=7000
ISOTROPY_START_ITER=7000

# =================================================================================

# --- [ 4. 辅助函数 (保持不变) ] ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [${exp_name}] for scene [${scene_name}] ---"
    if [ -d "${model_path}" ]; then echo "       -> 结果已存在，跳过。"; return; fi
    echo "       -> 输出至: ${model_path}"; mkdir -p "${model_path}"
    
    export CUDA_LAUNCH_BLOCKING=1
    python "${PROJECT_DIR}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    local exit_code=${PIPESTATUS[0]}
    export CUDA_LAUNCH_BLOCKING=0
    
    if [ ${exit_code} -eq 0 ]; then echo "       -> ✅ 成功完成。"; 
    else echo "       -> ❌ 失败！(错误码 ${exit_code})。标记失败并继续下一个实验。"; touch "${model_path}/_FAILED.log"; fi
}

# --- [ 5. 主执行逻辑 ] ---
echo "🚀🚀🚀 开始运行 Relief 场景综合测试 (v1.1) 🚀🚀🚀"
cd "$PROJECT_DIR"

# 确定图像子目录路径
scene_path="$DATA_ROOT_DIR/$TARGET_SCENE"
image_subdir_rel_path="images"
if [ -d "$scene_path/images/dslr_images_undistorted" ]; then
    image_subdir_rel_path="images/dslr_images_undistorted"
fi
echo "     -> 将使用图像子目录: ${image_subdir_rel_path}"

# 基础参数数组
base_args=(
    -s "${scene_path}" --images "${image_subdir_rel_path}"
    --iterations "${ITERATIONS}" --resolution "${RESOLUTION}" --eval
    --scaling_lr "${BASE_SCALING_LR}" --rotation_lr "${BASE_ROTATION_LR}"
)

# 定义信度模式和约束组合
declare -A confidence_modes
confidence_modes["no_conf"]="--no-confidence"
confidence_modes["raw_conf"]="--confidence_gamma 1.0"
confidence_modes["gamma_0p3_conf"]="--confidence_gamma 0.3"

# --- [ 🚀 关键修正 🚀 ] ---
# 根据您的最新要求，精确定义五种约束组合
declare -A constraint_combos
constraint_combos["baseline"]="--alpha_normals 0.0 --lambda_isotropy 0.0"
constraint_combos["normal_weak"]="--alpha_normals 0.2 --lambda_isotropy 0.0"
constraint_combos["normal_strong_synergy"]="--alpha_normals 0.4 --lambda_isotropy 0.2"
constraint_combos["synergy_weak"]="--alpha_normals 0.2 --lambda_isotropy 0.2"
constraint_combos["synergy_strong"]="--alpha_normals 0.4 --lambda_isotropy 0.4"
# --- [ 修正结束 ] ---

exp_counter=1

# --- [ 实验矩阵主循环 ] ---
for conf_mode_name in "${!confidence_modes[@]}"; do
    for const_combo_name in "${!constraint_combos[@]}"; do
        
        exp_name="exp$(printf "%02d" $exp_counter)_${conf_mode_name}_${const_combo_name}"
        model_path="${EXPERIMENTS_ROOT_DIR}/${TARGET_SCENE}/${exp_name}"
        
        current_args=("${base_args[@]}" --model_path "$model_path")
        
        # --- 处理信度模式 ---
        CONF_DIR="$scene_path/geometry_priors"
        CONF_DIR_DISABLED="$scene_path/geometry_priors_DISABLED"
        
        if [ -d "$CONF_DIR_DISABLED" ]; then mv "$CONF_DIR_DISABLED" "$CONF_DIR"; fi

        if [ "$conf_mode_name" == "no_conf" ]; then
            if [ -d "$CONF_DIR" ]; then
                mv "$CONF_DIR" "$CONF_DIR_DISABLED"
                echo "   -> [${exp_name}] 禁用信度图..."
            fi
        else
            gamma_arg=(${confidence_modes[$conf_mode_name]})
            current_args+=("${gamma_arg[@]}")
        fi
        
        # --- 处理约束组合 ---
        constraint_args_str=${constraint_combos[$const_combo_name]}
        read -r alpha_arg alpha_val lambda_arg lambda_val <<< "$constraint_args_str"
        
        if (( $(echo "$alpha_val > 0.0" | bc -l) )); then
            current_args+=(--geometry_constraint_type normal "$alpha_arg" "$alpha_val" --geometry_start_iter ${NORMAL_START_ITER})
        fi
        
        if (( $(echo "$lambda_val > 0.0" | bc -l) )); then
            current_args+=("$lambda_arg" "$lambda_val" --isotropy_start_iter ${ISOTROPY_START_ITER})
        fi
        
        # --- 运行实验 ---
        run_single_experiment "$TARGET_SCENE" "$exp_name" "$model_path" "${current_args[@]}"

        exp_counter=$((exp_counter + 1))
    done
done

# --- 清理 ---
CONF_DIR="$scene_path/geometry_priors"
CONF_DIR_DISABLED="$scene_path/geometry_priors_DISABLED"
if [ -d "$CONF_DIR_DISABLED" ]; then
    mv "$CONF_DIR_DISABLED" "$CONF_DIR"
    echo "   -> 清理：已恢复信度图文件夹。"
fi

echo; echo "######################################################################"
echo "### 🎉🎉🎉 Relief 场景综合测试运行完毕！ ###"
echo "######################################################################"