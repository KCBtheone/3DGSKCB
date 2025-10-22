#!/bin/bash

# ===================================================================================
#      Office 场景最终专项测试 (v2.0 - 验证与探索)
#
# 目标: 1. 验证 gamma=0.3 基线的效果。
#        2. 探索在更晚的启动时机下，使用 gamma=1.0 (原始信度) 的可能性。
#        3. 测试新的协同约束组合。
# ===================================================================================

# --- [ 1. 终止信号陷阱 (保持不变) ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###   检测到 Ctrl+C！正在强制终止所有子进程...   ###" && kill -9 -$$
}

# --- [ 2. 全局配置区 ] ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$PROJECT_DIR/data"
# !!! 继续使用同一输出目录，方便对比 !!!
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/office_final_test"

# --- [ 3. 固定实验配置 ] ---
TARGET_SCENE="office"
ITERATIONS=20000
RESOLUTION=8
NORMAL_START_ITER=7000

# 基础学习率配置
BASE_SCALING_LR=0.005
BASE_ROTATION_LR=0.001

# =================================================================================

# --- [ 4. 辅助函数 (保持不变) ] ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [${exp_name}] for scene [${scene_name}] ---"
    if [ -d "${model_path}" ]; then echo "       -> 结果已存在，跳过。"; return; fi
    echo "       -> 输出至: ${model_path}"; mkdir -p "${model_path}"
    
    python "${PROJECT_DIR}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    local exit_code=${PIPESTATUS[0]}
    
    if [ ${exit_code} -eq 0 ]; then echo "       -> ✅ 成功完成。"; 
    else echo "       -> ❌ 失败！(错误码 ${exit_code})。标记失败并继续下一个实验。"; touch "${model_path}/_FAILED.log"; fi
}

# --- [ 5. 主执行逻辑 ] ---
echo "🚀🚀🚀 开始运行 Office 场景最终专项测试 v2.0 🚀🚀🚀"
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

# --- [ 实验 1: 新增 - gamma=0.3 的无约束基线 ] ---
echo; echo "--- [ Stage 1: Running New Baseline (gamma=0.3, no constraints) ] ---"
exp_name="exp11_baseline_gamma0p3"
model_path="${EXPERIMENTS_ROOT_DIR}/${TARGET_SCENE}/${exp_name}"
run_single_experiment "$TARGET_SCENE" "$exp_name" "$model_path" \
    "${base_args[@]}" --model_path "$model_path" --confidence_gamma 0.3

# --- [ 实验 2: 探索 gamma=1.0 的新组合 ] ---
echo; echo "--- [ Stage 2: Exploring new combos for gamma=1.0 (late start) ] ---"
exp_counter=12
# [核心] 使用更晚的启动时机来避免CUDA错误
LATE_ISO_START_ITER=12000 
declare -a combos_g1=("0.1 0.1" "0.2 0.4")

for combo in "${combos_g1[@]}"; do
    read -r alpha lambda <<< "$combo"
    
    alpha_str="a$(echo $alpha | tr '.' 'p')"
    lambda_str="l$(echo $lambda | tr '.' 'p')"
    start_str="iso_start_$(($LATE_ISO_START_ITER / 1000))k"

    exp_name="exp$(printf "%02d" $exp_counter)_gamma1p0_${alpha_str}_${lambda_str}_${start_str}"
    model_path="${EXPERIMENTS_ROOT_DIR}/${TARGET_SCENE}/${exp_name}"
    
    # 使用 gamma=1.0
    current_args=("${base_args[@]}" --model_path "$model_path" --confidence_gamma 1.0)

    # 添加法线约束
    current_args+=(--geometry_constraint_type normal --alpha_normals "$alpha" --geometry_start_iter ${NORMAL_START_ITER})
    # 添加浮游物约束
    current_args+=(--lambda_isotropy "$lambda" --isotropy_start_iter "$LATE_ISO_START_ITER")

    run_single_experiment "$TARGET_SCENE" "$exp_name" "$model_path" "${current_args[@]}"
    exp_counter=$((exp_counter + 1))
done

# --- [ 实验 3: 探索 gamma=0.3 的新组合 ] ---
echo; echo "--- [ Stage 3: Exploring new combo for gamma=0.3 ] ---"
# [核心] 测试 "强法线 + 弱浮游物" 组合
alpha="0.4"
lambda="0.1"
iso_start_iter="10000" # 使用已验证的安全启动时机

alpha_str="a$(echo $alpha | tr '.' 'p')"
lambda_str="l$(echo $lambda | tr '.' 'p')"
start_str="iso_start_$(($iso_start_iter / 1000))k"

exp_name="exp$(printf "%02d" $exp_counter)_gamma0p3_${alpha_str}_${lambda_str}_${start_str}"
model_path="${EXPERIMENTS_ROOT_DIR}/${TARGET_SCENE}/${exp_name}"

# 使用 gamma=0.3
current_args=("${base_args[@]}" --model_path "$model_path" --confidence_gamma 0.3)
# 添加法线约束
current_args+=(--geometry_constraint_type normal --alpha_normals "$alpha" --geometry_start_iter ${NORMAL_START_ITER})
# 添加浮游物约束
current_args+=(--lambda_isotropy "$lambda" --isotropy_start_iter "$iso_start_iter")

run_single_experiment "$TARGET_SCENE" "$exp_name" "$model_path" "${current_args[@]}"

echo; echo "######################################################################"
echo "### 🎉🎉🎉 Office 场景 v2.0 测试运行完毕！ ###"
echo "######################################################################"