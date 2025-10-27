#!/bin/bash
# set -x # 如果需要调试，可以取消此行的注释

# ===================================================================================
#      Tanks and Temples 数据集 V5 冠军配置基准测试
#
#  目标: 在Tanks and Temples数据集的多个场景上，运行已验证的V5冠军配置。
#  工作流: 使用预先下采样好的 'images_2' 文件夹，并设置 resolution=1 以避免二次缩放。
# ===================================================================================

# --- [ 1. 全局配置与辅助函数 ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###  检测到 Ctrl+C！正在强制终止所有子进程...  ###" && kill -9 -$$
}

# --- 请根据您的环境修改以下路径 ---
MY_PROJECT_DIR="/root/autodl-tmp/gaussian-splatting" # 你的代码库路径
DATA_ROOT_DIR="$MY_PROJECT_DIR/data/tnt"             # TnT 数据集的根目录
# ---------------------------------

EXPERIMENTS_ROOT_DIR="$MY_PROJECT_DIR/output/TNT_V5_CHAMPION_BENCHMARK"

# --- 待运行的 TnT 场景列表 ---
# 这是您下载日志中包含的所有场景
SCENES_TO_RUN=(
    # Intermediate
    "family" "francis" "horse" "lighthouse" "m60" "panther" "playground" "train"
    # Advanced
    "auditorium" "ballroom" "courtroom" "museum" "palace" "temple"
    # Additional
    "barn" "caterpillar" "church" "courthouse" "ignatius" "meetingroom" "truck"
)

# [!!] 关键配置: 数据集已经是2倍下采样，所以我们直接使用，不再进行内存缩放。
IMAGES_SUBDIR="images_2"
RESOLUTION_FACTOR=1 # 设置为1，加载器将使用图像的原始尺寸

ITERATIONS=30000
TEST_ITERS=$(seq 7000 1000 ${ITERATIONS})
CHECKPOINT_ITERS="${ITERATIONS}"

# --- 辅助函数 (无需修改) ---
run_single_experiment() {
    local project_dir=$1; local scene_name=$2; local exp_name=$3; local model_path=$4; shift 4; local python_args=("$@")
    echo; echo "--- [SCENE: ${scene_name} | EXPERIMENT: ${exp_name}] ---";
    if [ -d "${model_path}" ]; then echo "        -> Path exists. Deleting for a clean run..." && rm -rf "${model_path}"; fi
    echo "        -> Codebase: ${project_dir}";
    echo "        -> Output: ${model_path}";
    mkdir -p "${model_path}";
    stdbuf -oL -eL python "${project_dir}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    local exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -eq 0 ] && [ -f "${model_path}/best.ply" ]; then
        echo "        -> ✅ SUCCESS: ${exp_name}"
    else
        echo "        -> ❌ FAILED (Code ${exit_code} or best.ply not found): ${exp_name}"
        touch "${model_path}/_FAILED.log"
    fi
}

# --- [ 2. 实验调度 ] ---
echo "🚀🚀🚀 Starting Tanks and Temples Dataset V5 Champion Benchmark 🚀🚀🚀"

# --- V5 冠军配置参数 ---
CHAMPION_ARGS=(
    --lambda_dssim 0.2
    --structural_loss_mode "ms_sobel"
    --lambda_struct_loss 0.05
    --synergy_mode "v5_ultimate"
    --feedback_p_weighting_beta 0.5
    --alpha_l1_feedback 0.7
    --alpha_ssim_feedback 1.0
    --feedback_nonlinear_gamma 1.2
)

for SCENE in "${SCENES_TO_RUN[@]}"; do
    
    echo; echo "===================================================================="
    echo "                   PROCESSING SCENE: ${SCENE}"
    echo "===================================================================="

    scene_path="$DATA_ROOT_DIR/$SCENE"

    # --- 动态构建基础参数列表 ---
    base_args=(-s "$scene_path" --images "$IMAGES_SUBDIR" --iterations "$ITERATIONS" --resolution "$RESOLUTION_FACTOR" --eval)
    if [[ -n "$TEST_ITERS" ]]; then base_args+=(--test_iterations $TEST_ITERS); fi
    if [[ -n "$CHECKPOINT_ITERS" ]]; then base_args+=(--checkpoint_iterations $CHECKPOINT_ITERS); fi

    exp_name="v5_champion_${SCENE}";
    model_path="${EXPERIMENTS_ROOT_DIR}/${exp_name}";
    
    run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" \
        "${base_args[@]}" \
        -m "$model_path" \
        "${CHAMPION_ARGS[@]}"
done

echo; echo "### 🎉🎉🎉 Tanks and Temples Benchmark Suite Finished! ###";
echo "Check results in ${EXPERIMENTS_ROOT_DIR}";