#!/bin/bash
# set -x # 如果需要调试，可以取消此行的注释

# ===================================================================================
#      统一基准测试套件 (LLFF, Mip-NeRF 360, Tanks and Temples)
#
#  算法配置: V5 性能冠军配置 (bonsai 实验二)
#  工作流:   为每个数据集自动适配正确的数据加载和下采样策略。
# ===================================================================================

# --- [ 1. 全局配置与辅助函数 ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###  检测到 Ctrl+C！正在强制终止所有子进程...  ###" && kill -9 -$$
}

# --- 请根据您的环境修改以下路径 ---
MY_PROJECT_DIR="/root/autodl-tmp/gaussian-splatting" # 你的代码库路径
LLFF_DATA_DIR="$MY_PROJECT_DIR/data/LLFF/nerf_llff_data"
MIP360_DATA_DIR="$MY_PROJECT_DIR/data/nerf_360" # 假设Mip-NeRF 360数据在此
TNT_DATA_DIR="$MY_PROJECT_DIR/data/tnt"
# ---------------------------------

EXPERIMENTS_ROOT_DIR="$MY_PROJECT_DIR/output/UNIFIED_V5_CHAMPION_BENCHMARK"
ITERATIONS=30000
TEST_ITERS=$(seq 7000 1000 ${ITERATIONS})
CHECKPOINT_ITERS="${ITERATIONS}"

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
    fi
}

# --- V5 性能冠军配置 ---
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

echo "🚀🚀🚀 开始运行统一基准测试套件 (LLFF, Mip-NeRF 360, TnT) 🚀🚀🚀"


# # ====================================================================
# #                          第一部分: LLFF
# # ====================================================================
# echo; echo "===================================================================="
# echo "                       PART 1: LLFF BENCHMARK"
# echo "===================================================================="
# LLFF_SCENES=("fern" "flower" "fortress" "horns" "room" "trex" "leaves" "orchids")
# for SCENE in "${LLFF_SCENES[@]}"; do
#     scene_path="$LLFF_DATA_DIR/$SCENE"
#     base_args=(-s "$scene_path" --images "images" --resolution 4 --eval --iterations "$ITERATIONS"
#                --test_iterations $TEST_ITERS --checkpoint_iterations $CHECKPOINT_ITERS)
#     exp_name="llff_${SCENE}"; model_path="${EXPERIMENTS_ROOT_DIR}/${exp_name}";
#     run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" \
#         "${base_args[@]}" -m "$model_path" "${CHAMPION_ARGS[@]}"
# done


# # ====================================================================
# #                        第二部分: Mip-NeRF 360
# # ====================================================================
# echo; echo "===================================================================="
# echo "                     PART 2: MIP-NERF 360 BENCHMARK"
# echo "===================================================================="
# MIP360_OUTDOOR_SCENES=("bicycle" "flowers" "garden" "stump" "treehill")
# MIP360_INDOOR_SCENES=("bonsai" "counter" "kitchen" "room")

# # --- 运行室外场景 (4倍下采样) ---
# for SCENE in "${MIP360_OUTDOOR_SCENES[@]}"; do
#     scene_path="$MIP360_DATA_DIR/$SCENE"
#     base_args=(-s "$scene_path" --images "images" --resolution 4 --eval --iterations "$ITERATIONS"
#                --test_iterations $TEST_ITERS --checkpoint_iterations $CHECKPOINT_ITERS)
#     exp_name="mip360_outdoor_${SCENE}"; model_path="${EXPERIMENTS_ROOT_DIR}/${exp_name}";
#     run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" \
#         "${base_args[@]}" -m "$model_path" "${CHAMPION_ARGS[@]}"
# done

# # --- 运行室内场景 (2倍下采样) ---
# for SCENE in "${MIP360_INDOOR_SCENES[@]}"; do
#     scene_path="$MIP360_DATA_DIR/$SCENE"
#     base_args=(-s "$scene_path" --images "images" --resolution 2 --eval --iterations "$ITERATIONS"
#                --test_iterations $TEST_ITERS --checkpoint_iterations $CHECKPOINT_ITERS)
#     exp_name="mip360_indoor_${SCENE}"; model_path="${EXPERIMENTS_ROOT_DIR}/${exp_name}";
#     run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" \
#         "${base_args[@]}" -m "$model_path" "${CHAMPION_ARGS[@]}"
# done


# ====================================================================
#                     第三部分: Tanks and Temples
# ====================================================================
echo; echo "===================================================================="
echo "                 PART 3: TANKS AND TEMPLES BENCHMARK"
echo "===================================================================="
TNT_SCENES=( "family" "francis" "horse" "lighthouse" "m60" "panther" "playground" "train"
             "auditorium" "ballroom" "courtroom" "museum" "palace" "temple"
             "barn" "caterpillar" "church" "courthouse" "ignatius" "meetingroom" "truck" )
for SCENE in "${TNT_SCENES[@]}"; do
    scene_path="$TNT_DATA_DIR/$SCENE"
    # [!!] TnT 使用预下采样图，所以 resolution=1
    base_args=(-s "$scene_path" --images "images_2" --resolution 1 --eval --iterations "$ITERATIONS"
               --test_iterations $TEST_ITERS --checkpoint_iterations $CHECKPOINT_ITERS)
    exp_name="tnt_${SCENE}"; model_path="${EXPERIMENTS_ROOT_DIR}/${exp_name}";
    run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" \
        "${base_args[@]}" -m "$model_path" "${CHAMPION_ARGS[@]}"
done

echo; echo "### 🎉🎉🎉 统一基准测试套件全部执行完毕！ ###";
echo "请检查目录 ${EXPERIMENTS_ROOT_DIR} 以获取所有实验的结果。";