#!/bin/bash
# set -x

# ===================================================================================
#      LLFF 数据集 V5 冠军配置基准测试 (v3 - 最终修复版)
#
#  - --images 参数现在固定指向原始高分辨率的 'images' 文件夹。
#  - --resolution 参数控制在内存中执行的下采样因子。
# ===================================================================================

trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() { echo "" && echo "###  Ctrl+C detected! Forcing kill...  ###" && kill -9 -$$; }

MY_PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$MY_PROJECT_DIR/data/LLFF/nerf_llff_data"
EXPERIMENTS_ROOT_DIR="$MY_PROJECT_DIR/output/LLFF_V5_CHAMPION_BENCHMARK_FINAL"

SCENES_TO_RUN=("fern" "flower" "fortress" "horns" "room" "trex" "leaves" "orchids")
RESOLUTION_FACTOR=4
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
        touch "${model_path}/_FAILED.log"
    fi
}

echo "🚀🚀🚀 Starting LLFF Dataset V5 Champion Benchmark 🚀🚀🚀"

CHAMPION_ARGS=( --lambda_dssim 0.2 --structural_loss_mode "ms_sobel" --lambda_struct_loss 0.05
    --synergy_mode "v5_ultimate" --feedback_p_weighting_beta 0.5 --alpha_l1_feedback 0.7
    --alpha_ssim_feedback 1.0 --feedback_nonlinear_gamma 1.2 )

for SCENE in "${SCENES_TO_RUN[@]}"; do
    echo; echo "===================================================================="
    echo "                   PROCESSING SCENE: ${SCENE} (r=${RESOLUTION_FACTOR})"
    echo "===================================================================="
    scene_path="$DATA_ROOT_DIR/$SCENE"
    
    # CRITICAL: '--images' points to original 'images' folder. '--resolution' controls downsampling.
    base_args=(-s "$scene_path" --images "images" --iterations "$ITERATIONS" --resolution "$RESOLUTION_FACTOR" --eval
               --test_iterations $TEST_ITERS --checkpoint_iterations $CHECKPOINT_ITERS)

    exp_name="v5_champion_${SCENE}";
    model_path="${EXPERIMENTS_ROOT_DIR}/${exp_name}";
    
    run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" \
        "${base_args[@]}" -m "$model_path" "${CHAMPION_ARGS[@]}"
done

echo; echo "### 🎉🎉🎉 LLFF Benchmark Suite Finished! ###";
echo "Check results in ${EXPERIMENTS_ROOT_DIR}";