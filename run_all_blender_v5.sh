#!/bin/bash
set -e # 如果任何命令失败，脚本将立即退出

# ===================================================================================
#      V5 终极协同框架 - 全Blender数据集泛化能力测试脚本
#
#  本脚本将使用在 Bonsai 场景上验证过的“终极组合”冠军超参数，
#  在所有标准的 Blender 场景 (nerf_synthetic_colmap) 上进行训练和评估。
#
#  目标：验证这套参数在不同场景下的泛化性能。
# ===================================================================================

# --- [ 1. 全局配置 ] ---
# --- 请根据您的环境修改以下路径 ---
CODE_DIR="/root/autodl-tmp/gaussian-splatting" 
CONVERTED_DATA_DIR="/root/autodl-tmp/gaussian-splatting/data/nerf_synthetic_colmap"
# 创建一个新的、清晰的输出目录
OUTPUT_DIR="/root/autodl-tmp/gaussian-splatting/output/BLENDER_V5_ULTIMATE_COMBO_RUN"

# --- 实验核心配置 ---
RESOLUTION=2
ITERATIONS=30000
# 从 7000 次迭代开始，每 1000 次测试一次，以捕获最佳模型
TEST_ITERS=$(seq 7000 1000 ${ITERATIONS})

# --- V5 终极融合框架 核心基础参数 ---
CORE_V5FUSION_ARGS=(
    --lambda_dssim 0.2
    --structural_loss_mode "ms_sobel"
    --lambda_struct_loss 0.05
    --synergy_mode "v5_ultimate"
    --feedback_p_weighting_beta 0.5
)

# --- [核心] Bonsai 冠军参数选择 ---
# 默认使用“实验2：终极组合”的参数。
# 如果您想切换到“实验3：解耦测试”，请注释掉当前行，并取消下一组的注释。

# --- 实验2: 终极组合 (默认启用) ---
CHAMPION_ARGS=(
    --alpha_l1_feedback 0.7
    --alpha_ssim_feedback 1.0
    --feedback_nonlinear_gamma 1.2
)

# # --- 实验3: 解耦测试 (如需使用，请取消注释并注释掉上面的组合) ---
# CHAMPION_ARGS=(
#     --alpha_l1_feedback 0.5
#     --alpha_ssim_feedback 1.0
#     --feedback_nonlinear_gamma 1.0
# )


# --- [ 2. 辅助函数 ] ---
run_single_experiment() {
    local project_dir=$1; local scene_name=$2; local exp_name=$3; local model_path=$4; shift 4; local python_args=("$@")
    echo; echo "--- [开始处理场景: ${scene_name} | 实验: ${exp_name}] ---";
    if [ -f "${model_path}/best.ply" ]; then
        echo -e "\e[32m        -> 结果 best.ply 已存在，跳过。\e[0m"
        return
    elif [ -d "${model_path}" ]; then
        echo "        -> 目录已存在但 best.ply 未找到，将重新运行..."
    fi
    echo "        -> 输出至: ${model_path}";

    ( # 将命令放在子shell中，这样即使单个实验失败，整个脚本也不会因 set -e 而退出
        stdbuf -oL -eL python "${project_dir}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    )
    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ] && [ -f "${model_path}/best.ply" ]; then
        echo -e "\e[32m        -> ✅ 成功完成: ${exp_name}\e[0m"
    else
        echo -e "\e[31m        -> ❌ 失败！(错误码 ${exit_code} 或 best.ply 未生成): ${exp_name}\e[0m"
        touch "${model_path}/_FAILED.log"
    fi
}

# --- [ 3. 脚本主循环 ] ---
echo "🚀🚀🚀 开始在整个Blender数据集上运行V5冠军参数测试 🚀🚀🚀"
echo "输出根目录: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# 自动发现所有已转换的场景
SCENES_TO_RUN=($(ls "${CONVERTED_DATA_DIR}"))
echo "将要运行的场景: ${SCENES_TO_RUN[*]}"

# 构建所有实验共用的基础参数 (注意：Blender需要 --white_background)
base_args=(--iterations ${ITERATIONS} --resolution ${RESOLUTION} --eval --white_background)
if [[ -n "$TEST_ITERS" ]]; then base_args+=(--test_iterations $TEST_ITERS); fi

# 遍历所有场景并执行训练
for scene in "${SCENES_TO_RUN[@]}"; do
    scene_path="${CONVERTED_DATA_DIR}/${scene}"
    exp_name="v5_ultimate_combo_${scene}" # 实验名清晰反映了参数配置
    model_path="${OUTPUT_DIR}/${exp_name}"

    run_single_experiment \
        "${CODE_DIR}" \
        "${scene}" \
        "${exp_name}" \
        "${model_path}" \
        -s "${scene_path}" \
        -m "${model_path}" \
        "${base_args[@]}" \
        "${CORE_V5FUSION_ARGS[@]}" \
        "${CHAMPION_ARGS[@]}"
done

echo "-----------------------------------------------------"
echo "🎉🎉🎉 所有Blender场景训练完毕！"
echo "请检查目录 ${OUTPUT_DIR} 以获取所有实验的结果。"