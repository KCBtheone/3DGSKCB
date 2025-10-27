#!/bin/bash
# set -x # 如果需要调试，可以取消此行的注释

# ===================================================================================
#      V6 核心模组对决实验 v1.2 (最终扩展版)
#
# 实验设计:
# - 基线: 强大的 v5_error_dynamics (Ours-Base)。
# - 消融: 在 Ours-Base 上，独立测试 DINO, SVAS, DEID, OAI 四个模组的贡献。
# - 协同: 测试关键模组的组合效果，冲击SOTA。
# - 探索: 测试所有模组的极限性能，并与一个先进的替代方案 (Physical Alpha) 对比。
# ===================================================================================

# --- [ 1. 全局配置与辅助函数 ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###  检测到 Ctrl+C！正在强制终止所有子进程...  ###" && kill -9 -$$
}

# --- 请根据您的环境修改以下路径 ---
MY_PROJECT_DIR="/root/autodl-tmp/gaussian-splatting" # 你的代码库路径
# ---------------------------------

DATA_ROOT_DIR="$MY_PROJECT_DIR/data/nerf_360"
# [!!] 使用一个全新的根目录来存放这次关键实验的结果
EXPERIMENTS_ROOT_DIR="$MY_PROJECT_DIR/output/V6_MODULE_SHOWDOWN_FINAL"

SCENE="bonsai"
RESOLUTION=8
ITERATIONS=30000

# --- 实验核心配置 ---
TEST_ITERS=$(seq 7000 1000 ${ITERATIONS})
SAVE_ITERS=""
CHECKPOINT_ITERS="${ITERATIONS}"

# --- 辅助函数：运行单个实验 (保持不变) ---
run_single_experiment() {
    local project_dir=$1; local scene_name=$2; local exp_name=$3; local model_path=$4; shift 4; local python_args=("$@")
    echo; echo "--- [场景: ${scene_name} | 实验: ${exp_name}] ---";
    if [ -f "${model_path}/best.ply" ]; then
        echo "        -> 结果 best.ply 已存在，跳过。"
        return
    elif [ -d "${model_path}" ]; then
        echo "        -> 目录已存在但 best.ply 未找到，将重新运行..."
        rm -rf "${model_path}"
    fi
    echo "        -> 使用代码库: ${project_dir}";
    echo "        -> 输出至: ${model_path}";
    mkdir -p "${model_path}";

    stdbuf -oL -eL python "${project_dir}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"

    local exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -eq 0 ] && [ -f "${model_path}/best.ply" ]; then
        echo "        -> ✅ 成功完成: ${exp_name}"
    else
        echo "        -> ❌ 失败！(错误码 ${exit_code} 或 best.ply 未生成): ${exp_name}"
        touch "${model_path}/_FAILED.log"
    fi
}

# --- [ 2. 实验调度 ] ---
echo "🚀🚀🚀 开始运行 V6 核心模组对决实验 (11组) 🚀🚀🚀"

IMAGES_SUBDIR="images_${RESOLUTION}"
scene_path="$DATA_ROOT_DIR/$SCENE"
scene_output_root="$EXPERIMENTS_ROOT_DIR/$SCENE"
mkdir -p "$scene_output_root"

# --- 动态构建基础参数列表 ---
base_args=(-s "$scene_path" --images "$IMAGES_SUBDIR" --iterations "$ITERATIONS" --resolution "$RESOLUTION" --eval)
if [[ -n "$TEST_ITERS" ]]; then base_args+=(--test_iterations $TEST_ITERS); fi
if [[ -n "$SAVE_ITERS" ]]; then base_args+=(--save_iterations $SAVE_ITERS); fi
if [[ -n "$CHECKPOINT_ITERS" ]]; then base_args+=(--checkpoint_iterations $CHECKPOINT_ITERS); fi

echo; echo "===================================================================="
echo "                   开始处理场景: ${SCENE} (r=${RESOLUTION})"
echo "===================================================================="

# ===================================================================================
#                                 11 组对决实验
# ===================================================================================

# --- 实验 01: [你的起点] 我们强大的自适应基线 ---
exp_name="exp01_ours_base"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "ms_grad" \
    --synergy_mode "v5_error_dynamics" \
    --lambda_struct_loss 0.0 `# 禁用固定权重，因为是动态的` \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.015


# --- 实验 02: [消融A] 测试 DINO 诊断模块 ---
exp_name="exp02_ablation_dino"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "dino_feat" `# <-- 核心改动` \
    --synergy_mode "v5_error_dynamics" \
    --lambda_struct_loss 0.0 \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.005 `# 注意: DINO的阈值可能需要根据实际误差值进行调整`


# --- 实验 03: [消融B] 测试 SVAS 策略模块 ---
exp_name="exp03_ablation_svas"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "ms_grad" \
    --synergy_mode_spatial `# <-- 核心改动` \
    `# SVAS 也需要这些 base lambda 来定义宏观/微观损失` \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.015


# --- 实验 04: [消融C] 测试 DEID 执行模块 ---
exp_name="exp04_ablation_deid"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "ms_grad" \
    --synergy_mode "v5_error_dynamics" \
    --intelligent_densification `# <-- 核心改动` \
    --lambda_struct_loss 0.0 \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.015


# --- 实验 05: [消融D] 测试 OAI 引导模块 ---
exp_name="exp05_ablation_oai"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "ms_grad" \
    --synergy_mode "v5_error_dynamics" \
    --optimizer_intervention `# <-- 核心改动` \
    --lambda_struct_loss 0.0 \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.015


# --- 实验 06: [关键协同] DINO + SVAS ---
exp_name="exp06_synergy_dino_svas"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "dino_feat" `# <-- 组合1` \
    --synergy_mode_spatial `# <-- 组合2` \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.005


# --- 实验 07: [最终模型] DINO + SVAS + DEID ---
exp_name="exp07_ours_full_model"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "dino_feat" `# <-- 组合1` \
    --synergy_mode_spatial `# <-- 组合2` \
    --intelligent_densification `# <-- 组合3` \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.005


# --- [ 3. 新增三组协同与探索实验 ] ---

# --- 实验 08: [极限性能] DINO + SVAS + DEID + OAI (所有模组) ---
exp_name="exp08_ours_ultimate_all_modules"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "dino_feat" `# <-- 组合1` \
    --synergy_mode_spatial `# <-- 组合2` \
    --intelligent_densification `# <-- 组合3` \
    --optimizer_intervention `# <-- 组合4` \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.005


# --- 实验 09: [高效协同] DINO + DEID + OAI (跳过SVAS) ---
exp_name="exp09_synergy_dino_deid_oai"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "dino_feat" `# <-- 组合1` \
    --synergy_mode "v5_error_dynamics" `# <-- 使用基础的动态协同` \
    --intelligent_densification `# <-- 组合2` \
    --optimizer_intervention `# <-- 组合3` \
    --lambda_struct_loss 0.0 \
    --lambda_struct_loss_base 0.1 \
    --lambda_grad_loss_base 0.05 \
    --error_dynamics_threshold 0.005


# --- 实验 10: [外部对比] 物理Alpha (Physical Alpha) ---
# 注意：这是一个完全不同的分支，它不使用动态lambda或OAI等。
exp_name="exp10_alternative_physical_alpha"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$MY_PROJECT_DIR" "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --structural_loss_mode "ms_grad" \
    --synergy_mode "v5_physical_alpha" `# <-- 核心改动` \
    --lambda_struct_loss 0.05 `# physical_alpha 使用固定的结构损失`

# ========================== [ 实验组结束 ] ==========================
echo; echo "### 🎉🎉🎉 V6 核心模组对决实验 (最终扩展版) 执行完毕！ ###";
echo "请检查目录 ${EXPERIMENTS_ROOT_DIR}/${SCENE} 以获取所有实验的结果。";