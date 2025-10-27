#!/bin/bash
# set -x # 如果需要调试，可以取消此行的注释

# ===================================================================================
#      感知损失对决实验 (v1.2 - Bonsai快速对焦版)
#
# 设计目标:
# 1. [聚焦] 集中所有计算资源在 `bonsai` 单一场景上进行深度消融实验。
# 2. [快速] 使用 20000 次迭代和 8 倍降采样，以在有限时间内获得结果。
# 3. [纯净] 所有实验均在官方、未经修改的致密化逻辑上运行，排除一切干扰。
# 4. [深入] 扩展至8组实验，系统性地验证你的方法、感知损失以及它们之间的协同效应。
# ===================================================================================

# --- [ 1. 全局配置与辅助函数 ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###  检测到 Ctrl+C！正在强制终止所有子进程...  ###" && kill -9 -$$
}

# --- 请根据您的环境修改以下路径 ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting" # 您的项目根目录
DATA_ROOT_DIR="$PROJECT_DIR/data/nerf_360"        # 您的数据集根目录
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/output/RAPID_PERCEPTUAL_BONSAI" # 新的、独立的实验输出目录
# ---------------------------------

# --- [核心] 实验参数 ---
SCENE_NAME="bonsai" # 聚焦单一场景
ITERATIONS=20000
RESOLUTION=8 # 8倍降采样以加速
IMAGES_SUBDIR="images_${RESOLUTION}"
SAVE_AND_TEST_ITERS="7000 ${ITERATIONS}"

# --- 辅助函数：运行单个实验 ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [场景: ${scene_name} | 实验: ${exp_name}] ---";
    if [ -d "${model_path}" ]; then
        echo "        -> 结果已存在，跳过。"
        return
    fi
    echo "        -> 输出至: ${model_path}";
    mkdir -p "${model_path}";
    
    stdbuf -oL -eL python "${PROJECT_DIR}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    
    local exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -eq 0 ]; then
        echo "        -> ✅ 成功完成: ${exp_name}"
    else
        echo "        -> ❌ 失败！(错误码 ${exit_code}): ${exp_name}"
        touch "${model_path}/_FAILED.log"
    fi
}

# --- [ 2. 实验调度 ] ---
echo "🚀🚀🚀 开始运行感知损失对决实验 (v1.2 - Bonsai快速对焦版) 🚀🚀🚀"
cd "$PROJECT_DIR" || exit

scene_path="$DATA_ROOT_DIR/$SCENE_NAME"
scene_output_root="$EXPERIMENTS_ROOT_DIR/$SCENE_NAME"
mkdir -p "$scene_output_root"

# [核心修改] 基础参数现在不包含任何致密化修复
base_args=(-s "$scene_path" --images "$IMAGES_SUBDIR" --iterations "$ITERATIONS" --resolution "$RESOLUTION" --eval \
    --save_iterations $SAVE_AND_TEST_ITERS --test_iterations $SAVE_AND_TEST_ITERS --densify_until_iter 15000
)

echo; echo "===================================================================="
echo "                   开始处理场景: ${SCENE_NAME}"
echo "===================================================================="

# ===================================================================================
#                      核心8组消融实验 (全部基于官方致密化逻辑)
# ===================================================================================

# --- 组 1: 【纯净基准】 Official Vanilla Baseline ---
# 100% 官方逻辑，不含任何修改。这是衡量一切改进的“绝对零点”。
exp_name="exp01_official_baseline"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "none"

# --- 组 2: 【我的方法基准】 My Method Baseline ---
# 你的核心方法（几何引导+置信度解耦），用于对比感知损失带来的附加值。
exp_name="exp02_my_method_baseline"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "dual_l1" \
    --use_normal_guidance \
    --confidence_thresh 0.3 \
    --lambda_low_confidence 0.05

# --- 组 3: 【消融】 纯净基准 + 感知损失 ---
# 在官方基准上只加入感知损失，用于独立评估感知损失本身的效果。
exp_name="exp03_official_plus_perceptual"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "none" \
    --use_perceptual_loss --lambda_perceptual 0.01 --perceptual_start_iter 10000

# --- 组 4: 【协同】 我的方法 + 感知损失 ---
# 在你的方法基准上加入感知损失，测试协同效应。
exp_name="exp04_my_method_plus_perceptual"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "dual_l1" \
    --use_normal_guidance \
    --confidence_thresh 0.3 \
    --lambda_low_confidence 0.05 \
    --use_perceptual_loss --lambda_perceptual 0.01 --perceptual_start_iter 10000

# --- 组 5: 【强化】 我的方法 + 强感知损失 ---
# 在你的方法上，增强感知损失的权重和作用时间，探索其潜力。
exp_name="exp05_my_method_strong_perceptual"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "dual_l1" \
    --use_normal_guidance \
    --confidence_thresh 0.3 \
    --lambda_low_confidence 0.05 \
    --use_perceptual_loss --lambda_perceptual 0.05 --perceptual_start_iter 7000 # 权重x5, 提前启动

# --- 组 6: 【视觉巅峰候选】 我的方法 + 强感知 + 清理 ---
# 在组5的基础上，加入浮游物清理，旨在达到最佳视觉质量 (LPIPS)。
exp_name="exp06_visual_peak_candidate"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "dual_l1" \
    --use_normal_guidance \
    --confidence_thresh 0.3 \
    --lambda_low_confidence 0.05 \
    --use_perceptual_loss --lambda_perceptual 0.05 --perceptual_start_iter 7000 \
    --use_isotropy_loss --lambda_isotropy 0.2 --isotropy_start_iter 10000

# --- 组 7: 【激进炼丹】 感知主导 ---
# 进一步提高感知损失权重，同时降低SSIM权重，让模型更关注感知真实性而非结构相似性。
exp_name="exp07_perceptual_dominant"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "dual_l1" \
    --use_normal_guidance \
    --lambda_dssim 0.1 \
    --use_perceptual_loss --lambda_perceptual 0.1 --perceptual_start_iter 7000

# --- 组 8: 【纯粹感知消融】 Pure Perceptual Ablation ---
# 一个重要的对照组：抛弃你的方法，只用最基础的法线引导+强感知损失。
exp_name="exp08_ablation_pure_perceptual"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE_NAME" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --confidence_scheme "none" \
    --use_normal_guidance \
    --lambda_dssim 1.0 \
    --use_perceptual_loss --lambda_perceptual 0.2 --perceptual_start_iter 1000

# ========================== [ 实验组结束 ] ==========================
echo; echo "### 🎉🎉🎉 Bonsai快速对焦实验执行完毕！ ###";
echo "请检查目录 ${EXPERIMENTS_ROOT_DIR}/${SCENE_NAME} 以获取结果。";
echo "分析建议: "
echo "1. (exp01 vs exp03): 感知损失本身有多大效果？"
echo "2. (exp02 vs exp04): 感知损失在你的方法上能带来多少附加值？"
echo "3. 重点对比 exp02, exp04, exp05, exp06 的 PSNR 和 LPIPS 指标，找到最佳的权衡点。"
echo "4. exp07 和 exp08 的结果将告诉你感知损失的极限在哪里。"