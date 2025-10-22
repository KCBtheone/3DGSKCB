#!/bin/bash
# set -x # 如果还需要调试，可以取消此行的注释

# ===================================================================================
#      Bonsai场景几何增强策略扫描脚本 (v1.2 - 20k迭代优化版)
#
# 1. 场景固定为 "bonsai"。
# 2. [核心] 总迭代次数优化为 20000 次，用于快速验证。
# 3. [核心] 几何约束起始点按比例提前至 6000 次。
# 4. 系统性地测试策略 #2 (平滑度损失) 和策略 #3 (几何致密化) 的效果。
# ===================================================================================

# --- [ 1. 终止信号陷阱 ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###  检测到 Ctrl+C！正在强制终止所有子进程...  ###" && kill -9 -$$
}

# --- [ 2. 全局配置区 ] ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$PROJECT_DIR/data/nerf_360"
# 新的输出目录，以反映20k的迭代设置
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/output/BONSAI_GEOMETRY_ENHANCED_SWEEP_20K"

# --- [ 3. 场景列表 ] ---
SCENE="bonsai"

# --- [ 4. 固定实验配置 ] ---
ITERATIONS=20000
# 在7k和最终迭代时保存和测试
SAVE_AND_TEST_ITERS="7000 ${ITERATIONS}"

# 按比例提前的几何约束起始迭代次数 (大约在总时长的 1/3 处)
GEOMETRY_START_ITER=6000

# =================================================================================
#            [ 核心辅助函数：与您的脚本相同 ]
# =================================================================================

update_ranking_file() {
    local model_path=$1; local exp_name=$2
    local ranking_file=$(dirname "$model_path")/_ranking.txt; local log_file="${model_path}/console.log"
    
    if [ ! -f "$log_file" ] || [ -f "${model_path}/_FAILED.log" ]; then
        echo "${exp_name} | FAILED" >> "$ranking_file"; return;
    fi
    
    # 只取最终迭代 (20000) 的PSNR进行排名
    local final_psnr=$(grep -E "^\[ITER ${ITERATIONS}\] Validation Results: L1" "$log_file" | awk '{print $NF}' | tail -1)
    
    if [[ "$final_psnr" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        printf "%-45s | %s\n" "${exp_name}" "${final_psnr}" >> "$ranking_file"
        echo "        -> 📈 记录到排名文件: Final PSNR = ${final_psnr}"
    else
        printf "%-45s | %s\n" "${exp_name}" "PARSE_ERROR" >> "$ranking_file"
        echo "        -> ⚠️ 无法从日志解析最终PSNR，已记录错误。"
    fi
}

# --- [ 5. 主执行函数 ] ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [场景: ${scene_name} | 实验: ${exp_name}] ---";
    
    if [ -d "${model_path}" ]; then echo "        -> 结果已存在，跳过。"; return; fi
    
    echo "        -> 输出至: ${model_path}"; mkdir -p "${model_path}"
    export CUDA_LAUNCH_BLOCKING=1
    
    stdbuf -oL -eL python "${PROJECT_DIR}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    local exit_code=${PIPESTATUS[0]}
    
    export CUDA_LAUNCH_BLOCKING=0
    
    if [ ${exit_code} -eq 0 ]; then
        echo "        -> ✅ 成功完成。"; update_ranking_file "$model_path" "$exp_name";
    else
        echo "        -> ❌ 失败！(错误码 ${exit_code})。"; touch "${model_path}/_FAILED.log"; update_ranking_file "$model_path" "$exp_name";
    fi
}

# --- [ 6. 实验调度 ] ---
echo "🚀🚀🚀 开始运行 Bonsai 几何增强策略扫描 (20k 迭代优化版) 🚀🚀🚀"
cd "$PROJECT_DIR" || exit

echo; echo "############################################################"
echo "###    开始处理场景: [${SCENE}]"
echo "############################################################"
scene_path="$DATA_ROOT_DIR/$SCENE"; scene_output_root="$EXPERIMENTS_ROOT_DIR/$SCENE"
resolution=4; image_subdir="images_4"

ranking_file="${scene_output_root}/_ranking.txt"; 
echo "# ${SCENE} 场景几何增强策略性能排行榜 (Final PSNR @ ${ITERATIONS} iters)" > "$ranking_file"
echo "------------------------------------------------------------------" >> "$ranking_file"

# --- 基础参数 (所有实验共享) ---
# densify_until_iter 也需要相应缩短，官方默认是15k，对于20k的总迭代，我们可以设为12k-15k，这里取15k
base_args=(-s "$scene_path" --images "$image_subdir" --iterations "$ITERATIONS" --resolution "$resolution" --eval \
    --save_iterations $SAVE_AND_TEST_ITERS \
    --test_iterations $SAVE_AND_TEST_ITERS \
    --checkpoint_iterations $SAVE_AND_TEST_ITERS \
    --densify_until_iter 15000
)

# ========================= [ 实验组开始 ] =========================

# --- 实验 01: 基线 (Baseline) ---
# 不使用任何新的几何增强策略，作为所有对比的基准。
exp_name="exp01_baseline"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path"

# --- 实验 02-04: 单独测试策略 #2 (法线平滑度损失) ---
# 在 6000 次迭代后引入

# exp02: 平滑度损失 (弱)
lambda=0.001; exp_name="exp02_smooth_weak_lambda${lambda//./p}"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --use_smoothness_loss --lambda_smooth "$lambda" --smooth_start_iter "$GEOMETRY_START_ITER"

# exp03: 平滑度损失 (中)
lambda=0.01; exp_name="exp03_smooth_medium_lambda${lambda//./p}"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --use_smoothness_loss --lambda_smooth "$lambda" --smooth_start_iter "$GEOMETRY_START_ITER"

# exp04: 平滑度损失 (强)
lambda=0.05; exp_name="exp04_smooth_strong_lambda${lambda//./p}"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --use_smoothness_loss --lambda_smooth "$lambda" --smooth_start_iter "$GEOMETRY_START_ITER"

# --- 实验 05: 单独测试策略 #3 (几何感知的致密化) ---
# 同样在 6000 次迭代后引入
exp_name="exp05_geo_densify_only"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --use_geometric_densify --geo_densify_start_iter "$GEOMETRY_START_ITER"

# --- 实验 06: 组合测试 (策略 #2 + #3) ---
# 将两个策略结合，使用一个中等强度的平滑度损失
lambda=0.01; exp_name="exp06_combined_smooth_medium_geo_densify"; model_path="${scene_output_root}/${exp_name}";
run_single_experiment "$SCENE" "$exp_name" "$model_path" "${base_args[@]}" -m "$model_path" \
    --use_smoothness_loss --lambda_smooth "$lambda" --smooth_start_iter "$GEOMETRY_START_ITER" \
    --use_geometric_densify --geo_densify_start_iter "$GEOMETRY_START_ITER"

# ========================== [ 实验组结束 ] ==========================

# --- [ 最终总结 ] ---
echo; echo "######################################################################"
echo "### 🎉🎉🎉 Bonsai 场景的几何增强策略实验执行完毕！ ###"
echo "### 性能排行榜已保存在 ${ranking_file} 文件中。 ###"
echo "######################################################################"
# 自动排序并显示最终排名
echo; echo "--- 最终性能排名 (PSNR @ ${ITERATIONS} iters) ---"
(head -n 2 "$ranking_file" && tail -n +3 "$ranking_file" | sort -k3 -nr) | column -t -s '|'
echo "------------------------------------------------"