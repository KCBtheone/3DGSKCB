#!/bin/bash
set -e # 任何命令失败则立即退出，确保健壮性

# ===================================================================================
#      多场景最终版扫描脚本 (v8.0 - 硬编码最佳配置)
#
# 核心逻辑:
#  - 基于已有的分析结果，为每个场景硬编码其最佳的L1约束配置。
#  - 自动跳过已完成的exp00-exp05实验。
#  - 基于各场景的最佳配置，继续完成exp06-exp09的实验。
#  - 最后，为每个场景生成独立的 _ranking.txt 报告。
# ===================================================================================

# --- [ 1. 终止信号陷阱 ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###  检测到 Ctrl+C！正在强制终止所有子进程...  ###" && kill -9 -$$
}

# --- [ 2. 全局配置区 ] ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$PROJECT_DIR/data/nerf_360"
# [!] 确保这个目录与您已有的实验结果目录一致
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/output/MULTISCENE_10G_TRUE_BASELINE_SWEEP"

# --- [ 3. 场景列表 ] ---
SCENES_TO_RUN=( "bicycle" "bonsai" "counter" "kitchen" "room" "stump" "garden" )

# --- [ 4. 固定实验配置 ] ---
ITERATIONS=15000
GEOMETRY_START_ITER=7000; ISOTROPY_START_ITER=5000

# =================================================================================

# --- [ 5. 核心辅助函数 ] ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [场景: ${scene_name} | 实验: ${exp_name}] ---";
    if [ -d "${model_path}" ]; then echo "        -> 结果已存在，跳过。"; return; fi
    echo "        -> 输出至: ${model_path}"; mkdir -p "${model_path}"
    export CUDA_LAUNCH_BLOCKING=1
    stdbuf -oL -eL python "${PROJECT_DIR}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    local exit_code=${PIPESTATUS[0]}
    export CUDA_LAUNCH_BLOCKING=0
    if [ ${exit_code} -ne 0 ]; then
        echo "        -> ❌ 失败！(错误码 ${exit_code})。标记失败并继续。"; touch "${model_path}/_FAILED.log";
    else
        echo "        -> ✅ 成功完成。";
    fi
}

# --- [ 6. 实验调度 ] ---
echo "🚀🚀🚀 开始运行多场景最终版扫描 (硬编码最佳配置) 🚀🚀🚀"
cd "$PROJECT_DIR" || exit

for scene in "${SCENES_TO_RUN[@]}"; do
    echo; echo "############################################################"
    echo "###    开始处理场景: [${scene}]"
    echo "############################################################"
    scene_path="$DATA_ROOT_DIR/$scene"; scene_output_root="$EXPERIMENTS_ROOT_DIR/$scene"
    resolution=8; image_subdir="images_8"
    if [ "$scene" == "bonsai" ]; then resolution=4; image_subdir="images_4"; fi
    
    priors_source_path="$scene_path/derived_data"; priors_target_path="$scene_path/priors"
    if [ ! -d "$priors_source_path" ]; then echo "       -> ❌ 错误: 场景 [${scene}] 缺少 'derived_data'。跳过。" && continue; fi

    # --- [!] 硬编码各场景的最佳L1配置 ---
    best_alpha_l1=0.0
    case $scene in
        kitchen)
            best_alpha_l1=0.1
            echo "       -> 场景 [kitchen] 已指定最佳 alpha = 0.1"
            ;;
        bicycle|room|counter|garden|bonsai|stump)
            best_alpha_l1=0.0 # 对应 exp01 (gamma=1.0) 或 exp02 (gamma=0.3)
            echo "       -> 场景 [${scene}] 已指定最佳 alpha = 0.0"
            ;;
        *)
            echo "       -> 警告: 场景 [${scene}] 无特定配置, 使用默认 alpha = 0.0"
            best_alpha_l1=0.0
            ;;
    esac

    # --- 阶段三：SA-SSIM及消融实验 (exp06-08) ---
    echo; echo "### [阶段三] 基于最佳L1配置 [alpha=${best_alpha_l1}] 进行SA-SSIM实验... ###"
    if [ -L "$priors_target_path" ]; then rm -f "$priors_target_path"; fi; ln -s "$priors_source_path" "$priors_target_path"
    
    l1_base_args=(-s "$scene_path" --images "$image_subdir" --iterations "$ITERATIONS" --resolution "$resolution" --eval --confidence_gamma 0.3 --save_iterations 7000 "$ITERATIONS" --checkpoint_iterations "$ITERATIONS")
    ssim_base_args=("${l1_base_args[@]}" --alpha_normals "$best_alpha_l1" --geometry_start_iter "$GEOMETRY_START_ITER")
    
    exp_name="exp06_l1best_plus_sassim"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${ssim_base_args[@]}" -m "$model_path" --use_sa_ssim --adaptive_gamma --beta_geo 0.5 --gamma_base 1.0 --gamma_warmup 5000
    exp_name="exp07_ablation_no_adaptive_gamma"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${ssim_base_args[@]}" -m "$model_path" --use_sa_ssim --beta_geo 0.5 --gamma_base 1.0
    exp_name="exp08_ablation_beta0p8"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${ssim_base_args[@]}" -m "$model_path" --use_sa_ssim --adaptive_gamma --beta_geo 0.8 --gamma_base 1.0 --gamma_warmup 5000

    # --- 阶段四：最终的“王者归来”浮游物实验 (exp09) ---
    echo; echo "### [阶段四] 基于前序总冠军，进行最终的浮游物约束实验... ###"
    # 我们需要先跑完06-08，再找出00-08的总冠军
    
    # 评选00-08的总冠军
    best_psnr_overall=0; best_exp_name_overall=""
    for i in {0..8}; do
        exp_prefix="exp$(printf "%02d" $i)"; exp_dir=$(find "$scene_output_root" -maxdepth 1 -type d -name "${exp_prefix}*")
        if [ -z "$exp_dir" ]; then continue; fi
        log_file="${exp_dir}/console.log"; if [ ! -f "$log_file" ] || [ -f "${exp_dir}/_FAILED.log" ]; then continue; fi
        current_psnr=$(tac "$log_file" | grep -m 1 -E "^\[ITER [0-9]+\] Validation Results: L1" | awk '{print $NF}')
        if [[ "$current_psnr" =~ ^[0-9]+(\.[0-9]+)?$ ]] && (( $(echo "$current_psnr > $best_psnr_overall" | bc -l) )); then
            best_psnr_overall=$current_psnr; best_exp_name_overall=$(basename "$exp_dir")
        fi
    done

    if [ -z "$best_exp_name_overall" ]; then
        echo "        -> ❌ 警告: 场景 [${scene}] 的前9组实验均失败，无法进行最终的浮游物约束实验。";
    else
        echo "        -> 读取到前9组总冠军: [${best_exp_name_overall}]"
        
        # 从冠军配置中解析出所有参数
        best_alpha=$(echo "$best_exp_name_overall" | sed -n 's/.*_a\([0-9]p[0-9]\)_.*/\1/p' | tr 'p' '.'); best_alpha=${best_alpha:-0.0}
        
        # 准备冠军配置的参数
        vanilla_args_for_winner=(-s "$scene_path" --images "$image_subdir" --iterations "$ITERATIONS" --resolution "$resolution" --eval --save_iterations 7000 "$ITERATIONS" --checkpoint_iterations "$ITERATIONS")
        l1_args_for_winner=(-s "$scene_path" --images "$image_subdir" --iterations "$ITERATIONS" --resolution "$resolution" --eval --confidence_gamma 0.3 --save_iterations 7000 "$ITERATIONS" --checkpoint_iterations "$ITERATIONS")
        winner_args=()
        
        if [[ "$best_exp_name_overall" == "exp00_true_baseline" ]]; then
            winner_args=("${vanilla_args_for_winner[@]}")
        elif [[ "$best_exp_name_overall" == "exp01_gamma1p0_baseline" ]]; then
            # exp01 使用 vanilla_args 但有 priors
            winner_args=("${vanilla_args_for_winner[@]}")
        else
            winner_args=("${l1_args_for_winner[@]}" --alpha_normals "$best_alpha")
        fi

        # 添加SA-SSIM相关参数（如果冠军配置包含它们）
        if [[ "$best_exp_name_overall" == *"sassim"* ]] || [[ "$best_exp_name_overall" == *"ablation"* ]]; then
            winner_args+=(--use_sa_ssim)
            if [[ "$best_exp_name_overall" != *"no_adaptive"* ]]; then winner_args+=(--adaptive_gamma --gamma_warmup 5000); fi
            if [[ "$best_exp_name_overall" == *"beta0p8"* ]]; then winner_args+=(--beta_geo 0.8); else winner_args+=(--beta_geo 0.5); fi
            winner_args+=(--gamma_base 1.0)
        fi

        exp_name="exp09_winner_plus_floaters"; model_path="${scene_output_root}/${exp_name}"
        run_single_experiment "$scene" "$exp_name" "$model_path" "${winner_args[@]}" -m "$model_path" --lambda_isotropy 0.1 --isotropy_start_iter "$ISOTROPY_START_ITER"
    fi
    
    rm -f "$priors_target_path"
done

# --- [ 最终总结报告生成 ] ---
# (此部分与v7.1版本完全一致，用于最后生成独立的txt报告)
echo; echo "######################################################################"
echo "### 📊 所有实验运行完毕！开始为每个场景生成最终性能报告... ###"
echo "######################################################################"
for scene in "${SCENES_TO_RUN[@]}"; do
    scene_output_root="$EXPERIMENTS_ROOT_DIR/$scene"; if [ ! -d "$scene_output_root" ]; then continue; fi
    report_file="${scene_output_root}/_final_ranking.txt"; echo "--- 场景 [${scene}] 性能排行榜 ---" > "$report_file"; echo "====================================" >> "$report_file"
    declare -A results_map
    for exp_dir in "$scene_output_root"/*/; do
        exp_name=$(basename "$exp_dir"); log_file="${exp_dir}/console.log"
        if [ ! -f "$log_file" ] || [ -f "${exp_dir}/_FAILED.log" ]; then continue; fi
        max_psnr=$(grep -E "^\[ITER [0-9]+\] Validation Results: L1" "$log_file" | awk '{print $NF}' | sort -nr | head -1)
        if [[ "$max_psnr" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then results_map["$exp_name"]=$max_psnr; fi
    done
    if [ ${#results_map[@]} -eq 0 ]; then echo "该场景没有成功的实验结果" >> "$report_file"; else
        printf "%s\n" "${!results_map[@]}" | while read -r exp; do printf "%s | %s\n" "$exp" "${results_map[$exp]}"; done | sort -t '|' -k 2 -r -n | while IFS= read -r line; do
            exp_name=$(echo "$line" | awk -F ' | ' '{print $1}'); psnr_val=$(echo "$line" | awk -F ' | ' '{print $2}')
            printf "%-40s | %s\n" "$exp_name" "$psnr_val" >> "$report_file"
        done
        champion=$(printf "%s\n" "${!results_map[@]}" | while read -r exp; do printf "%s | %s\n" "$exp" "${results_map[$exp]}"; done | sort -t '|' -k 2 -r -n | head -1 | awk -F ' | ' '{print $1}')
        echo "====================================" >> "$report_file"; echo "🏆 本场景冠军: ${champion}" >> "$report_file"
    fi
    echo " -> 报告已生成: ${report_file}"
done

echo; echo "######################################################################"
echo "### 🎉🎉🎉 批量扫描与报告生成全部完成！ ###"
echo "######################################################################"