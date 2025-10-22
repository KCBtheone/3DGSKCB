#!/bin/bash
# set -x # 如果还需要调试，可以取消此行的注释

# ===================================================================================
#      多场景最终决定版扫描脚本 (v7.5 - 保存 15k .pth 检查点)
#
# 1. 实验逻辑保持 v7.4 的 "10组真·基线" 方案不变。
# 2. [!] 按照要求，为所有实验添加 --checkpoint_iterations 15000，
#        以便在训练结束时保存 .pth 训练检查点。
# ===================================================================================

# --- [ 1. 终止信号陷阱 ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM
cleanup_and_exit() {
    echo "" && echo "###  检测到 Ctrl+C！正在强制终止所有子进程...  ###" && kill -9 -$$
}

# --- [ 2. 全局配置区 ] ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$PROJECT_DIR/data/nerf_360"
# [!] 结果将保存在这个新目录中
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/output/MULTISCENE_10G_TRUE_BASELINE_SWEEP"

# --- [ 3. 场景列表 ] ---
SCENES_TO_RUN=( "bicycle" "bonsai" "counter" "kitchen" "room" "stump" "garden" )

# --- [ 4. 固定实验配置 ] ---
ITERATIONS=15000
GEOMETRY_START_ITER=7000; ISOTROPY_START_ITER=5000

# =================================================================================
#           [ 核心辅助函数：动态读写TXT ]
# =================================================================================

update_ranking_file() {
    local model_path=$1; local exp_name=$2
    local ranking_file=$(dirname "$model_path")/_ranking.txt; local log_file="${model_path}/console.log"
    
    if [ ! -f "$log_file" ] || [ -f "${model_path}/_FAILED.log" ]; then
        echo "${exp_name} | FAILED" >> "$ranking_file"; return;
    fi
    
    local max_psnr=$(grep -E "^\[ITER [0-9]+\] Validation Results: L1" "$log_file" | awk '{print $NF}' | sort -nr | head -1)
    
    if [[ "$max_psnr" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "${exp_name} | ${max_psnr}" >> "$ranking_file"
        echo "        -> 📈 记录到排名文件: PSNR = ${max_psnr}"
    else
        echo "${exp_name} | PARSE_ERROR" >> "$ranking_file"
        echo "        -> ⚠️ 无法从日志解析PSNR，已记录错误。"
    fi
}

get_best_config_from_file() {
    local ranking_file=$1; shift; local search_prefixes=("$@")
    
    if [ ! -f "$ranking_file" ]; then echo "PARSE_FAIL"; return; fi
    
    local best_psnr=0; local best_name=""
    for prefix in "${search_prefixes[@]}"; do
        local line=$(grep "^${prefix}" "$ranking_file" | grep -v "FAILED" | grep -v "PARSE_ERROR" | sort -t '|' -k 2 -nr | head -1)
        
        if [ -n "$line" ]; then
            local current_psnr=$(echo "$line" | awk -F ' | ' '{print $3}')
            
            if (( $(echo "$current_psnr > $best_psnr" | bc -l) )); then
                best_psnr=$current_psnr; best_name=$(echo "$line" | awk -F ' | ' '{print $1}')
            fi
        fi
    done
    
    if [ -n "$best_name" ]; then echo "$best_name"; else echo "PARSE_FAIL"; fi
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
echo "🚀🚀🚀 开始运行多场景10组("真·基线" + .pth)扫描 v7.5 🚀🚀🚀"
cd "$PROJECT_DIR" || exit

for scene in "${SCENES_TO_RUN[@]}"; do
    echo; echo "############################################################"
    echo "###    开始处理场景: [${scene}]"
    echo "############################################################"
    scene_path="$DATA_ROOT_DIR/$scene"; scene_output_root="$EXPERIMENTS_ROOT_DIR/$scene"
    resolution=8; image_subdir="images_8"
    
    if [ "$scene" == "bonsai" ]; then resolution=4; image_subdir="images_4"; fi
    
    priors_source_path="$scene_path/derived_data"; priors_target_path="$scene_path/priors"
    
    if [ ! -d "$priors_source_path" ]; then echo "        -> ❌ 错误: 场景 [${scene}] 缺少 'derived_data'。跳过。" && continue; fi
    
    ranking_file="${scene_output_root}/_ranking.txt"; echo "# ${scene} 场景性能排行榜 (动态更新)" > "$ranking_file"

    # --- 基础参数 ---
    # [!] 已添加 --checkpoint_iterations "$ITERATIONS"
    vanilla_args=(-s "$scene_path" --images "$image_subdir" --iterations "$ITERATIONS" --resolution "$resolution" --eval \
        --save_iterations 7000 "$ITERATIONS" \
        --checkpoint_iterations "$ITERATIONS")
    
    # [!] 已添加 --checkpoint_iterations "$ITERATIONS"
    l1_base_args=(-s "$scene_path" --images "$image_subdir" --iterations "$ITERATIONS" --resolution "$resolution" --eval --confidence_gamma 0.3 \
        --save_iterations 7000 "$ITERATIONS" \
        --checkpoint_iterations "$ITERATIONS")

    # --- 实验 00: "真·基线" (无Priors) ---
    echo "### [阶段 0] 运行 exp00 (真·基线) - 必须 *没有* priors 软链接 ###"
    if [ -L "$priors_target_path" ]; then rm -f "$priors_target_path"; fi
    
    exp_name="exp00_true_baseline"; model_path="${scene_output_root}/${exp_name}"; 
    run_single_experiment "$scene" "$exp_name" "$model_path" "${vanilla_args[@]}" -m "$model_path"

    # --- 创建Priors软链接 (后续所有实验都需要) ---
    echo "### [系统] 创建 priors 软链接供后续实验使用... ###"
    if [ ! -L "$priors_target_path" ]; then ln -s "$priors_source_path" "$priors_target_path"; fi

    # --- 实验 01: Gamma=1.0 基线 (有Priors) ---
    exp_name="exp01_gamma1p0_baseline"; model_path="${scene_output_root}/${exp_name}"; 
    run_single_experiment "$scene" "$exp_name" "$model_path" "${vanilla_args[@]}" -m "$model_path"

    # --- 实验 02-05: L1 搜索 (有Priors) ---
    exp_name="exp02_l1_gamma_only"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${l1_base_args[@]}" -m "$model_path"
    alpha=0.1; exp_name="exp03_l1_a${alpha//./p}_l0p0"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${l1_base_args[@]}" -m "$model_path" --alpha_normals "$alpha" --geometry_start_iter "$GEOMETRY_START_ITER"
    alpha=0.2; exp_name="exp04_l1_a${alpha//./p}_l0p0"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${l1_base_args[@]}" -m "$model_path" --alpha_normals "$alpha" --geometry_start_iter "$GEOMETRY_START_ITER"
    alpha=0.4; exp_name="exp05_l1_a${alpha//./p}_l0p0"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${l1_base_args[@]}" -m "$model_path" --alpha_normals "$alpha" --geometry_start_iter "$GEOMETRY_START_ITER"

    # --- 动态决策 1: 寻找最佳L1配置 ---
    echo; echo "### [动态决策1] 从TXT读取最佳L1配置 (exp02-05)... ###"
    best_l1_exp_name=$(get_best_config_from_file "$ranking_file" "exp02" "exp03" "exp04" "exp05")
    
    if [ "$best_l1_exp_name" == "PARSE_FAIL" ]; then
        echo "        -> ❌ 警告: L1阶段无成功结果，跳过后续SA-SSIM实验。";
    else
        best_alpha_l1=$(echo "$best_l1_exp_name" | sed -n 's/.*_a\([0-9]p[0-9]\)_.*/\1/p' | tr 'p' '.'); best_alpha_l1=${best_alpha_l1:-0.0}
        echo "        -> 读取到最佳L1配置: [${best_l1_exp_name}] (alpha=${best_alpha_l1})"
        
        # 实验 06-08: 基于最佳L1配置的SA-SSIM实验
        ssim_base_args=("${l1_base_args[@]}" --alpha_normals "$best_alpha_l1" --geometry_start_iter "$GEOMETRY_START_ITER")
        exp_name="exp06_l1best_plus_sassim"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${ssim_base_args[@]}" -m "$model_path" --use_sa_ssim --adaptive_gamma --beta_geo 0.5 --gamma_base 1.0 --gamma_warmup 5000
        exp_name="exp07_ablation_no_adaptive_gamma"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${ssim_base_args[@]}" -m "$model_path" --use_sa_ssim --beta_geo 0.5 --gamma_base 1.0
        exp_name="exp08_ablation_beta0p8"; model_path="${scene_output_root}/${exp_name}"; run_single_experiment "$scene" "$exp_name" "$model_path" "${ssim_base_args[@]}" -m "$model_path" --use_sa_ssim --adaptive_gamma --beta_geo 0.8 --gamma_base 1.0 --gamma_warmup 5000
    fi
    
    # --- 动态决策 2: 寻找前9组的总冠军 (exp00 ~ exp08) ---
    echo; echo "### [动态决策2] 从TXT读取前9组的总冠军... ###"
    best_overall_exp_name=$(get_best_config_from_file "$ranking_file" "exp00" "exp01" "exp02" "exp03" "exp04" "exp05" "exp06" "exp07" "exp08")
    
    if [ "$best_overall_exp_name" == "PARSE_FAIL" ]; then
        echo "        -> ❌ 警告: 前9组无成功结果，跳过最终的浮游物约束实验。";
    else
        echo "        -> 读取到前9组总冠军: [${best_overall_exp_name}]"
        
        # --- 实验 09: "王者归来" ---
        # 从冠军配置中解析出所有参数
        best_alpha=$(echo "$best_overall_exp_name" | sed -n 's/.*_a\([0-9]p[0-9]\)_.*/\1/p' | tr 'p' '.'); best_alpha=${best_alpha:-0.0}
        
        # 准备冠军配置的参数
        winner_args=()
        # 检查是否是vanilla组 (!! 新增了 exp00 的检查 !!)
        if [[ "$best_overall_exp_name" == "exp00_true_baseline" ]] || [[ "$best_overall_exp_name" == "exp01_gamma1p0_baseline" ]]; then
            winner_args=("${vanilla_args[@]}")
        else
            winner_args=("${l1_base_args[@]}" --alpha_normals "$best_alpha")
            if [[ "$best_overall_exp_name" == *"sassim"* ]] || [[ "$best_overall_exp_name" == *"ablation"* ]]; then
                winner_args+=(--use_sa_ssim)
            fi
            if [[ "$best_overall_exp_name" != *"no_adaptive"* ]]; then
                winner_args+=(--adaptive_gamma --gamma_warmup 5000)
            fi
            if [[ "$best_overall_exp_name" == *"beta0p8"* ]]; then
                winner_args+=(--beta_geo 0.8)
            else
                winner_args+=(--beta_geo 0.5)
            fi
            winner_args+=(--gamma_base 1.0) # gamma_base始终为1.0
        fi

        exp_name="exp09_winner_plus_floaters"; model_path="${scene_output_root}/${exp_name}"
        run_single_experiment "$scene" "$exp_name" "$model_path" "${winner_args[@]}" -m "$model_path" --lambda_isotropy 0.1 --isotropy_start_iter "$ISOTROPY_START_ITER"
    fi

    rm -f "$priors_target_path"
done

# --- [ 最终总结 ] ---
echo; echo "######################################################################"
echo "### 🎉🎉🎉 所有场景的实验执行与动态记录已完成！ ###"
echo "### 每个场景的性能排行榜已保存在其各自的 _ranking.txt 文件中。 ###"
echo "######################################################################"