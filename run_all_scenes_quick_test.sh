#!/bin/bash

# ===================================================================================
#      3DGS 选择性超参数搜索与 PSNR 汇总脚本 (v1.0 - 稳定低LR/快速测试)
# ===================================================================================

# --- [ 1. 终止信号陷阱 (保持不变) ] ---
trap 'cleanup_and_exit' SIGINT SIGTERM

cleanup_and_exit() {
    echo ""
    echo "############################################################"
    echo "###   检测到 Ctrl+C！正在强制终止所有子进程...   ###"
    echo "############################################################"
    # 杀死整个进程组
    kill -9 -$$
}

# --- [ 2. 全局配置区 ] ---
PROJECT_DIR="/root/autodl-tmp/gaussian-splatting"
DATA_ROOT_DIR="$PROJECT_DIR/data"
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/selective_sweep_results_stable"

# --- [ 3. 默认保守参数 (其他场景的单组实验参数) ] ---
DEFAULT_ITERATIONS=15000
DEFAULT_RESOLUTION=8 # 降8倍采样
DEFAULT_SCALING_LR=0.0005
DEFAULT_ROTATION_LR=0.0005
DEFAULT_ALPHA=0.0 # 不使用法线约束

# --- [ 4. KICKER 实验矩阵配置区 (保持不变) ] ---
KICKER_ALPHA_VALUES=("0.0" "0.05" "0.2" "0.4")
declare -A KICKER_LR_CONFIGS
KICKER_LR_CONFIGS["default_lr"]="0.005 0.001"
KICKER_LR_CONFIGS["alternative_lr"]="0.005 0.0005"

# --- [ 5. 场景列表 ] ---
KICKER_SCENE="kicker"
OTHER_SCENES=(
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "meadow"
    "office"
    "pipes"
    "playground"
    "relief"
    "relief_2"
)

# --- [ 6. 保存和测试的迭代次数 ] ---
SAVE_AND_CHECKPOINT_ITERS="7000 ${DEFAULT_ITERATIONS}"
TEST_ITERS="7000 ${DEFAULT_ITERATIONS}" # 简化测试点
# =================================================================================

# --- [ 7. 辅助函数 ] ---

# 检查 TXT 文件 (保持不变)
preprocess_colmap_format() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local sparse_dir="$scene_path/sparse/0"
    
    echo "--- [COLMAP TXT 检查] 正在处理场景 [${scene_name}] ---"
    if [ ! -d "$sparse_dir" ]; then
        echo "     -> ❌ 错误: 找不到 COLMAP sparse/0 目录: ${sparse_dir}"
        return 1
    fi
    local cameras_txt="${sparse_dir}/cameras.txt"
    local images_txt="${sparse_dir}/images.txt"
    local points_txt="${sparse_dir}/points3D.txt"
    if [ -f "$cameras_txt" ] && [ -f "$images_txt" ] && [ -f "$points_txt" ]; then
        echo "     -> ✅ 所有必要的 .txt 文件均存在。"
        return 0
    else
        echo "     -> ❌ 错误: 缺少一个或多个 .txt 文件。"
        return 1
    fi
}

# 运行单个实验 (修改：移除超时，只用错误码判断，确保失败继续)
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [${exp_name}] for scene [${scene_name}] ---"
    if [ -d "${model_path}" ]; then echo "       -> 结果已存在，跳过。"; return; fi
    echo "       -> 输出至: ${model_path}"; mkdir -p "${model_path}"
    
    # !!! 关键：移除 timeout 8h，使训练不中断 !!!
    python "${PROJECT_DIR}/train.py" "${python_args[@]}" | tee "${model_path}/console.log"
    local exit_code=${PIPESTATUS[0]} # 获取 python train.py 的退出码
    
    if [ ${exit_code} -eq 0 ]; then 
        echo "       -> ✅ 成功完成。"; 
    else 
        # !!! 关键：实验失败后不返回，而是继续执行脚本 !!!
        echo "       -> ❌ 失败！(错误码 ${exit_code})。标记失败并继续下一个实验。"; 
        touch "${model_path}/_FAILED.log";
    fi
}

# --- [ 8. 场景实验总控 ] ---
run_scene_experiments() {
    local scene_name=$1
    local is_kicker=$2 # 1 if kicker, 0 otherwise
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local scene_name_safe=${scene_name//\//_} 
    local scene_output_dir="$EXPERIMENTS_ROOT_DIR/$scene_name_safe"

    echo; echo "######################################################################"
    echo "###    开始处理场景: [${scene_name}] (KICKER_SWEEP: ${is_kicker}) "
    echo "######################################################################"

    # 步骤1: 检查TXT文件
    preprocess_colmap_format "$scene_name"
    if [ $? -ne 0 ]; then echo "### ❌ 场景 [${scene_name}] COLMAP .txt 文件检查失败，跳过。 ###"; return; fi
    
    # 步骤2: 设置基础参数 (通用)
    local image_subdir="images"
    if [ -d "$scene_path/images/dslr_images_undistorted" ]; then 
        image_subdir="images/dslr_images_undistorted"; 
    fi

    local base_args=(-s "${scene_path}" --images "${image_subdir}" --resolution "${DEFAULT_RESOLUTION}" --iterations "${DEFAULT_ITERATIONS}" --eval --save_iterations ${SAVE_AND_CHECKPOINT_ITERS} --checkpoint_iterations ${SAVE_AND_CHECKPOINT_ITERS} --test_iterations ${TEST_ITERS})

    if [ $is_kicker -eq 1 ]; then
        # === KICKER 场景：运行完整的实验矩阵 ===
        local exp_counter=1
        for lr_name in "${!KICKER_LR_CONFIGS[@]}"; do
            local lr_values=(${KICKER_LR_CONFIGS[$lr_name]})
            local scaling_lr=${lr_values[0]}
            local rotation_lr=${lr_values[1]}
            for alpha in "${KICKER_ALPHA_VALUES[@]}"; do
                local alpha_str="alpha_$(echo $alpha | tr '.' 'p')"
                local exp_name_str="K${exp_counter}/9: ${alpha_str}, ${lr_name}"
                local model_path_str="${scene_output_dir}/exp${exp_counter}_${alpha_str}_${lr_name}"
                local current_args=("${base_args[@]}")
                current_args+=(--model_path "${model_path_str}" --scaling_lr "${scaling_lr}" --rotation_lr "${rotation_lr}")
                
                if (( $(echo "$alpha > 0.0" | bc -l) )); then
                    current_args+=(--geometry_constraint_type normal --alpha_normals "${alpha}" --geometry_start_iter 5000)
                else
                    current_args+=(--geometry_constraint_type none)
                fi
                run_single_experiment "$scene_name_safe" "${exp_name_str}" "${model_path_str}" "${current_args[@]}"
                exp_counter=$((exp_counter + 1))
            done
        done
        echo "✅ 场景 [${scene_name}] 的 8 组实验已尝试运行。"
    else
        # === 其他场景：只运行一组保守实验 ===
        local exp_name_str="S1/1: Base_Conserative"
        local model_path_str="${scene_output_dir}/exp1_base_conservative"
        local current_args=("${base_args[@]}")
        current_args+=(--model_path "${model_path_str}" --scaling_lr "${DEFAULT_SCALING_LR}" --rotation_lr "${DEFAULT_ROTATION_LR}")
        current_args+=(--geometry_constraint_type none) # 仅 baseline
        
        run_single_experiment "$scene_name_safe" "${exp_name_str}" "${model_path_str}" "${current_args[@]}"
        echo "✅ 场景 [${scene_name}] 的 1 组保守实验已尝试运行。"
    fi
}

# ---------------------------------------------------------------------------------
# --- [ 9. 主执行循环 ] ---
# ---------------------------------------------------------------------------------
echo "🚀🚀🚀 开始运行选择性超参数搜索 🚀🚀🚀"
cd "$PROJECT_DIR"

# 1. 运行 Kicker 的完整实验矩阵 (is_kicker=1)
run_scene_experiments "$KICKER_SCENE" 1

# 2. 运行其他场景的单组实验 (is_kicker=0)
for scene in "${OTHER_SCENES[@]}"; do
    run_scene_experiments "$scene" 0
done

echo; echo "######################################################################"
echo "### 🎉🎉🎉 所有场景训练流程执行完毕！开始汇总指标... 🎉🎉🎉"
echo "######################################################################"

# ---------------------------------------------------------------------------------
# --- [ 10. 最终指标汇总 (需要 Python 环境) ] ---
# ---------------------------------------------------------------------------------

# [Python 汇总脚本] 内嵌一个 Python 脚本来解析 CSV 文件并打印最终表格
python - "$EXPERIMENTS_ROOT_DIR" "$KICKER_SCENE" << EOF
import os
import pandas as pd
import numpy as np
import sys
import glob
import re

root_dir = sys.argv[1]
kicker_scene = sys.argv[2]
all_results = []

print("\n--- 正在解析训练日志文件 ---")

# 1. KICKER 场景
kicker_dir = os.path.join(root_dir, kicker_scene)
kicker_logs = glob.glob(os.path.join(kicker_dir, "exp*_alpha_*.csv")) # 匹配所有实验日志

for log_path in glob.glob(os.path.join(kicker_dir, "exp*/training_log.csv")):
    exp_name = os.path.basename(os.path.dirname(log_path))
    
    max_psnr = np.nan
    
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            if 'Test_PSNR' in df.columns and not df['Test_PSNR'].isnull().all():
                df_filtered = df[df['Test_PSNR'] > 10]
                max_psnr = df_filtered['Test_PSNR'].max() if not df_filtered.empty else df['Test_PSNR'].max() 

            all_results.append({
                'Scene': kicker_scene,
                'Experiment': exp_name,
                'Max_Test_PSNR': f"{max_psnr:.4f}" if not np.isnan(max_psnr) else "N/A",
                'Status': 'SUCCESS'
            })
        except Exception as e:
            all_results.append({'Scene': kicker_scene, 'Experiment': exp_name, 'Max_Test_PSNR': 'N/A', 'Status': f"ERROR"})
    else:
        all_results.append({'Scene': kicker_scene, 'Experiment': exp_name, 'Max_Test_PSNR': 'N/A', 'Status': 'LOG NOT FOUND'})

# 2. 其他场景
other_scenes = [
    "courtyard", "delivery_area", "electro", "facade", "meadow", "office", 
    "pipes", "playground", "relief", "relief_2"
]

for scene in other_scenes:
    log_path = os.path.join(root_dir, scene, "exp1_base_conservative", "training_log.csv")
    
    max_psnr = np.nan
    
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            if 'Test_PSNR' in df.columns and not df['Test_PSNR'].isnull().all():
                df_filtered = df[df['Test_PSNR'] > 10]
                max_psnr = df_filtered['Test_PSNR'].max() if not df_filtered.empty else df['Test_PSNR'].max() 

            all_results.append({
                'Scene': scene,
                'Experiment': 'exp1_base_conservative',
                'Max_Test_PSNR': f"{max_psnr:.4f}" if not np.isnan(max_psnr) else "N/A",
                'Status': 'SUCCESS'
            })
        except Exception as e:
            all_results.append({'Scene': scene, 'Experiment': 'exp1_base_conservative', 'Max_Test_PSNR': 'N/A', 'Status': f"ERROR"})
    else:
        all_results.append({'Scene': scene, 'Experiment': 'exp1_base_conservative', 'Max_Test_PSNR': 'N/A', 'Status': 'LOG NOT FOUND'})


# 打印最终表格
if all_results:
    df_final = pd.DataFrame(all_results)
    
    # 尝试将 Max_Test_PSNR 转换为数值以便排序
    df_final['Max_Test_PSNR_Float'] = pd.to_numeric(df_final['Max_Test_PSNR'], errors='coerce')
    df_final = df_final.sort_values(by='Max_Test_PSNR_Float', ascending=False).drop(columns=['Max_Test_PSNR_Float'])
    
    # 重新组织列的顺序
    df_final = df_final[['Scene', 'Experiment', 'Max_Test_PSNR', 'Status']]

    print("\n" + "="*80)
    print("                      📊 选择性实验结果汇总表 (1/8 采样, 15K 迭代) 📊")
    print("="*80)
    print(df_final.to_string(index=False))
    print("="*80)
else:
    print("\n❌ 未找到任何结果进行汇总。")
EOF