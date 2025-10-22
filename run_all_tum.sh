#!/bin/bash

# ===================================================================================
#      3DGS 法线权重 - 全数据集自动化基准测试脚本 (v22.0 - 集成重建版)
# ===================================================================================
# 变更:
# - [v22.0] 整合了 v21.0 的 COLMAP 稳健重建功能。
# -         现在脚本会对每个场景首先执行一次完整的稀疏重建流程，
#           以解决原始数据中 point3D 为空的问题。
# -         调整了主控流程：1. 重建 -> 2. 生成先验 -> 3. 运行实验。
# ===================================================================================

# --- [ 1. 全局配置区 ] ---
PROJECT_DIR=$(pwd)
DATA_ROOT_DIR="$PROJECT_DIR/data"
# 新的实验输出目录名称
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/NORMAL_EXPERIMENTS_WITH_RECON"

# --- 按照指定顺序排列的场景列表 (T&T -> NeRF -> DTU) ---
SCENE_NAMES=(
    # --- Tanks and Temples - Intermediate (8个) ---
    "tankandtemples/intermediate/Family"
    "tankandtemples/intermediate/Francis"
    "tankandtemples/intermediate/Horse"
    "tankandtemples/intermediate/Lighthouse"
    "tankandtemples/intermediate/M60"
    "tankandtemples/intermediate/Panther"
    "tankandtemples/intermediate/Playground"
    "tankandtemples/intermediate/Train"

    # --- Tanks and Temples - Advanced (6个) ---
    "tankandtemples/advanced/Auditorium"
    "tankandtemples/advanced/Ballroom"
    "tankandtemples/advanced/Courtroom"
    "tankandtemples/advanced/Museum"
    "tankandtemples/advanced/Palace"
    "tankandtemples/advanced/Temple"

    # --- NeRF Synthetic 场景 (8个) ---
    "nerf_synthetic/chair"     "nerf_synthetic/drums"    "nerf_synthetic/ficus"
    "nerf_synthetic/hotdog"    "nerf_synthetic/lego"     "nerf_synthetic/materials"
    "nerf_synthetic/mic"       "nerf_synthetic/ship"

    # --- DTU 场景 (22个) ---
    "dtu/scan1"   "dtu/scan4"   "dtu/scan9"   "dtu/scan10"  "dtu/scan11"
    "dtu/scan12"  "dtu/scan13"  "dtu/scan15"  "dtu/scan23"  "dtu/scan24"
    "dtu/scan29"  "dtu/scan32"  "dtu/scan33"  "dtu/scan34"  "dtu/scan48"
    "dtu/scan49"  "dtu/scan62"  "dtu/scan75"  "dtu/scan77"  "dtu/scan110"
    "dtu/scan114" "dtu/scan118"
)

DEFAULT_ITERATIONS=20000

# --- [ 2. 核心预处理函数 ] ---

force_reconstruct_sparse_cloud_robust() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local image_dir_name=""

    echo "--- [1/3] 开始稳健稀疏重建 [${scene_name}] ---"

    local single_camera_param="--ImageReader.single_camera 1"
    if [[ $scene_name == nerf_synthetic* ]]; then
        image_dir_name="train"
        single_camera_param="--ImageReader.single_camera_per_folder 1"
    elif [[ $scene_name == dtu* ]]; then image_dir_name="images"
    elif [[ $scene_name == tankandtemples* ]]; then image_dir_name="images"
    else echo "     -> ❌ 错误: 未知的场景类型!"; return 1; fi

    local image_path="$scene_path/$image_dir_name"
    local sparse_dir="$scene_path/sparse"
    local db_path="$scene_path/database.db"

    # 清理旧的重建结果以确保全新重建
    rm -f "$db_path"
    rm -rf "$sparse_dir"
    mkdir -p "$sparse_dir"
    local start_time=$(date +%s)

    echo "     -> [步骤 1/3] 特征提取..."
    xvfb-run colmap feature_extractor --database_path "$db_path" --image_path "$image_path" --ImageReader.camera_model PINHOLE $single_camera_param
    if [ $? -ne 0 ]; then echo "     -> ❌ 特征提取失败!"; return 1; fi

    echo "     -> [步骤 2/3] 特征匹配 (Exhaustive Matcher)..."
    xvfb-run colmap exhaustive_matcher --database_path "$db_path"
    if [ $? -ne 0 ]; then echo "     -> ❌ 特征匹配失败!"; return 1; fi

    echo "     -> [步骤 3/3] 稀疏重建/建图..."
    xvfb-run colmap mapper \
        --database_path "$db_path" \
        --image_path "$image_path" \
        --output_path "$sparse_dir" \
        --Mapper.tri_min_angle 1.0 \
        --Mapper.tri_ignore_two_view_tracks 0
    if [ $? -ne 0 ]; then echo "     -> ❌ 稀疏重建/建图失败!"; return 1; fi

    # COLMAP mapper 可能会在 sparse/0, sparse/1 ... 创建模型, 我们需要统一
    # 通常最好的模型是 sparse/0
    if [ -d "$sparse_dir/0" ]; then
        echo "     -> 检测到子模型目录, 将 sparse/0 的内容移动到 sparse/"
        # 将所有文件从 sparse/0 移动到 sparse，然后删除空的 sparse/0
        mv "$sparse_dir"/0/* "$sparse_dir"/
        rmdir "$sparse_dir"/0
    fi
    
    # 检查重建是否真的成功（生成了关键文件）
    if ! [ -f "$sparse_dir/points3D.bin" ] && ! [ -f "$sparse_dir/points3D.txt" ]; then
        echo "     -> ❌ 重建后未找到 points3D 文件，重建可能已失败！"
        return 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo "     -> ✅ COLMAP 重建成功！ (用时: ${duration} 秒)"
    return 0
}

generate_geometry_priors() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local priors_dir="$scene_path/geometry_priors"

    echo "--- [2/3] 开始生成几何先验 [${scene_name}] ---"
    if [ -d "${priors_dir}" ]; then echo "     -> ✅ 几何先验目录 '${priors_dir}' 已存在，跳过生成。"; return 0; fi
    echo "     -> 正在运行 generate_colmap_priors.py..."
    python generate_colmap_priors.py "${scene_path}"
    if [ $? -eq 0 ]; then echo "     -> ✅ 几何先验生成成功。"; return 0; else echo "     -> ❌ 错误: 几何先验生成失败！"; return 1; fi
}

# --- [ 3. 核心执行函数 ] ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [${exp_name}] for scene [${scene_name}] ---"
    if [ -d "${model_path}" ]; then echo "       -> 结果已存在，跳过。"; return; fi
    echo "       -> 输出至: ${model_path}"
    timeout 6h python train.py "${python_args[@]}" --model_path "${model_path}"
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        echo "       -> ✅ 成功完成。"
    elif [ ${exit_code} -eq 124 ]; then
        echo "       -> ❌ 超时！实验运行超过6小时。"
        touch "${model_path}_TIMED_OUT.log"
    else
        echo "       -> ❌ 失败！Python 脚本以错误码 ${exit_code} 退出。"
        touch "${model_path}_FAILED.log"
    fi
}

# --- [ 4. 场景实验总控 ] ---
run_scene_experiments() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local scene_name_safe=${scene_name//\//_} # 将 "a/b" 转换为 "a_b"
    local scene_output_dir="$EXPERIMENTS_ROOT_DIR/$scene_name_safe"
    mkdir -p "$scene_output_dir"

    echo; echo "######################################################################"
    echo "###    开始处理场景: [${scene_name}]"
    echo "######################################################################"

    # --- 步骤 1: 强制执行一次全新的 COLMAP 稀疏重建 ---
    force_reconstruct_sparse_cloud_robust "$scene_name"
    if [ $? -ne 0 ]; then echo "### ❌ 场景 [${scene_name}] COLMAP 重建失败，跳过所有实验。 ###"; return; fi
    
    # --- 步骤 2: 基于新生成的模型，生成几何先验 ---
    generate_geometry_priors "$scene_name"
    if [ $? -ne 0 ]; then echo "### ❌ 场景 [${scene_name}] 几何先验生成失败，跳过所有实验。 ###"; return; fi
    
    echo "--- [3/3] 开始运行9组对比实验 [${scene_name}] ---"
    
    # --- 步骤 3: 数据集适配逻辑 ---
    local image_subdir=""
    local iterations=$DEFAULT_ITERATIONS
    local resolution_scale=4

    if [[ $scene_name == tankandtemples* ]]; then
        echo "     -> 检测到 Tanks and Temples 场景。降采样因子: 8"
        image_subdir="images"
        resolution_scale=8
    elif [[ $scene_name == nerf_synthetic* ]]; then
        echo "     -> 检测到 NeRF Synthetic 场景。降采样因子: 2"
        image_subdir="train" 
        resolution_scale=2
    elif [[ $scene_name == dtu* ]]; then
        echo "     -> 检测到 DTU 场景。降采样因子: 4"
        image_subdir="images"
        resolution_scale=4
    fi
    
    local common_args=(
        -s "${scene_path}" --images "${image_subdir}" --resolution "${resolution_scale}"
        --iterations "${iterations}" --save_iterations "${iterations}" --test_iterations "${iterations}" --eval
    )
    local geo_start_default=7000

    # --- 步骤 4: 实验组 ---
    run_single_experiment "$scene_name_safe" "1/9: Baseline" "${scene_output_dir}/exp1_base" "${common_args[@]}" --geometry_constraint_type none
    run_single_experiment "$scene_name_safe" "2/9: Depth Only" "${scene_output_dir}/exp2_depth_only" "${common_args[@]}" --geometry_constraint_type depth --lambda_depth 0.001 --geometry_start_iter 7000
    local alpha_weak=0.05; run_single_experiment "$scene_name_safe" "3/9: Normal (Weak, α=${alpha_weak})" "${scene_output_dir}/exp3_normal_a${alpha_weak//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_weak}" --geometry_start_iter "${geo_start_default}"
    local alpha_medium=0.10; run_single_experiment "$scene_name_safe" "4/9: Normal (Medium, α=${alpha_medium})" "${scene_output_dir}/exp4_normal_a${alpha_medium//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_medium}" --geometry_start_iter "${geo_start_default}"
    local alpha_strong=0.20; run_single_experiment "$scene_name_safe" "5/9: Normal (Strong, α=${alpha_strong})" "${scene_output_dir}/exp5_normal_a${alpha_strong//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_strong}" --geometry_start_iter "${geo_start_default}"
    local alpha_higher1=0.30; run_single_experiment "$scene_name_safe" "6/9: Normal (Higher, α=${alpha_higher1})" "${scene_output_dir}/exp6_normal_a${alpha_higher1//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_higher1}" --geometry_start_iter "${geo_start_default}"
    local alpha_higher2=0.40; run_single_experiment "$scene_name_safe" "7/9: Normal (Max, α=${alpha_higher2})" "${scene_output_dir}/exp7_normal_a${alpha_higher2//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_higher2}" --geometry_start_iter "${geo_start_default}"
    local alpha_default=0.10; local geo_start_late=12000; run_single_experiment "$scene_name_safe" "8/9: Normal (Late, iter=${geo_start_late})" "${scene_output_dir}/exp8_normal_late${geo_start_late}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_default}" --geometry_start_iter "${geo_start_late}"
    local geo_start_early=3000; run_single_experiment "$scene_name_safe" "9/9: Normal (Early, iter=${geo_start_early})" "${scene_output_dir}/exp9_normal_early${geo_start_early}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_default}" --geometry_start_iter "${geo_start_early}"

    echo "✅ 场景 [${scene_name}] 的所有流程已尝试运行。"
}

# --- [ 5. 主执行循环 ] ---
echo "🚀🚀🚀 开始全数据集重建与法线权重基准测试 (v22.0) (共 ${#SCENE_NAMES[@]} 个场景) 🚀🚀🚀"
cd "$PROJECT_DIR"
mkdir -p "$EXPERIMENTS_ROOT_DIR"

for scene in "${SCENE_NAMES[@]}"; do
    run_scene_experiments "$scene"
done

echo; echo "# ======================================================================"
echo "# 🎉🎉🎉 所有场景的基准测试流程执行完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "所有结果已保存至 '$EXPERIMENTS_ROOT_DIR' 文件夹。"