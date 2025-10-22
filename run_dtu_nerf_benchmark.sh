#!/bin/bash

# ===================================================================================
#      3DGS 法线权重 - DTU & NeRF-Synthetic 自动化基准测试脚本 (v15.0)
# ===================================================================================
# 改进:
# - [v15.0] 场景列表已精简至只包含 DTU 和 NeRF Synthetic 数据集。
# - [v15.0] 修正了 robust_model_converter 函数中导致 'Aborted (core dumped)' 的
#           “逐个文件转换”逻辑，现在只使用可靠的文件夹转换模式。
# ===================================================================================

# --- [ 1. 全局配置区 ] ---
PROJECT_DIR=$(pwd)
# 假设数据（data）目录在项目根目录下
DATA_ROOT_DIR="$PROJECT_DIR/data"
# 新的实验输出目录名称
EXPERIMENTS_ROOT_DIR="$PROJECT_DIR/NORMAL_EXPERIMENTS_DTU_NERF"

# --- 已过滤的 DTU 和 NeRF Synthetic 数据集场景列表 ---
SCENE_NAMES=(
    # --- DTU 场景 (22个) ---
    "dtu/scan1"   "dtu/scan4"   "dtu/scan9"   "dtu/scan10"  "dtu/scan11"
    "dtu/scan12"  "dtu/scan13"  "dtu/scan15"  "dtu/scan23"  "dtu/scan24"
    "dtu/scan29"  "dtu/scan32"  "dtu/scan33"  "dtu/scan34"  "dtu/scan48"
    "dtu/scan49"  "dtu/scan62"  "dtu/scan75"  "dtu/scan77"  "dtu/scan110"
    "dtu/scan114" "dtu/scan118"

    # --- NeRF Synthetic 场景 (8个) ---
    "nerf_synthetic/chair"     "nerf_synthetic/drums"    "nerf_synthetic/ficus"
    "nerf_synthetic/hotdog"    "nerf_synthetic/lego"     "nerf_synthetic/materials"
    "nerf_synthetic/mic"       "nerf_synthetic/ship"
)

# 默认训练迭代次数
DEFAULT_ITERATIONS=20000

# --- [ 2. 预处理函数 (已修正 colmap 崩溃问题) ] ---
robust_model_converter() {
    local sparse_dir=$1
    echo "     -> ⚠️ 未找到二进制文件，正在尝试从文本(.txt)文件转换..."
    
    # 尝试命令 1: 标准的文件夹转换模式 (大写 BINARY)
    echo "        - 尝试命令 1: colmap model_converter (文件夹模式, 大写 BINARY)"
    colmap model_converter --input_path "${sparse_dir}" --output_path "${sparse_dir}" --output_type BINARY > /dev/null 2>&1
    if [ -f "${sparse_dir}/images.bin" ]; then echo "        -> ✅ 转换成功。"; return 0; fi

    # 尝试命令 2: 文件夹转换模式 (小写 binary) - 兼容某些老版本
    echo "        - 尝试命令 2: colmap model_converter (文件夹模式, 小写 binary)"
    colmap model_converter --input_path "${sparse_dir}" --output_path "${sparse_dir}" --output_type binary > /dev/null 2>&1
    if [ -f "${sparse_dir}/images.bin" ]; then echo "        -> ✅ 转换成功。"; return 0; fi
    
    # 【注意】已移除：导致 'Aborted (core dumped)' 的逐个文件转换逻辑。
    # model_converter 应该接收目录路径，而非单个文件路径。

    # 尝试命令 3: 明确指定输入类型 (文本转二进制)
    echo "        - 尝试命令 3: colmap model_converter (指定 TXT 到 BINARY)"
    colmap model_converter --input_path "${sparse_dir}" --output_path "${sparse_dir}" --input_type TXT --output_type BINARY > /dev/null 2>&1
    if [ -f "${sparse_dir}/images.bin" ]; then echo "        -> ✅ 转换成功。"; return 0; fi

    # 如果所有方法都失败了
    echo "        -> ❌ 错误: 所有 COLMAP 模型转换尝试均失败！"
    return 1
}

preprocess_colmap_format() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local colmap_sparse_dir="$scene_path/sparse/0"

    echo "--- [COLMAP格式检查] 正在处理场景 [${scene_name}] ---"
    if [ ! -d "$colmap_sparse_dir" ]; then 
        echo "     -> ❌ 错误: 找不到 COLMAP sparse/0 目录: ${colmap_sparse_dir}"
        echo "     -> 提示: 请确保已为该场景运行了数据转换脚本！"
        return 1
    fi
    
    if [ -f "$colmap_sparse_dir/images.bin" ]; then 
        echo "     -> ✅ 已存在二进制(.bin)文件，跳过转换。"
        return 0
    fi
    
    if [ ! -f "$colmap_sparse_dir/images.txt" ]; then 
        echo "     -> ❌ 错误: 在 ${colmap_sparse_dir} 中既未找到 .bin 文件也未找到 .txt 文件。"
        return 1
    fi

    robust_model_converter "$colmap_sparse_dir"
    if [ $? -ne 0 ]; then
        echo "     -> ❌ 错误: 所有 COLMAP 模型转换尝试均失败！请检查您的 COLMAP 安装和版本。"
        return 1
    fi
    return 0
}


# --- [ 3. 几何先验生成函数 (保持不变) ] ---
generate_geometry_priors() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local priors_dir="$scene_path/geometry_priors"

    echo "--- [几何先验生成] 正在处理场景 [${scene_name}] ---"
    if [ -d "${priors_dir}" ]; then echo "     -> ✅ 几何先验目录 '${priors_dir}' 已存在，跳过生成。"; return 0; fi
    echo "     -> 正在运行 generate_colmap_priors.py..."
    # 确保 generate_colmap_priors.py 脚本在您的 $PROJECT_DIR 或 PATH 中
    python generate_colmap_priors.py "${scene_path}"
    if [ $? -eq 0 ]; then echo "     -> ✅ 几何先验生成成功。"; return 0; else echo "     -> ❌ 错误: 几何先验生成失败！"; return 1; fi
}

# --- [ 4. 核心执行函数 (保持不变) ] ---
run_single_experiment() {
    local scene_name=$1; local exp_name=$2; local model_path=$3; shift 3; local python_args=("$@")
    echo; echo "--- [${exp_name}] for scene [${scene_name}] ---"
    if [ -d "${model_path}" ]; then echo "       -> 结果已存在，跳过。"; return; fi
    echo "       -> 输出至: ${model_path}"
    # 使用 timeout 限制运行时间，防止无限循环或僵死
    timeout 6h python train.py "${python_args[@]}" --model_path "${model_path}"
    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then echo "       -> ✅ 成功完成。";
    elif [ ${exit_code} -eq 124 ]; then echo "       -> ❌ 超时！实验运行超过6小时。"; touch "${model_path}_TIMED_OUT.log";
    else echo "       -> ❌ 失败！Python 脚本以错误码 ${exit_code} 退出。"; touch "${model_path}_FAILED.log"; fi
}

# --- [ 5. 场景实验总控 (已适配 DTU & NeRF-Synthetic) ] ---
run_scene_experiments() {
    local scene_name=$1
    local scene_path="$DATA_ROOT_DIR/$scene_name"
    local scene_name_safe=${scene_name//\//_} # 将 "dtu/scan1" 转换为 "dtu_scan1"
    local scene_output_dir="$EXPERIMENTS_ROOT_DIR/$scene_name_safe"
    mkdir -p "$scene_output_dir"

    echo; echo "######################################################################"
    echo "###    开始处理场景: [${scene_name}]"
    echo "######################################################################"

    # # --- 1. COLMAP 格式预处理 (修正后的转换逻辑) ---
    # preprocess_colmap_format "$scene_name"
    # if [ $? -ne 0 ]; then echo "### ❌ 场景 [${scene_name}] COLMAP 格式预处理失败，跳过所有实验。 ###"; return; fi

    # # --- 2. 几何先验生成 ---
    # generate_geometry_priors "$scene_name"
    # if [ $? -ne 0 ]; then echo "### ❌ 场景 [${scene_name}] 几何先验生成失败，跳过所有实验。 ###"; return; fi

    # --- 3. 数据集适配逻辑 ---
    local image_subdir=""
    local resolution_scale=4      # 默认值
    local iterations=$DEFAULT_ITERATIONS

    if [[ $scene_name == nerf_synthetic* ]]; then
        echo "     -> 检测到 NeRF Synthetic 场景。"
        image_subdir="train" # 训练图像在 'train' 子目录
        resolution_scale=1   # 800x800 -> 400x400 (通常为 2x downsampling)
    elif [[ $scene_name == dtu* ]]; then
        echo "     -> 检测到 DTU 场景。"
        image_subdir="images"
        resolution_scale=2   # 1600x1200 -> 400x300 (通常为 4x downsampling)
    else # 理论上不会执行，因为场景列表已被过滤
        echo "     -> ⚠️ 未知场景类型，使用默认配置。"
        image_subdir="images"
        resolution_scale=4
    fi
    echo "     -> 使用图像路径: '${image_subdir}', 分辨率缩放: ${resolution_scale}"
    # --- ----------------- ---

    # 定义所有实验共享的通用参数
    local common_args=(
        -s "${scene_path}" --images "${image_subdir}" --resolution "${resolution_scale}"
        --iterations "${iterations}" --save_iterations "${iterations}" --test_iterations "${iterations}" --eval
    )

    local geo_start_default=7000

    # --- 4. 实验组 ---
    run_single_experiment "$scene_name_safe" "1/9: Baseline" "${scene_output_dir}/exp1_base" "${common_args[@]}" --geometry_constraint_type none
    run_single_experiment "$scene_name_safe" "2/9: Depth Only" "${scene_output_dir}/exp2_depth_only" "${common_args[@]}" --geometry_constraint_type depth --lambda_depth 0.001 --geometry_start_iter 7000
    local alpha_weak=0.05; run_single_experiment "$scene_name_safe" "3/9: Normal (Weak, α=${alpha_weak})" "${scene_output_dir}/exp3_normal_a${alpha_weak//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_weak}" --geometry_start_iter "${geo_start_default}"
    local alpha_medium=0.10; run_single_experiment "$scene_name_safe" "4/9: Normal (Medium, α=${alpha_medium})" "${scene_output_dir}/exp4_normal_a${alpha_medium//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_medium}" --geometry_start_iter "${geo_start_default}"
    local alpha_strong=0.20; run_single_experiment "$scene_name_safe" "5/9: Normal (Strong, α=${alpha_strong})" "${scene_output_dir}/exp5_normal_a${alpha_strong//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_strong}" --geometry_start_iter "${geo_start_default}"
    local alpha_higher1=0.30; run_single_experiment "$scene_name_safe" "6/9: Normal (Higher, α=${alpha_higher1})" "${scene_output_dir}/exp6_normal_a${alpha_higher1//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_higher1}" --geometry_start_iter "${geo_start_default}"
    local alpha_higher2=0.40; run_single_experiment "$scene_name_safe" "7/9: Normal (Max, α=${alpha_higher2})" "${scene_output_dir}/exp7_normal_a${alpha_higher2//./p}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_higher2}" --geometry_start_iter "${geo_start_default}"
    local alpha_default=0.10; local geo_start_late=12000; run_single_experiment "$scene_name_safe" "8/9: Normal (Late, iter=${geo_start_late})" "${scene_output_dir}/exp8_normal_late${geo_start_late}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_default}" --geometry_start_iter "${geo_start_late}"
    local geo_start_early=3000; run_single_experiment "$scene_name_safe" "9/9: Normal (Early, iter=${geo_start_early})" "${scene_output_dir}/exp9_normal_early${geo_start_early}" "${common_args[@]}" --geometry_constraint_type normal --alpha_normals "${alpha_default}" --geometry_start_iter "${geo_start_early}"

    echo "✅ 场景 [${scene_name}] 的所有实验已尝试运行。"
}

# --- [ 6. 主执行循环 ] ---
echo "🚀🚀🚀 开始 DTU & NeRF-Synthetic 法线权重基准测试 (共 ${#SCENE_NAMES[@]} 个场景) 🚀🚀🚀"
# 确保在项目根目录执行 python train.py
cd "$PROJECT_DIR"

# 确保实验根目录存在
mkdir -p "$EXPERIMENTS_ROOT_DIR"

for scene in "${SCENE_NAMES[@]}"; do
    run_scene_experiments "$scene"
done

echo; echo "# ======================================================================"
echo "# 🎉🎉🎉 所有场景的基准测试流程执行完毕！ 🎉🎉🎉"
echo "# ======================================================================"
echo "所有结果已保存至 '$EXPERIMENTS_ROOT_DIR' 文件夹。"