#!/bin/bash

# ===================================================================================
#      使用 COLMAP MVS 批量重建稠密点云 (V4 - 终极精准修复版)
# ===================================================================================
#
# 目的: 1. (精准修复) 只对 image_undistorter 使用 xvfb-run，解决其崩溃问题。
#       2. (精准修复) 让 patch_match_stereo 和 stereo_fusion 直接运行，
#          确保它们能正确检测并使用系统的 CUDA 环境。
#
# ===================================================================================

set -e # 遇到任何错误立即停止脚本

PROJECT_DIR=$(pwd)
DATA_ROOT_DIR="$PROJECT_DIR/data"

# --- [ 核心配置：只运行这两个场景 ] ---
SCENE_NAMES=(
    "delivery_area"
    "meadow"
)

echo "🚀🚀🚀 开始为指定场景 [${SCENE_NAMES[*]}] 运行 COLMAP MVS (终极精准修复版)... 🚀🚀🚀"

for scene in "${SCENE_NAMES[@]}"; do
    echo
    echo "# ======================================================================"
    echo "# 场景: [ ${scene} ]"
    echo "# ======================================================================"

    SCENE_PATH="$DATA_ROOT_DIR/$scene"
    DENSE_FOLDER="$SCENE_PATH/dense"

    if [ ! -d "$SCENE_PATH/sparse/0" ]; then
        echo "⚠️ 警告: 未找到稀疏重建结果 '$SCENE_PATH/sparse/0'，跳过此场景。"
        continue
    fi
    
    # 清理掉上次失败的运行所产生的半成品文件
    if [ -d "$DENSE_FOLDER" ]; then
        echo "ℹ️ 检测到上次失败的 'dense' 文件夹，正在清理..."
        rm -rf "$DENSE_FOLDER"
    fi
    
    mkdir -p "$DENSE_FOLDER"

    # --- 核心步骤: 分别使用最适合的环境运行 ---
    echo "STEP 1/3: 图像畸变校正 (在虚拟显示器中)..."
    # <-- 核心修正: 只有这一步需要 xvfb-run -->
    xvfb-run -a colmap image_undistorter \
        --image_path "$SCENE_PATH/images" \
        --input_path "$SCENE_PATH/sparse/0" \
        --output_path "$DENSE_FOLDER" \
        --output_type COLMAP

    echo "STEP 2/3: 立体匹配 (直接运行以使用 GPU)..."
    # <-- 关键修复: 将 --gpu_index 0 更改为 --PatchMatchStereo.gpu_index 0 -->
    colmap patch_match_stereo \
        --workspace_path "$DENSE_FOLDER" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.gpu_index 0 

    echo "STEP 3/3: 点云融合 (直接运行)..."
    # <-- 核心修正: 移除 xvfb-run -->
    colmap stereo_fusion \
        --workspace_path "$DENSE_FOLDER" \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path "$DENSE_FOLDER/fused.ply"

    echo "✅ [${scene}] 的稠密点云生成完毕: $DENSE_FOLDER/fused.ply"

done

echo
echo "# ======================================================================"
echo "# 🎉🎉🎉 指定场景的稠密点云已生成！ 🎉🎉🎉"
echo "# ======================================================================"