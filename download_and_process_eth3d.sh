#!/bin/bash

# =====================================================================
# 扩展版: ETH3D 多个 High-Res 场景下载、解压和格式化脚本
# =====================================================================

# 基础信息
ETH3D_BASE_URL="http://www.eth3d.net/data"
TARGET_DIR="./data"
PROJECT_ROOT=$(pwd)

# 核心修复: 确保直连下载
unset http_proxy
unset https_proxy

# 所有 ETH3D High-res multi-view 训练场景列表 (仅包含 dslr_undistorted 版本)
DATASETS=(
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "kicker"
    "meadow"
    "office"
    "pipes" # 包含你之前下载的场景
    "playground"
    "relief"
    "relief_2"
    "terrace"
    "terrains"
)

# 1. 检查和创建目标目录
mkdir -p "$TARGET_DIR"

# 1. 检查并安装 7zip
echo "检查并安装 7zip..."
if ! command -v 7z &> /dev/null; then
    echo "⚠️ 警告: 7zip 工具未安装。正在尝试安装..."
    apt-get update && apt-get install -y p7zip-full
fi
echo "✅ 7zip 检查完成。"
echo ""

# 2. 循环处理所有数据集
for DATASET_NAME in "${DATASETS[@]}"; do
    FILE_NAME="${DATASET_NAME}_dslr_undistorted.7z"
    DOWNLOAD_URL="${ETH3D_BASE_URL}/${FILE_NAME}"
    FINAL_SCENE_DIR="$TARGET_DIR/${DATASET_NAME}"
    TEMP_7Z="$TARGET_DIR/${FILE_NAME}"
    TEMP_EXTRACT_DIR="$TARGET_DIR/${DATASET_NAME}-temp"

    echo "======================================================================"
    echo ">>>> 正在处理数据集: ${DATASET_NAME}"
    echo "======================================================================"
    
    # --- 2.1 跳过已存在的完整目录 ---
    if [ -d "$FINAL_SCENE_DIR/images" ] && [ -d "$FINAL_SCENE_DIR/sparse/0" ]; then
        echo "✅ 目录已存在且完整，跳过下载和处理。"
        continue
    fi

    # --- 2.2 下载文件 ---
    if [ ! -f "$TEMP_7Z" ]; then
        echo "🚀 开始下载文件..."
        wget -O "$TEMP_7Z" "$DOWNLOAD_URL"
        if [ $? -ne 0 ]; then
            echo "❌ 错误: 下载失败。URL可能已失效: $DOWNLOAD_URL" >&2
            rm -f "$TEMP_7Z" # 清理失败文件
            continue
        fi
        echo "✅ 下载完成。"
    else
        echo "✅ 压缩包 $TEMP_7Z 已存在，跳过下载。"
    fi

    # --- 2.3 解压、整理和格式化文件 ---
    echo "开始解压和整理文件结构..."
    mkdir -p "$TEMP_EXTRACT_DIR"
    
    # 7z 解压到临时目录
    7z x "$TEMP_7Z" -o"$TEMP_EXTRACT_DIR"
    
    # 清理旧的最终目录（防止残留）
    rm -rf "$FINAL_SCENE_DIR"
    mkdir -p "$FINAL_SCENE_DIR/images" "$FINAL_SCENE_DIR/sparse/0" 

    # 查找解压后的根目录 (通常是 datasets/scene_name/)
    # 我们知道解压后会有 images 和 dslr_calibration_undistorted 两个关键目录
    # 查找 images 目录的通用路径
    IMG_BASE_DIR=$(find "$TEMP_EXTRACT_DIR" -type d -name "images" -print -quit)
    if [ -z "$IMG_BASE_DIR" ]; then
        echo "❌ 错误: 找不到 images 目录。文件结构不匹配，已跳过。" >&2
        rm -rf "$TEMP_EXTRACT_DIR"
        continue
    fi

    # 1. 移动图像文件
    # 图像文件通常在 images/dslr_images_undistorted/
    IMG_SOURCE_DIR=$(find "$IMG_BASE_DIR" -type d -name "dslr_images_undistorted" -print -quit)
    if [ -d "$IMG_SOURCE_DIR" ]; then
        cp -r "$IMG_SOURCE_DIR/"* "$FINAL_SCENE_DIR/images/"
    else
        # Fallback: 如果没有子目录，就复制 images 目录下的全部内容
        cp -r "$IMG_BASE_DIR/"* "$FINAL_SCENE_DIR/images/"
    fi

    # 2. 移动 COLMAP 稀疏结果到 sparse/0
    # COLMAP 文件通常在 dslr_calibration_undistorted/
    CALIB_SOURCE_DIR=$(find "$TEMP_EXTRACT_DIR" -type d -name "dslr_calibration_undistorted" -print -quit)
    if [ -f "$CALIB_SOURCE_DIR/cameras.txt" ]; then
        cp "$CALIB_SOURCE_DIR/cameras.txt" "$FINAL_SCENE_DIR/sparse/0/"
        cp "$CALIB_SOURCE_DIR/images.txt" "$FINAL_SCENE_DIR/sparse/0/"
        cp "$CALIB_SOURCE_DIR/points3D.txt" "$FINAL_SCENE_DIR/sparse/0/"
        echo "✅ COLMAP 稀疏结果已成功整理。"
    else
        echo "❌ 错误: 找不到 COLMAP 稀疏文件。请手动检查下载的文件结构。" >&2
        rm -rf "$FINAL_SCENE_DIR" "$TEMP_EXTRACT_DIR"
        continue
    fi

    # 3. 清理临时文件
    rm -rf "$TEMP_EXTRACT_DIR"
    # 可选: rm "$TEMP_7Z" # 如果想保留压缩包，可以注释掉这行

done

echo "======================================================================"
echo "🎉 所有数据集下载和处理流程已完成。现在可以运行 Line Loss 实验。"
echo "======================================================================"

