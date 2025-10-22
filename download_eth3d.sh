#!/bin/bash

# =====================================================================
# ETH3D 'pipes' 最小训练集下载和准备脚本
# 文件: pipes_dslr_undistorted.7z (约 0.1 GB)
# =====================================================================

DATASET_NAME="pipes"
FILE_NAME="${DATASET_NAME}_dslr_undistorted.7z"
DOWNLOAD_URL="http://www.eth3d.net/data/${FILE_NAME}"
PROJECT_ROOT=$(cd .. && pwd) # 获取项目根目录的绝对路径
TARGET_DIR="./data"
TEMP_7Z="${FILE_NAME}"

# 核心修复: 确保直连下载
unset http_proxy
unset https_proxy

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "🚀 开始下载 ETH3D - ${DATASET_NAME} (0.1 GB)..."
echo "URL: $DOWNLOAD_URL"

# 1. 检查并安装 7zip
if ! command -v 7z &> /dev/null; then
    echo "⚠️ 警告: 7zip 工具未安装。正在尝试安装..."
    apt-get update && apt-get install -y p7zip-full
fi

# 2. 下载文件
wget -O "$TEMP_7Z" "$DOWNLOAD_URL"
if [ $? -ne 0 ]; then
    echo "❌ 错误: 下载失败。请检查您的网络连接。" >&2
    exit 1
fi

echo "✅ 下载完成。开始解压和整理文件..."

# 3. 解压文件
# 解压后会得到一个名为 pipes/ 的目录 (或者直接解压到当前目录)
7z x "$TEMP_7Z"
rm "$TEMP_7Z"

# 4. 移动文件到标准 COLMAP 结构
FINAL_SCENE_DIR="./${DATASET_NAME}"

# --- 检查解压结果并重命名 ---
if [ -d "$DATASET_NAME" ]; then
    mv "$DATASET_NAME" "$FINAL_SCENE_DIR-temp"
elif [ -d "pipes/images" ]; then # 某些解压软件可能直接解压到当前目录
    mv images "$FINAL_SCENE_DIR-temp/images"
    mv dslr_calibration_undistorted "$FINAL_SCENE_DIR-temp/dslr_calibration_undistorted"
    mv . "$FINAL_SCENE_DIR-temp"
else
    # 假设文件已经解压到当前目录
    if [ ! -d "images" ] || [ ! -d "dslr_calibration_undistorted" ]; then
        echo "❌ 错误: 找不到预期的解压目录。请手动检查解压结果。" >&2
        exit 1
    fi
    # 临时创建 temp 目录以便后续处理
    mkdir -p "$FINAL_SCENE_DIR-temp"
    mv images "$FINAL_SCENE_DIR-temp/"
    mv dslr_calibration_undistorted "$FINAL_SCENE_DIR-temp/"
fi

# 创建最终目录
mkdir -p "$FINAL_SCENE_DIR/images" "$FINAL_SCENE_DIR/sparse/0" 

# 1. 移动图像文件
# 图像文件位于: pipes-temp/images/dslr_images_undistorted/
IMG_SOURCE_DIR="$FINAL_SCENE_DIR-temp/images/dslr_images_undistorted"
if [ -d "$IMG_SOURCE_DIR" ]; then
    cp -r "$IMG_SOURCE_DIR/"* "$FINAL_SCENE_DIR/images/"
else
    echo "❌ 错误: 找不到图像源目录 '$IMG_SOURCE_DIR'" >&2
    exit 1
fi

# 2. 移动 COLMAP 稀疏结果到 sparse/0
# COLMAP 文件位于: pipes-temp/dslr_calibration_undistorted/
COLMAP_SOURCE_DIR="$FINAL_SCENE_DIR-temp/dslr_calibration_undistorted"
if [ -f "$COLMAP_SOURCE_DIR/cameras.txt" ]; then
    # 只需要复制核心的 .txt 文件
    cp "$COLMAP_SOURCE_DIR/cameras.txt" "$FINAL_SCENE_DIR/sparse/0/"
    cp "$COLMAP_SOURCE_DIR/images.txt" "$FINAL_SCENE_DIR/sparse/0/"
    cp "$COLMAP_SOURCE_DIR/points3D.txt" "$FINAL_SCENE_DIR/sparse/0/"
    echo "✅ COLMAP 稀疏结果已成功整理到 $FINAL_SCENE_DIR/sparse/0/"
else
    echo "❌ 错误: 找不到 COLMAP 稀疏文件 cameras.txt。结构不匹配。" >&2
    exit 1
fi

# 5. 清理临时文件
rm -rf "$FINAL_SCENE_DIR-temp"

echo "🎉 数据集准备完成! 场景路径为: $PROJECT_ROOT/data/$DATASET_NAME"
echo "---"

# 返回项目根目录
cd "$PROJECT_ROOT"