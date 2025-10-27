#!/bin/bash
set -e # 如果任何命令失败，脚本将立即退出

# --- 1. 用户配置 ---
CODE_DIR="/root/autodl-tmp/gaussian-splatting" 
RAW_DATA_DIR="/root/autodl-tmp/gaussian-splatting/data/nerf_synthetic"
CONVERTED_DATA_DIR="/root/autodl-tmp/gaussian-splatting/data/nerf_synthetic_colmap"

# --- 2. 脚本正文 ---
echo "🚀 开始批量转换Blender数据集..."
cd "${CODE_DIR}"
mkdir -p "${CONVERTED_DATA_DIR}"

for scene_dir in "${RAW_DATA_DIR}"/*; do
    if [ -d "${scene_dir}" ]; then
        scene_name=$(basename "${scene_dir}")
        output_path="${CONVERTED_DATA_DIR}/${scene_name}"

        if [ -d "${output_path}" ]; then
            echo "✅ 场景 '${scene_name}' 已转换，跳过。"
        else
            echo "⏳ 正在转换场景: ${scene_name} ..."
            
            # 【*** 关键修正 ***】
            # 1. 使用 PYTHONPATH=. 保证模块导入
            # 2. 增加 --white_background 参数
            PYTHONPATH=. python convert.py -s "${scene_dir}" -m "${output_path}" --white_background
            
            echo "✅ 场景 '${scene_name}' 转换完成！"
        fi
    fi
done

echo "🎉🎉🎉 所有场景转换完毕！"