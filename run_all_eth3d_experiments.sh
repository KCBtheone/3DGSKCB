#!/bin/bash

# ===================================================================================
#               ETH3D 全场景 Line Loss 对比实验流程 (指定保存点版)
# ===================================================================================
# 策略: 使用正确的数据集文件结构，强制为每个场景生成 lines.json，
#      并通过检查文件是否真实存在来验证成功，然后再进行对比实验。
#      仅保存第 7000 次和第 30000 次的检查点。
# ===================================================================================

# --- [ 1. 核心训练参数与路径 ] ---

PROJECT_DIR=$(pwd)
TRAIN_ITERATIONS=30000 
RESOLUTION_SCALE=4 # 训练时的降采样率

# 定义直线检测脚本的路径
HOUGH_SCRIPT="$PROJECT_DIR/detect_lines.py"

# 为全分辨率图像调整的霍夫直线检测参数
HOUGH_THRESHOLD=150
MIN_LINE_LENGTH=50
MAX_LINE_GAP=10

# 【核心修改】只定义第 7000 次和最后一次迭代为保存点
SAVE_CHECKPOINT_LIST=(7000 ${TRAIN_ITERATIONS})
ITERATION_ARGS="--checkpoint_iterations ${SAVE_CHECKPOINT_LIST[@]} --save_iterations ${SAVE_CHECKPOINT_LIST[@]}"
BASE_TRAIN_ARGS="--iterations ${TRAIN_ITERATIONS} -r ${RESOLUTION_SCALE} ${ITERATION_ARGS}"

# --- [ 2. 实验配置 ] ---

# 所有 ETH3D 训练场景列表
DATASETS=(
    "courtyard" "delivery_area" "electro" "facade" "kicker" "meadow" "office" 
    "pipes" "playground" "relief" "relief_2" "terrace" "terrains"
)

# Line Loss 参数列表 (基线测试在最后)
LAMBDA_LINES=("0.1" "1.0" "0.0")

# 最终输出的根目录
GLOBAL_OUTPUT_ROOT="./ETH3D_LINE_EXPERIMENTS_$(date +%Y%m%d_%H%M)"
mkdir -p "$GLOBAL_OUTPUT_ROOT"

# --- [ 3. 循环运行实验 ] ---

echo "======================================================================"
echo ">>>> 开始大规模实验，结果将保存至: $GLOBAL_OUTPUT_ROOT"
echo "======================================================================"

for DATASET_NAME in "${DATASETS[@]}"; do
    DATA_PATH="./data/${DATASET_NAME}"
    LINES_FILE="${DATA_PATH}/lines.json"
    
    # 定义数据集中图像文件夹的正确相对路径
    IMAGE_DIR_RELATIVE_PATH="images/dslr_images_undistorted"
    FULL_IMAGE_PATH="${DATA_PATH}/${IMAGE_DIR_RELATIVE_PATH}"

    # --- 前置检查 ---
    if [ ! -d "$DATA_PATH" ]; then
        echo "⚠️ 警告: 数据集 $DATASET_NAME 目录不存在，跳过。"
        continue
    fi
    if [ ! -d "${FULL_IMAGE_PATH}" ]; then
        echo "⚠️ 警告: 在 ${DATA_PATH} 中找不到图像目录 '${IMAGE_DIR_RELATIVE_PATH}'，跳过场景 ${DATASET_NAME}。"
        continue
    fi

    # --- [ 强制生成并验证 lines.json ] ---
    echo "--------------------------------------------------------------------"
    echo ">>> 正在为场景 [${DATASET_NAME}] 强制生成直线数据..."
    
    # 为了确保验证的可靠性，先删除旧的 lines.json (如果存在)
    rm -f "$LINES_FILE"

    # 调用直线检测脚本，并传入正确的图像目录相对路径
    python "$HOUGH_SCRIPT" \
        --dataset_path "$DATA_PATH" \
        --image_dir "$IMAGE_DIR_RELATIVE_PATH" \
        --hough_threshold $HOUGH_THRESHOLD \
        --min_length $MIN_LINE_LENGTH \
        --max_gap $MAX_LINE_GAP
    
    # 直接检查文件是否被成功创建，这是最可靠的验证方法
    if [ -f "$LINES_FILE" ]; then
        echo "✅ 成功为 [${DATASET_NAME}] 生成了 lines.json。"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "❌ 严重错误: 为场景 ${DATASET_NAME} 生成 lines.json 失败! 文件未被创建。将跳过此场景的所有实验。"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        continue # 跳到下一个场景
    fi
    
    # --- [ 训练步骤 ] ---
    for LAMBDA_VAL in "${LAMBDA_LINES[@]}"; do
        
        # 构造实验名
        if [ "$LAMBDA_VAL" == "0.0" ]; then
            EXP_NAME="BASELINE"
        elif [ "$LAMBDA_VAL" == "0.1" ]; then
            EXP_NAME="LINES_LOW"
        else
            EXP_NAME="LINES_HIGH"
        fi

        # 构造最终输出目录
        MODEL_OUTPUT_DIR="${GLOBAL_OUTPUT_ROOT}/${DATASET_NAME}_${EXP_NAME}_L${LAMBDA_VAL}_R${RESOLUTION_SCALE}"
        
        echo "--------------------------------------------------------------------"
        echo ">>> 启动场景: ${DATASET_NAME} | 模式: ${EXP_NAME} (lambda=${LAMBDA_VAL})"
        echo ">>> 输出至: ${MODEL_OUTPUT_DIR}"
        echo "--------------------------------------------------------------------"

        # 运行训练命令
        # train.py 的 -s 参数指向场景根目录，其内部加载器会自动处理子目录结构
        python train.py \
            -s "$DATA_PATH" \
            -m "$MODEL_OUTPUT_DIR" \
            --lambda_line "$LAMBDA_VAL" \
            ${BASE_TRAIN_ARGS}
        
        # 检查训练是否成功
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "❌ 错误: ${DATASET_NAME} (${EXP_NAME}) 训练过程中发生错误! 正在继续下一个实验..."
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        else
            echo "✅ 实验完成: ${DATASET_NAME} (${EXP_NAME})"
        fi
        
        # 插入短暂的暂停，有助于GPU资源释放
        sleep 2
    done
done

echo "======================================================================"
echo "🎉 所有实验批处理已完成。"
echo "======================================================================"