import numpy as np
import os

def read_images_txt(path):
    """
    从 images.txt 文件中读取相机位姿和图像名称。
    """
    images = {}
    with open(path, "r") as f:
        # 跳过文件头的注释
        for i in range(4):
            line = f.readline()
            if i == 3: # Number of images: 38...
                assert line.startswith("# Number of images:"), "Invalid images.txt file format"

        # 逐行读取图像数据
        while True:
            line = f.readline()
            if not line:
                break
            
            # 读取外参行
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            
            # 简单地将位置存储起来
            images[image_id] = {'name': name, 'tvec': np.array([tx, ty, tz])}
            
            # 跳过下一行的 2D 点数据
            f.readline()
            
    return images

def main():
    # --- 请将此路径修改为您 images.txt 文件的实际路径 ---
    images_txt_path = "/root/autodl-tmp/gaussian-splatting/data/courtyard/sparse/0/images.txt"
    # ----------------------------------------------------

    if not os.path.exists(images_txt_path):
        print(f"❌ 错误: 文件未找到 at '{images_txt_path}'")
        return

    print(f"🔍 正在读取和分析: {images_txt_path}\n")

    # 1. 读取所有相机数据
    all_images = read_images_txt(images_txt_path)
    
    # 将字典转换为按 image_id 排序的列表，以模拟文件中的顺序
    sorted_images = sorted(all_images.items(), key=lambda item: item[0])
    
    # 2. 根据 "每8个选1个" 的规则划分训练集和测试集
    train_cameras = []
    test_cameras = []
    
    print("--- 训练/测试集划分结果 ---")
    for i, (image_id, data) in enumerate(sorted_images):
        if (i % 8 == 0): # COLMAP/LLFF 默认划分规则
            test_cameras.append(data)
            print(f"[测试集] Image #{i+1}: {data['name']}")
        else:
            train_cameras.append(data)
            # print(f"[训练集] Image #{i+1}: {data['name']}")

    # 3. 计算训练相机的几何中心
    train_positions = np.array([cam['tvec'] for cam in train_cameras])
    train_centroid = np.mean(train_positions, axis=0)
    
    print("\n--- 相机位置分析 ---")
    print(f"训练相机几何中心 (平均位置): {np.round(train_centroid, 2)}")

    # 4. 计算每个测试相机到训练中心的距离
    print("\n[关键] 测试相机与训练中心距离:")
    total_distance = 0
    for cam in test_cameras:
        distance = np.linalg.norm(cam['tvec'] - train_centroid)
        total_distance += distance
        print(f"  - {cam['name']}: 位置 {np.round(cam['tvec'], 2)}, 距离中心: {distance:.2f} 米")
        
    avg_distance = total_distance / len(test_cameras)
    print(f"\n🔥 平均距离: {avg_distance:.2f} 米")

    print("\n--- 诊断 ---")
    if avg_distance > 5.0: # 这个阈值可以调整，但对于一般场景，大于5米通常意味着偏离
        print("✅ 结论: 问题确认！测试相机的位置显著偏离了训练相机的主体区域。")
        print("这解释了为什么渲染结果是模糊的'浓雾'并且所有实验的PSNR都一样。")
    else:
        print("🤔 结论: 测试相机距离中心不算太远，问题可能更复杂，但自动划分仍是最大嫌疑。")


if __name__ == "__main__":
    main()