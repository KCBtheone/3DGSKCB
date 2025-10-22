# ==============================================================================
#      COLMAP 模型相机参数转换器 (v_final - 写回 BIN 格式)
#
# 功能:
#   读取 COLMAP 二进制模型，将 SIMPLE_RADIAL 相机模型转换为 PINHOLE，
#   并直接写回为二进制 (.bin) 格式，以覆盖旧文件。
# ==============================================================================

import os
import argparse
import collections
import struct
import numpy as np

# --- 数据结构定义 ---
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

# 我们只需要支持这几个模型
CAMERA_MODELS = {
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
}
CAMERA_MODEL_IDS = {cm.model_id: cm for cm in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {cm.model_name: cm for cm in CAMERA_MODELS}

# --- 二进制读写函数 ---
def read_next_bytes(fid, num_bytes, format_char, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char, data)

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = read_next_bytes(fid, 24, "iidd")
            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"错误: 遇到未知相机模型 ID: {model_id}")

            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = Camera(
                id=cam_id,
                model=CAMERA_MODEL_IDS[model_id].model_name,
                width=width,
                height=height,
                params=np.array(params)
            )
    return cameras

def write_cameras_binary(cameras, path_to_model_file):
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for _, cam in sorted(cameras.items()):
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            fid.write(struct.pack("<iidd", cam.id, model_id, cam.width, cam.height))
            for p in cam.params:
                fid.write(struct.pack("<d", p))

def main():
    parser = argparse.ArgumentParser(description="转换 COLMAP 相机模型并写回 BIN 文件")
    parser.add_argument("--input_file", required=True, help="原始 cameras.bin 文件路径")
    parser.add_argument("--output_file", required=True, help="保存修正后 cameras.bin 的文件路径")
    args = parser.parse_args()

    print("启动相机模型转换 (输出为 BIN)...")
    
    # 1. 读取二进制相机数据
    print(f"    - 正在读取: {args.input_file}")
    cameras = read_cameras_binary(args.input_file)

    # 2. 核心转换逻辑
    print(f"    - 找到了 {len(cameras)} 个相机。正在转换模型...")
    converted_count = 0
    for cam_id, cam in cameras.items():
        if cam.model == "SIMPLE_RADIAL":
            # SIMPLE_RADIAL 参数: f, cx, cy, k
            # PINHOLE 参数: fx, fy, cx, cy
            f, cx, cy, k = cam.params
            cameras[cam_id] = cam._replace(
                model="PINHOLE",
                params=np.array([f, f, cx, cy]) # 假设 fx=fy, 丢弃畸变参数 k
            )
            converted_count += 1
    
    print(f"    - ✅ 成功转换 {converted_count} / {len(cameras)} 个相机从 SIMPLE_RADIAL 到 PINHOLE。")
    
    # 3. 将修改后的数据写回到新的二进制文件
    print(f"    - 正在写出修正后的文件到: {args.output_file}")
    write_cameras_binary(cameras, args.output_file)
    
    print("转换完成！")

if __name__ == "__main__":
    main()