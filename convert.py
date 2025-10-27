
# convert.py (最终修复版 v2)
import argparse, os, shutil, json
from PIL import Image
from tqdm import tqdm
import numpy as np
from scene.dataset_readers import CameraInfo

# ... qvec2rotmat 函数保持不变 ...
def qvec2rotmat(qvec):
    return np.array([[1-2*qvec[2]**2-2*qvec[3]**2, 2*qvec[1]*qvec[2]-2*qvec[0]*qvec[3], 2*qvec[3]*qvec[1]+2*qvec[0]*qvec[2]],
                     [2*qvec[1]*qvec[2]+2*qvec[0]*qvec[3], 1-2*qvec[1]**2-2*qvec[3]**2, 2*qvec[2]*qvec[3]-2*qvec[0]*qvec[1]],
                     [2*qvec[3]*qvec[1]-2*qvec[0]*qvec[2], 2*qvec[2]*qvec[3]+2*qvec[0]*qvec[1], 1-2*qvec[1]**2-2*qvec[2]**2]])

def read_nerf_synthetic_cameras_from_json(path, white_background, extension=".png"):
    cam_infos = []
    # 【*** 关键升级 ***】 我们将同时返回一个split信息列表
    split_infos = [] 
    
    # 我们只处理 train 和 test，val通常不用
    for split in ["train", "test"]:
        json_path = os.path.join(path, f"transforms_{split}.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]
            
            for idx, frame in enumerate(tqdm(contents["frames"], desc=f"Reading {split} set")):
                # ... 内部逻辑保持不变 ...
                image_path = os.path.join(path, frame["file_path"] + extension)
                image_name = os.path.basename(image_path)
                pil_image = Image.open(image_path)
                width, height = pil_image.size
                fovy = 2 * np.arctan(height / (2 * width / (2 * np.tan(fovx / 2))))
                c2w = np.array(frame["transform_matrix"])
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3]
                T = w2c[:3, 3]
                
                # 创建CameraInfo对象
                cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                      image_path=image_path, image_name=image_name,
                                      width=width, height=height)
                cam_infos.append(cam_info)
                split_infos.append(split) # 记录这个相机属于哪个split
    
    return cam_infos, split_infos

def main(args):
    all_cameras, all_splits = read_nerf_synthetic_cameras_from_json(args.source_path, args.white_background)
    os.makedirs(args.model_path, exist_ok=True)
    
    cameras_json_data = []
    for id, (cam, split) in enumerate(zip(all_cameras, all_splits)):
        json_cam = { "id": id, "img_name": cam.image_name, "width": cam.width, "height": cam.height,
                     "position": cam.T.tolist(), "rotation": cam.R.tolist(),
                     "fx": cam.width / (2 * np.tan(cam.FovX / 2)), "fy": cam.height / (2 * np.tan(cam.FovY / 2)),
                     "split": split } # 【*** 关键升级 ***】 将split信息写入JSON
        if args.white_background:
            json_cam["white_background"] = True
        cameras_json_data.append(json_cam)
        
    with open(os.path.join(args.model_path, "cameras.json"), "w") as file:
        json.dump(cameras_json_data, file, indent=4)

    images_dir = os.path.join(args.model_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    for cam in tqdm(all_cameras, desc="Copying images"):
        shutil.copy(cam.image_path, os.path.join(images_dir, cam.image_name))

    print("\nData conversion complete.")

if __name__ == "__main__":
    # ... parser部分保持不变 ...
    parser = argparse.ArgumentParser("Blender dataset converter for 3DGS codebases.")
    parser.add_argument("-s", "--source_path", required=True, type=str)
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()
    main(args)