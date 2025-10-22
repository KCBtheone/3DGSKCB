#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
# ============================================================================== #
#          >>> [ 最终修复: 从 dataset_readers 导入字典 ] <<<                     #
# ============================================================================== #
from scene.dataset_readers import sceneLoadTypeCallbacks 
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np


class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = []
        self.test_cameras = []

        # [修复后] 现在可以安全地使用导入的 sceneLoadTypeCallbacks 字典
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, f"Could not recognize scene type in {args.source_path}"

        if not self.loaded_iter:
            if os.path.exists(scene_info.ply_path):
                 with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())
            
            json_cams = []
            camlist = scene_info.train_cameras + scene_info.test_cameras
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        # cameraList_from_camInfos 的调用逻辑保持不变
        print(f"Loading {len(scene_info.train_cameras)} Training Cameras...")
        self.train_cameras = cameraList_from_camInfos(
            cam_infos=scene_info.train_cameras, 
            resolution_scale=1.0,  # 通常在 dataset_reader 中处理，这里设为 1
            args=args
        )
        
        print(f"Loading {len(scene_info.test_cameras)} Test Cameras...")
        self.test_cameras = cameraList_from_camInfos(
            cam_infos=scene_info.test_cameras, 
            resolution_scale=1.0, 
            args=args
        )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras