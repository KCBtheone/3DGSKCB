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
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        """
        Loads a scene from the input data.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # 自动识别场景类型 (Colmap vs Blender/NeRF)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender (NeRF) data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            raise Exception("Could not recognize scene type!")

        if not self.loaded_iter:
            # For camera visualization
            with open(os.path.join(args.model_path, "cameras.json"), 'w') as file:
                json.dump([], file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 从 CameraInfo 列表创建完整的 Camera 对象
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args.resolution, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args.resolution, args)

        # 根据情况加载或创建高斯点云
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        # 保存相机参数供查看器使用
        if not self.loaded_iter:
            json_cams = []
            for cam in self.train_cameras:
                json_cams.append(camera_to_JSON(cam.uid, cam))
            for cam in self.test_cameras:
                json_cams.append(camera_to_JSON(cam.uid, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

    # ‼️‼️ [核心修改] ‼️‼️
    # 添加 is_best 参数以支持保存最佳模型
    def save(self, iteration, is_best=False):
        """保存当前的高斯模型到.ply文件"""
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud", "best")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud", f"iteration_{iteration}")
        
        print(f"Saving Gaussian model to {point_cloud_path}")
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras