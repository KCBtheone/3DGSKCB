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
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, OptimizationParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# --- [ 新增导入: UDF Grid Manager ] ---
from scene.udf_grid import UDFGrid 


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, opt: OptimizationParams, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scale = args.resolution

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # --- [ 核心新增: 加载 UDF 网格管理器 ] ---
        # UDFGrid 实例将管理网格数据和查询逻辑
        self.udf_grid_manager = UDFGrid(args.source_path, args.data_device)


        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif "blender" in args.source_path.lower() or os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "associations.txt")):
            scene_info = sceneLoadTypeCallbacks["TUM"](args.source_path, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        # [新增] 提前加载直线数据
        lines = None
        if opt.lambda_line > 0 or opt.lambda_dynamic_weight > 0 or hasattr(opt, 'lambda_static_weight'): # 适应新旧参数
            lines_file = os.path.join(args.source_path, "lines.json")
            if os.path.exists(lines_file):
                print("Found lines.json, loading line data...")
                with open(lines_file, 'r') as f:
                    lines = json.load(f)
            else:
                # 即使没找到 lines.json，也允许加载，因为可能只用 UDF 约束
                print(f"[Warning] Lines.json not found in {args.source_path}. Line-based constraints might be inactive.")


        # [修正] 将 lines 传递给 cameraList_from_camInfos
        resolution_scales = [self.resolution_scale]
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras,
                resolution_scale,
                args,
                lines=lines
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras,
                resolution_scale,
                args,
                lines=lines
            )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scale]

    def getTestCameras(self):
        return self.test_cameras[self.resolution_scale]