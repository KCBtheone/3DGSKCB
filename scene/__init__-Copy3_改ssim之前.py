#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

try:
    import open3d as o3d
    OPEN3D_FOUND = True
except ImportError:
    OPEN3D_FOUND = False

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

        is_nerf_data = False
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender (NeRF) data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            is_nerf_data = True
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(os.path.join(args.source_path, "cameras.json"), 'w') as file:
                json.dump([], file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # =========================================================================================
        # >>> [ 🚀 终极修复 v3: 硬编码场景半径 ] <<<
        # 日志显示 self.cameras_extent (来自 scene_info) 为 0.0，这是所有问题的根源。
        # 我们在这里进行修正：如果检测到是 NeRF 数据且半径为0，就强制使用一个合理的默认值。
        # =========================================================================================
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        if is_nerf_data and self.cameras_extent < 1e-5:
            print(f"[WARNING] Detected zero radius for NeRF data. Overriding with a default radius of 1.3.")
            self.cameras_extent = 1.3
        # =========================================================================================

        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args.resolution, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args.resolution, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            if scene_info.point_cloud.points.shape[0] == 0:
                print("》》》 Detected an empty point cloud from data source.《《《")
                print(f"》》》 Initializing with 100,000 random points in a sphere of radius {self.cameras_extent:.4f}. 《《《")
                
                if not OPEN3D_FOUND:
                    raise ImportError("Open3D is required for random initialization. Please install it: pip install open3d")

                num_pts = 100_000 
                phi = np.random.uniform(0, 2 * np.pi, num_pts)
                costheta = np.random.uniform(-1, 1, num_pts)
                theta = np.arccos(costheta)
                u = np.random.uniform(0, 1, num_pts)
                r = self.cameras_extent * np.cbrt(u)
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                xyz = np.vstack((x, y, z)).T
                rgb = (np.random.rand(num_pts, 3) * 255).astype(np.uint8)
                
                from utils.graphics_utils import BasicPointCloud
                pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))
                self.gaussians.create_from_pcd(pcd, self.cameras_extent)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        if not self.loaded_iter:
            json_cams = []
            for cam in self.train_cameras:
                json_cams.append(camera_to_JSON(cam.uid, cam))
            for cam in self.test_cameras:
                json_cams.append(camera_to_JSON(cam.uid, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

    def save(self, iteration, is_best=False):
        if is_best:
             point_cloud_path = os.path.join(self.model_path, "point_cloud", "best")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud", f"iteration_{iteration}")
        
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras