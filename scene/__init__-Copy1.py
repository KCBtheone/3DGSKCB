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
from arguments import ModelParams, OptimizationParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# --- Conditionally import UDFGrid to avoid errors if the file is missing ---
try:
    from scene.udf_manager import UDFGrid
    UDF_FOUND = True
except ImportError:
    UDF_FOUND = False

class Scene:
    """
    The Scene class is the central manager for all elements of a 3D scene.
    It handles loading the dataset (point cloud and cameras), initializing
    the GaussianModel, and managing the UDFGrid if applicable. It acts as
    a bridge between the raw data and the training process.
    """
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, opt: OptimizationParams, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        # --- UDFGrid Manager Initialization (Conditional) ---
        self.udf_grid_manager = None
        if opt.geometry_constraint_type == 'udf':
            if UDF_FOUND:
                print("[UDF] UDF constraint enabled. Initializing UDFGrid manager...")
                self.udf_grid_manager = UDFGrid(scene_path=args.source_path)
            else:
                print("[UDF] ⚠️ WARNING: UDF constraint selected, but UDFGrid class not found (scene/udf_manager.py).")
                print("[UDF] ⚠️ UDF constraint will be disabled.")

        # --- Load a pre-trained model if specified ---
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = []
        self.test_cameras = []

        # --- Determine Scene Type and Load Data ---
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, f"Could not recognize scene type in {args.source_path}"

        # --- Save initial point cloud and cameras if not loading a model ---
        if not self.loaded_iter:
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

        # --- Line Data Loading (Conditional) ---
        lines = None
        if opt.geometry_constraint_type == 'line':
            lines_file = os.path.join(args.source_path, "lines.json")
            if os.path.exists(lines_file):
                print("[Line] Line constraint enabled. Loading line data from lines.json...")
                with open(lines_file, 'r') as f:
                    lines = json.load(f)
            else:
                print(f"[Line] ⚠️ WARNING: Line constraint selected, but lines.json not found in {args.source_path}.")
                print("[Line] ⚠️ Line constraint will be disabled. Please generate lines.json first.")
        
        # --- Create Camera Objects ---
        print(f"Loading {len(scene_info.train_cameras)} Training Cameras...")
        self.train_cameras = cameraList_from_camInfos(
            cam_infos=scene_info.train_cameras, 
            resolution_scale=args.resolution, 
            args=args,
            lines=lines,
            opt=opt
        )
        
        print(f"Loading {len(scene_info.test_cameras)} Test Cameras...")
        self.test_cameras = cameraList_from_camInfos(
            cam_infos=scene_info.test_cameras, 
            resolution_scale=args.resolution, 
            args=args,
            lines=lines,
            opt=opt
        )

        # --- Initialize Gaussians ---
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """Saves the current state of the Gaussians to a .ply file."""
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        """Returns the list of training cameras."""
        return self.train_cameras

    def getTestCameras(self):
        """Returns the list of test cameras."""
        return self.test_cameras