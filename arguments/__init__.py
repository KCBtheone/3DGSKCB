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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            kwargs = {}
            # This is a placeholder for potential future argument constraints
            # if key == 'some_future_param_with_choices':
            #     kwargs['choices'] = ['option1', 'option2']

            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true", help="<BOOL> " + key)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, help=f"<{t.__name__}> " + key)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", help="<BOOL> " + key)
                else:
                    group.add_argument("--" + key, default=value, type=t, help=f"<{t.__name__}> " + key)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # --- Official Core Parameters ---
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.random_background = False
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        # [!!! 修正 !!!] 将被意外删除的 min_opacity 参数加回来
        self.min_opacity = 0.005
        
        # =================================================================================
        # >>> [ 🚀 Custom Feature Parameters ] <<<
        # =================================================================================
        
        # --- [ 0. Direct Geometry Loss (您已有的方案) ] ---
        self.lambda_normals = 0.05

        # --- [ 1. (策略 #2) 法线平滑度正则化 ] ---
        self.use_smoothness_loss = False
        self.lambda_smooth = 0.001
        self.smooth_start_iter = 1000

        # --- [ 2. (策略 #3) 几何感知的致密化与剪枝 ] ---
        self.use_geometric_densify = False
        self.geo_densify_start_iter = 1000
        self.geo_inconsistency_threshold = 0.5
        self.geo_densify_relative_size_threshold = 0.1
        
        # --- [ 3. 您已有的其他自定义参数 (保留) ] ---
        self.confidence_loss_type = "multiplicative"
        self.confidence_gamma = 1.0
        self.confidence_alpha_fix = 1.0
        self.confidence_gamma_fix = 1.0
        self.use_normal_guidance = False
        self.alpha_normals = 0.02
        self.use_isotropy_loss = False
        self.lambda_isotropy = 0.1
        self.use_sa_ssim = False
        self.beta_geo = 0.5
        self.adaptive_gamma = True
        self.gamma_base = 1.0
        self.gamma_warmup = 5000
        self.geometry_start_iter = 7000
        self.isotropy_start_iter = 5000

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Reading cfg_args from", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)