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
    """
    A helper class to store extracted parameters for a specific group.
    """
    pass

class ParamGroup:
    """
    A base class for defining a group of related command-line arguments.
    It automatically adds arguments to the provided parser based on its
    own attributes.
    """
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            
            # Special handling for choice arguments like geometry_constraint_type
            kwargs = {}
            if key == 'geometry_constraint_type':
                kwargs['choices'] = ['none', 'line', 'udf']

            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true", **kwargs)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, **kwargs)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", **kwargs)
                else:
                    group.add_argument("--" + key, default=value, type=t, **kwargs)

    def extract(self, args):
        """
        Extracts the relevant arguments from the parsed namespace and returns
        them as a GroupParams object.
        """
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    """
    Parameters related to the model and dataset loading.
    """
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 1 
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    """
    Parameters controlling the rendering pipeline.
    """
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = True # Antialiasing is generally beneficial
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    """
    Parameters controlling the optimization and training process, including
    our flexible geometry constraints.
    """
    def __init__(self, parser):
        # --- Standard Optimization Parameters ---
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
        
        # ==============================================================================
        # >>> [核心] 几何约束方案选择器 <<<
        # ==============================================================================
        self.geometry_constraint_type = "none"

        # ==============================================================================
        # >>> 参数区: Line 约束方案 (仅当 geometry_constraint_type = 'line' 时生效) <<<
        # ==============================================================================
        self.line_static_alpha = 1.0
        self.line_static_sigma = 5.0
        self.line_dynamic_lambda = 0.1

        # ==============================================================================
        # >>> 参数区: UDF 约束方案 (仅当 geometry_constraint_type = 'udf' 时生效) <<<
        # ==============================================================================
        self.udf_dynamic_lambda = 1.0
        self.udf_blur_radius = 2

        # --- Densification & Pruning Parameters ---
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    """
    Utility function to combine arguments from the command line and a potential
    config file. Command line arguments take precedence.
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found.")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)