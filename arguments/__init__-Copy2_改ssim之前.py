#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.ininria.fr/graphdeco
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
            if key == 'geometry_constraint_type':
                kwargs['choices'] = ['none', 'normal', 'depth', 'normal_depth']

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
        self.antialiasing = False # 官方基线默认关闭
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # =================================================================================
        # >>> [ 🚀 A: 官方基线核心参数 ] <<<
        # =================================================================================
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
        
        # =================================================================================
        # >>> [ 🚀 B: 高级几何约束功能 ] <<<
        # =================================================================================
        # 控制几何约束的类型
        self.geometry_constraint_type = "none" 
        
        # 法线权重调节的强度 (alpha_N)
        self.alpha_normals = 0.0
        
        # 法线各向异性阈值: 只有 ratio(max_scale/min_scale) > 此阈值的高斯球才会被施加法线约束
        self.normal_anisotropy_threshold = 2.0
        
        # 几何约束应用的起始迭代次数
        self.geometry_start_iter = 7000
        
        # [ 🚀 关键新增 🚀 ] 对信度图应用Gamma校正的gamma值。
        # 默认值为1.0，表示不进行任何校正，直接使用原始信度图。
        # 建议使用 0.5 - 0.7 之间的值来“提亮”过暗的信度图。
        self.confidence_gamma = 1.0

        # (保留) 深度损失权重，当前无效
        self.lambda_depth = 0.0

        # =================================================================================
        # >>> [ 🚀 C: 各向同性浮游物抑制功能 ] <<<
        # =================================================================================
        # 各向同性惩罚的权重强度 (lambda_I)。默认0.0表示关闭。
        self.lambda_isotropy = 0.0

        # 各向同性惩罚应用的起始迭代次数。
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