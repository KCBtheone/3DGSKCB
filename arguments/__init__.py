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
            # MODIFICATION: Add choices for new parameters
            if key == 'confidence_scheme':
                kwargs['choices'] = ['none', 'multiplicative', 'gatekeeper', 'arbitrator', 'dual_l1']
            
            # [MODIFIED] Add choices for ALL diagnostic modes, including new ones
            if key == 'structural_loss_mode':
                kwargs['choices'] = [
                    'none',             # No structural loss
                    'sobel',            # V4 equivalent of base_grad (renamed for clarity)
                    'ms_sobel',         # V4 equivalent of ms_grad (renamed for clarity)
                    'scharr',           # Drop-in replacement for Sobel
                    'ms_scharr',        # Multi-scale Scharr
                    'log',              # Laplacian of Gaussian
                    'pfg',              # Perceptual Feature Gradient
                    'structure_tensor', # Structure Tensor analysis
                    'struct_ssim',      # Old struct mode, now renamed
                    'wavelet'           # Wavelet decomposition
                ]

            # Add choices for the synergy/guidance modes
            if key == 'synergy_mode':
                kwargs['choices'] = ['none', 'v1_linear', 'v2_p_weighted', 'v2_nonlinear', 'v2_ssim_guided', 'v4_fusion', 'v5_ultimate']
            
            #  Add choices for the gradient operator used in sobel/scharr modes
            if key == 'gradient_operator':
                 kwargs['choices'] = ['sobel', 'scharr']

            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true", help="<BOOL> " + key, **kwargs)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, help=f"<{t.__name__}> " + key, **kwargs)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", help="<BOOL> " + key, **kwargs)
                else:
                    group.add_argument("--" + key, default=value, type=t, help=f"<{t.__name__}> " + key, **kwargs)

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
        self.min_opacity = 0.005
        self.decouple_densification_grad = False # Decouple densification gradient from main loss

        # =================================================================================
        # >>> [ V5: 解耦融合框架 (The V5  Decoupled Fusion Framework) ] <<<
        # =================================================================================

        # --- [ A. 诊断模块 (Diagnostics Module) ] ---
        self.structural_loss_mode = "ms_sobel" # 控制使用哪种“诊断仪器”
        self.lambda_struct_loss = 0.05          # 诊断损失自身在总损失中的权重
        self.struct_loss_start_iter = 0         # 诊断模块启动的迭代次数

        # --- [ B. 协同引导模块 (Guidance Module) ] ---
        self.synergy_mode = "v5_ultimate"       # 控制如何利用诊断信息来引导L1和SSIM

        # --- [ C. V5 核心控制参数 (V5 Core Hyperparameters) ] ---
        self.alpha_l1_feedback = 1.5            # L1 引导的结构反馈强度 (α_L1)
        self.alpha_ssim_feedback = 1.5          # SSIM 引导的结构反馈强度 (α_SSIM)
        self.feedback_nonlinear_gamma = 2.0     # L1 非线性引导的伽马值 (γ)，>1.0 意味着“抓大放小”
        self.feedback_p_weighting_beta = 0.5    # 感知加权的强度 (β)，用于让诊断更关注人眼敏感区域

        # --- [ D. (兼容旧版) V4及之前版本的统一引导参数 ] ---
        self.alpha_struct_feedback = None # 旧版的统一引导强度，V5中被解耦的alpha_l1/ssim替代，设为None以避免冲突

        # =================================================================================
        # >>> [ 🔬 V5.1: 高级诊断仪器 (Advanced Diagnostic Instruments) ] <<<
        # =================================================================================
        # 这些参数仅在 `structural_loss_mode` 设置为对应模式时生效

        # --- [ 1. (模式 `ms_sobel`, `ms_scharr`) 多尺度参数 ] ---
        self.ms_grad_scales = 3                 # 多尺度计算的层数

        # --- [ 2. (模式 `log`) LoG算子参数 ] ---
        self.log_kernel_size = 5                # LoG 卷积核大小
        self.log_sigma = 1.4                    # LoG 中高斯平滑的标准差

        # --- [ 3. (模式 `pfg`) 感知特征梯度参数 ] ---
        self.pfg_feature_layer = "relu2_2"      # “驯服”策略: 使用更浅层的特征 (可选: 'relu1_2', 'relu3_3')

        # --- [ 4. (模式 `structure_tensor`) 结构张量参数 ] ---
        self.struct_tensor_neighborhood_size = 3 # 计算张量时邻域聚合的窗口大小 (e.g., 3x3 or 5x5)
        
        # --- [ 5. (模式 `struct_ssim`) 结构化SSIM参数 ] ---
        self.struct_ssim_window_size = 11       # SSIM诊断的窗口大小
        
        # --- [ 6. (模式 `wavelet`) 小波变换参数 ] ---
        self.wavelet_type = "db4"               # 小波基的类型
        self.wavelet_levels = 3                 # 小波分解的层数

        # =================================================================================
        # >>> [ 🧩 其他模块与损失函数 (Other Modules & Losses) ] <<<
        # =================================================================================

        # --- [ 感知损失 (Perceptual Loss) ] ---
        self.use_perceptual_loss = False
        self.lambda_perceptual = 0.01
        self.perceptual_start_iter = 15000
        
        # --- [ 信度调节方案，保留用于兼容性 ] ---
        self.confidence_scheme = "none"
        self.lambda_low_confidence = 0.1
        self.confidence_thresh = 0.5
        self.lambda_geo_low_conf = 1.0
        self.confidence_gamma = 1.0

        # --- [ 其他几何引导与正则化损失 ] ---
        self.use_normal_guidance = False
        self.alpha_normals = 0.02
        self.lambda_normals = 0.05
        self.use_smoothness_loss = False
        self.lambda_smooth = 0.001
        self.smooth_start_iter = 1000
        self.use_isotropy_loss = False
        self.lambda_isotropy = 0.1
        self.isotropy_start_iter = 5000
        self.use_sa_ssim = False
        self.beta_geo = 0.5
        self.adaptive_gamma = True
        self.gamma_base = 1.0
        self.gamma_warmup = 5000
        self.geometry_start_iter = 7000

        super().__init__(parser, "Optimization Parameters", fill_none=False)


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