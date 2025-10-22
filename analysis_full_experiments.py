# ==============================================================================
#           3DGS 基准测试分析脚本 (v3.0 - 感知与效率版)
# ==============================================================================
# 新增指标:
# - Geom_Normal_Smoothness: 渲染法线图平滑度 (越低越好)
# - Geom_Compactness: 高斯体平均尺度 (越低越好)
# - Test_PSNR_Std: 测试集PSNR标准差 (越低越好)
# - Efficiency_PSNR_per_100k_pts: 模型效率 (越高越好)
# [⭐ 新增]
# - Test_LPIPS: 感知图像相似度 (越低越好, 差异更显著)
# - Test_PSNR_High_Freq: 高频细节区域PSNR (越高越好, 衡量纹理清晰度)
# - Efficiency_Render_Time_ms: 平均渲染时间 (越低越好)
# - Efficiency_Peak_VRAM_MB: 峰值显存占用 (越低越好)
# ==============================================================================

import os
import sys
import json
import numpy as np
import torch
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
import re
import copy
import warnings
import time  # [⭐ 新增]
import cv2   # [⭐ 新增]
import lpips # [⭐ 新增]

warnings.filterwarnings("ignore", category=UserWarning)

# (辅助函数部分保持不变, 这里省略以保持简洁)
try:
    import torchvision.utils as vutils
except ImportError:
    vutils = None
try:
    import matplotlib_zh
    matplotlib_zh.use_zh()
except ImportError:
    pass
project_root = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, project_root)
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.system_utils import searchForMaxIteration

def load_config(cfg_path: str) -> dict:
    # (此函数保持不变)
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f: content = f.read().strip()
        if content.startswith("Namespace(") and content.endswith(")"): content = content[10:-1]
        pattern = re.compile(r"(\w+)\s*=\s*('([^']*)'|\"([^\"]*)\"|\[.*?\]|[\w.-]+|True|False|None)")
        matches = pattern.findall(content)
        cfg_dict = {}
        for key, val_group, str_val1, str_val2 in matches:
            val_str = val_group
            if (val_str.startswith("'") and val_str.endswith("'")) or (val_str.startswith('"') and val_str.endswith('"')): cfg_dict[key] = val_str[1:-1]
            elif val_str == 'True': cfg_dict[key] = True
            elif val_str == 'False': cfg_dict[key] = False
            elif val_str == 'None': cfg_dict[key] = None
            elif val_str.startswith('[') and val_str.endswith(']'):
                try: cfg_dict[key] = eval(val_str)
                except: cfg_dict[key] = val_str
            else:
                try:
                    if '.' in val_str: cfg_dict[key] = float(val_str)
                    else: cfg_dict[key] = int(val_str)
                except ValueError: cfg_dict[key] = val_str
        return cfg_dict
    except Exception as e: raise IOError(f"无法解析 '{os.path.basename(cfg_path)}'。错误: {e}")

def parse_csv_log(exp_path: str, log_filename: str):
    # (此函数保持不变)
    log_path = os.path.join(exp_path, log_filename)
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0: return None
    try:
        return pd.read_csv(log_path)
    except Exception: return None

# ========================= [ 指标计算区 ] =========================
def evaluate_geometry_without_gt(pcd: o3d.geometry.PointCloud, gaussians: GaussianModel):
    metrics = {}
    if not pcd.has_points(): return metrics
    if len(pcd.points) > 100:
        try:
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
            outlier_ratio = (len(pcd.points) - len(ind)) / len(pcd.points)
            metrics["Geom_Outlier_Ratio"] = outlier_ratio
        except Exception: pass
    try:
        scales = gaussians.get_scaling.detach().cpu().numpy()
        log_scales = np.log(scales + 1e-8)
        geom_mean_scale = np.exp(np.mean(log_scales))
        metrics["Geom_Compactness"] = geom_mean_scale
    except Exception: pass
    return metrics

def get_normal_smoothness(normal_map_tensor: torch.Tensor):
    normal_gray = normal_map_tensor.mean(dim=0).cpu().numpy()
    sobel_x = cv2.Sobel(normal_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(normal_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(grad_mag)
# ====================================================================

@torch.no_grad()
def evaluate_final_model(model_path: str, args: Namespace):
    exp_name = os.path.basename(model_path)
    # [⭐ 修改] 初始化所有指标
    computed_metrics = {
        "Total_Points": None, "Geom_Outlier_Ratio": None, "Geom_Compactness": None,
        "Geom_Normal_Smoothness": None, "Test_PSNR": None, "Test_SSIM": None,
        "Test_PSNR_Std": None, "Efficiency_PSNR_per_100k_pts": None,
        "Test_LPIPS": None, "Test_PSNR_High_Freq": None,
        "Efficiency_Render_Time_ms": None, "Efficiency_Peak_VRAM_MB": None
    }
    try:
        parser = ArgumentParser()
        model_params_def = ModelParams(parser)
        pipe_params_def = PipelineParams(parser)
        args_defaults = parser.parse_args([])
        saved_cfg_dict = load_config(os.path.join(model_path, "cfg_args"))
        for key, value in saved_cfg_dict.items():
            if hasattr(args_defaults, key): setattr(args_defaults, key, value)
        args_defaults.model_path = model_path
        model_params = model_params_def.extract(args_defaults)
        
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        ply_path_best = os.path.join(point_cloud_dir, "best", "point_cloud.ply")
        if os.path.exists(ply_path_best): ply_path = ply_path_best
        else:
            iteration = searchForMaxIteration(point_cloud_dir)
            if iteration is None: raise FileNotFoundError(f"找不到 'best' 或 'iteration_XXX' 目录")
            ply_path = os.path.join(point_cloud_dir, f"iteration_{iteration}", "point_cloud.ply")

        gaussians = GaussianModel(sh_degree=model_params.sh_degree)
        gaussians.load_ply(ply_path)
        computed_metrics["Total_Points"] = gaussians.get_xyz.shape[0]
        
        pred_points = gaussians.get_xyz.detach().cpu().numpy()
        pred_pcd_raw = o3d.geometry.PointCloud()
        pred_pcd_raw.points = o3d.utility.Vector3dVector(pred_points)
        
        computed_metrics.update(evaluate_geometry_without_gt(pred_pcd_raw, gaussians))

        if not args.skip_2d_eval:
            pipe_params = pipe_params_def.extract(args_defaults)
            model_params.eval = True
            scene = Scene(model_params, GaussianModel(model_params.sh_degree), load_iteration=None, shuffle=False)
            scene.gaussians = gaussians
            test_cameras = scene.getTestCameras() or scene.getTrainCameras()
            if not test_cameras: return computed_metrics
            
            # [⭐ 新增] 初始化 LPIPS 模型和指标列表
            lpips_fn = lpips.LPIPS(net='alex').to("cuda")
            torch.cuda.reset_peak_memory_stats("cuda")
            
            psnr_list, ssim_list, lpips_list, normal_smoothness_list, psnr_hf_list, render_time_list = [], [], [], [], [], []
            background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device="cuda")
            
            for camera in tqdm(test_cameras, desc=f"  渲染评估 {exp_name}", leave=False, ncols=80):
                camera.to_device("cuda")
                
                # [⭐ 新增] 记录渲染时间
                torch.cuda.synchronize()
                start_time = time.time()
                render_pkg = render(camera, gaussians, pipe_params, background)
                torch.cuda.synchronize()
                render_time_list.append((time.time() - start_time) * 1000) # 转换为毫秒

                rendered_img = render_pkg["render"].clamp(0.0, 1.0)
                gt_img = camera.original_image.clamp(0.0, 1.0).to("cuda")
                
                rendered_np = (rendered_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                psnr_list.append(psnr_sk(gt_np, rendered_np, data_range=255))
                ssim_list.append(ssim_sk(gt_np, rendered_np, channel_axis=-1, data_range=255))

                # [⭐ 新增] 计算 LPIPS (越低越好)
                try:
                    rendered_lpips = rendered_img * 2.0 - 1.0 # 归一化到 [-1, 1]
                    gt_lpips = gt_img * 2.0 - 1.0
                    lpips_list.append(lpips_fn(rendered_lpips, gt_lpips).item())
                except Exception: pass

                # [⭐ 新增] 计算高频区域PSNR (越高越好)
                try:
                    gt_np_gray = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY)
                    laplacian = cv2.Laplacian(gt_np_gray, cv2.CV_64F)
                    # 定义高频区域为梯度值在前20%的像素
                    mask = np.abs(laplacian) > np.percentile(np.abs(laplacian), 80)
                    if mask.sum() > 0:
                        psnr_hf = psnr_sk(gt_np[mask], rendered_np[mask], data_range=255)
                        psnr_hf_list.append(psnr_hf)
                except Exception: pass

                # 计算法线平滑度 (保留)
                try:
                    normals = (gaussians.get_normals.detach() + 1.0) / 2.0
                    normal_render_pkg = render(camera, gaussians, pipe_params, background, override_color=normals)
                    rendered_normals = normal_render_pkg["render"]
                    normal_smoothness_list.append(get_normal_smoothness(rendered_normals))
                except Exception: pass

            # [⭐ 新增] 记录渲染过程中的峰值显存
            peak_vram_mb = torch.cuda.max_memory_allocated("cuda") / (1024 ** 2)
            computed_metrics["Efficiency_Peak_VRAM_MB"] = peak_vram_mb

            # 聚合所有指标
            if psnr_list:
                computed_metrics["Test_PSNR"] = np.mean(psnr_list)
                computed_metrics["Test_PSNR_Std"] = np.std(psnr_list)
            if ssim_list: computed_metrics["Test_SSIM"] = np.mean(ssim_list)
            if normal_smoothness_list: computed_metrics["Geom_Normal_Smoothness"] = np.mean(normal_smoothness_list)
            
            # [⭐ 新增] 聚合新指标
            if lpips_list: computed_metrics["Test_LPIPS"] = np.mean(lpips_list)
            if psnr_hf_list: computed_metrics["Test_PSNR_High_Freq"] = np.mean(psnr_hf_list)
            if render_time_list: computed_metrics["Efficiency_Render_Time_ms"] = np.mean(render_time_list)

            if computed_metrics["Test_PSNR"] and computed_metrics["Total_Points"]:
                efficiency = (computed_metrics["Test_PSNR"] / (computed_metrics["Total_Points"] / 100000.0))
                computed_metrics["Efficiency_PSNR_per_100k_pts"] = efficiency
    except Exception as e:
        print(f"  -> ❌ 在评估 {exp_name} 时出错: {e}")
    return computed_metrics

def plot_combined_figure(all_progress_data: dict, layout: list, output_path: str):
    # (此函数保持不变)
    if not any(df is not None for df in all_progress_data.values()): return
    print(f"🎨 正在为场景 {os.path.basename(output_path).replace('figure_','').replace('.png','')} 生成对比图...")
    num_plots = max(config['ax_idx'] for config in layout) + 1
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(20, 6 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]
    handles, labels = None, None
    for config in layout:
        ax = axes[config['ax_idx']]
        column_name = config['column']
        combined_df_list = []
        for exp_name, df in all_progress_data.items():
            if df is not None and column_name in df.columns:
                temp_df = df[['Iteration', column_name]].copy()
                temp_df.rename(columns={column_name: exp_name}, inplace=True)
                combined_df_list.append(temp_df.set_index('Iteration'))
        if not combined_df_list: continue
        combined_df = pd.concat(combined_df_list, axis=1)
        sns.lineplot(data=combined_df, ax=ax, lw=1.5, dashes=False)
        ax.set_title(config.get('title', column_name))
        if config.get('log_y', False): ax.set_yscale('log')
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend(): ax.get_legend().remove()
    if handles and labels: fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.95))
    axes[-1].set_xlabel('迭代次数')
    fig.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 对比图已保存: {output_path}")

def main(args):
    # (main 函数的主体逻辑保持不变)
    scenes_to_analyze = []
    base_path = ""
    dir_content = [d for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]
    is_scene_folder = any(os.path.exists(os.path.join(args.input_path, d, "cfg_args")) for d in dir_content)
    if is_scene_folder:
        scenes_to_analyze = [os.path.basename(args.input_path)]
        base_path = os.path.dirname(args.input_path)
    else:
        scenes_to_analyze = [d for d in dir_content]
        base_path = args.input_path
    
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📊 所有分析结果将保存至: {OUTPUT_DIR}")
    
    master_results = []
    for scene_name in tqdm(scenes_to_analyze, desc="处理所有场景"):
        scene_parent_dir = os.path.join(base_path, scene_name) 
        if not os.path.exists(scene_parent_dir): continue
        
        scene_final_metrics, scene_progress_data = [], {}
        sub_experiments = [os.path.join(scene_parent_dir, d) for d in os.listdir(scene_parent_dir) if os.path.isdir(os.path.join(scene_parent_dir, d)) and os.path.exists(os.path.join(scene_parent_dir, d, "cfg_args"))]
        print(f"\n--- 场景: {scene_name} | 发现 {len(sub_experiments)} 个实验 ---")

        for exp_path in sub_experiments:
            exp_name = os.path.basename(exp_path)
            progress_df = parse_csv_log(exp_path, args.csv_log_name)
            scene_progress_data[f"{exp_name}"] = progress_df
            computed_metrics = evaluate_final_model(exp_path, args)
            final_csv_metrics = progress_df.iloc[-1].to_dict() if (progress_df is not None and not progress_df.empty) else {}
            scene_final_metrics.append({"Scene": scene_name, "Experiment": exp_name, **final_csv_metrics, **computed_metrics})

        if scene_final_metrics:
            df_scene = pd.DataFrame(scene_final_metrics).set_index('Experiment')
            display_columns = [col for col in args.table_columns if col in df_scene.columns]
            table_path = os.path.join(OUTPUT_DIR, f"table_{scene_name}.csv")
            df_scene[display_columns].to_csv(table_path)
            print(f"--- 场景 [{scene_name}] 指标对比表 (已保存至 {table_path}) ---\n", df_scene[display_columns])
            
            plot_combined_figure(scene_progress_data, args.plot_layout, os.path.join(OUTPUT_DIR, f"figure_{scene_name}.png"))
            master_results.extend(scene_final_metrics)

    if master_results:
        df_master = pd.DataFrame(master_results)
        df_master.to_csv(os.path.join(OUTPUT_DIR, "master_raw_data.csv"), index=False)
        print(f"\n✅ 全局原始数据已保存至: {os.path.join(OUTPUT_DIR, 'master_raw_data.csv')}")

if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS 基准测试分析脚本 (v3.0 - 感知与效率版)")
    parser.add_argument("input_path", type=str, help="要分析的顶级实验目录或单个场景目录。")
    parser.add_argument("-o", "--output_dir", type=str, default="analysis_output_final", help="保存所有分析结果的输出目录。")
    parser.add_argument("--skip_2d_eval", action="store_true", help="跳过渲染和计算 2D 指标。")
    cli_args = parser.parse_args()

    class ScriptConfig(Namespace):
        csv_log_name = "training_log.csv"
        # [⭐ 修改] 增加新图表布局, LPIPS非常重要
        plot_layout = [
            {'ax_idx': 0, 'column': 'Test_PSNR', 'title': '测试集 PSNR (越高越好)'},
            {'ax_idx': 1, 'column': 'Test_LPIPS', 'title': '感知损失 LPIPS (越低越好)'},
            {'ax_idx': 2, 'column': 'Geom_Normal_Smoothness', 'title': '几何平滑度 (越低越好)'},
            {'ax_idx': 3, 'column': 'Efficiency_PSNR_per_100k_pts', 'title': '模型效率 (越高越好)'},
            {'ax_idx': 4, 'column': 'Total_Points', 'title': '总点数 (越低越好，在PSNR相近时)'},
        ]
        # [⭐ 修改] 增加新表格列, 全面展示模型性能
        table_columns = [
            'Test_PSNR', 'Test_SSIM', 'Test_LPIPS', 'Test_PSNR_High_Freq', 'Test_PSNR_Std', 
            'Geom_Normal_Smoothness', 'Geom_Compactness', 'Geom_Outlier_Ratio', 
            'Total_Points', 'Efficiency_PSNR_per_100k_pts',
            'Efficiency_Render_Time_ms', 'Efficiency_Peak_VRAM_MB',
            'Train_PSNR'
        ]
    
    args = ScriptConfig(**vars(cli_args))

    def camera_to_device(self, device):
        for attr in ['world_view_transform', 'full_proj_transform', 'camera_center']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
    Camera.to_device = camera_to_device

    main(args)
    print("\n🎉 分析完毕！")