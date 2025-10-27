# ==============================================================================
#           3DGS 基准测试分析脚本 (v5.0 - 全指标版)
# ==============================================================================
# [新增特性]
# - 新增3D几何指标:
#   - Geom_Anisotropy (各向异性): 衡量高斯点是否倾向于形成“薄片” (越高越好)。
#   - Geom_Planarity (平面度): 衡量“薄片”的平整程度 (越接近1越好)。
#   - Geom_Opacity_Purity (透明度纯净度): 衡量模型清晰度，少伪影 (越高越好)。
# - 新增2D无参考渲染质量指标:
#   - Test_Sharpness (感知锐度): 使用拉普拉斯方差评估图像清晰度 (越高越好)。
#   - Test_BRISQUE (盲图像质量): 使用BRISQUE模型评估图像自然度 (越低越好)。
#
# [依赖]
# - 请确保已安装 pyiqa: pip install pyiqa
#
# [保留]
# - 保留了 v4.1 的内存稳健性修复 (CPU K-近邻, 垃圾回收)。
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
import warnings
import time
import cv2
import lpips
import gc

# [新增] 引入高效的CPU K-近邻库 和 无参考图像质量评估库
from sklearn.neighbors import NearestNeighbors
try:
    import pyiqa
except ImportError:
    print("警告: `pyiqa` 库未安装。将无法计算 BRISQUE 指标。")
    print("请运行: pip install pyiqa")
    pyiqa = None

warnings.filterwarnings("ignore", category=UserWarning)


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

    log_path = os.path.join(exp_path, log_filename)
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0: return None
    try:
        return pd.read_csv(log_path)
    except Exception: return None

# ========================= [ 几何/图像质量指标计算函数 ] =========================

def calculate_local_normal_consistency(gaussians: GaussianModel, k: int = 8, batch_size: int = 8192):

    with torch.no_grad():
        normals = gaussians.get_normals.cuda()
        xyz_np = gaussians.get_xyz.detach().cpu().numpy()
        num_points = xyz_np.shape[0]
        if num_points < k + 1: return 1.0
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean', n_jobs=-1).fit(xyz_np)
        _, indices = nn_model.kneighbors(xyz_np)
        total_consistency = 0.0
        for i in tqdm(range(0, num_points, batch_size), desc="  计算法线一致性", leave=False, ncols=100):
            end = min(i + batch_size, num_points)
            source_normals_batch = normals[i:end]
            knn_indices_batch = torch.from_numpy(indices[i:end, 1:]).long().cuda()
            neighbor_normals = normals[knn_indices_batch]
            consistency_batch = torch.sum(source_normals_batch.unsqueeze(1) * neighbor_normals, dim=-1)
            total_consistency += torch.mean(consistency_batch).item() * (end - i)
        return total_consistency / num_points if num_points > 0 else 1.0

def calculate_anisotropy_planarity(gaussians: GaussianModel):
    """[新增] 计算高斯点的平均各向异性和平面度。"""
    with torch.no_grad():
        scales = gaussians.get_scaling
        if scales.shape[0] == 0: return 0.0, 0.0
        
        sorted_scales, _ = torch.sort(scales, dim=1)
        s_min, s_mid, s_max = sorted_scales[:, 0], sorted_scales[:, 1], sorted_scales[:, 2]
        
        epsilon = 1e-8
        anisotropy = s_max / (s_min + epsilon)
        planarity = s_mid / (s_max + epsilon)
        
        return torch.mean(anisotropy).item(), torch.mean(planarity).item()

def calculate_opacity_purity(gaussians: GaussianModel, low_thresh=0.1, high_thresh=0.9):
    """[新增] 计算透明度纯净度，即透明度接近0或1的点的百分比。"""
    with torch.no_grad():
        opacities = gaussians.get_opacity.squeeze()
        if opacities.numel() == 0: return 0.0
        
        is_pure = (opacities < low_thresh) | (opacities > high_thresh)
        purity_ratio = torch.mean(is_pure.float()).item()
        return purity_ratio

def get_depth_smoothness(depth_map_tensor: torch.Tensor):

    depth_map = depth_map_tensor.squeeze().cpu().numpy()
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    if not valid_mask.any(): return 0.0
    median_val = np.median(depth_map[valid_mask])
    depth_map[~valid_mask] = median_val
    laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
    return np.var(laplacian)

def get_normal_smoothness(normal_map_tensor: torch.Tensor):

    normal_gray = normal_map_tensor.mean(dim=0).cpu().numpy()
    sobel_x = cv2.Sobel(normal_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(normal_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(grad_mag)

def calculate_perceptual_sharpness(image_tensor: torch.Tensor):
    """[新增] 计算渲染图像的感知锐度 (拉普拉斯方差)。"""
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return laplacian.var()

# ========================= [ 主评估函数 ] =========================
@torch.no_grad()
def evaluate_final_model(model_path: str, args: Namespace):
    exp_name = os.path.basename(model_path)
    

    computed_metrics = {
        "Total_Points": None, "Geom_Outlier_Ratio": None, "Geom_Compactness": None,
        "Geom_Normal_Smoothness": None, "Test_PSNR": None, "Test_SSIM": None,
        "Test_PSNR_Std": None, "Efficiency_PSNR_per_100k_pts": None,
        "Test_LPIPS": None, "Test_PSNR_High_Freq": None,
        "Efficiency_Render_Time_ms": None, "Efficiency_Peak_VRAM_MB": None,
        "Geom_Local_Normal_Consistency": None, "Geom_Depth_Smoothness": None,
        "Geom_Anisotropy": None, "Geom_Planarity": None, "Geom_Opacity_Purity": None,
        "Test_Sharpness": None, "Test_BRISQUE": None
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
        

        computed_metrics["Geom_Local_Normal_Consistency"] = calculate_local_normal_consistency(gaussians)
        anisotropy, planarity = calculate_anisotropy_planarity(gaussians)
        computed_metrics["Geom_Anisotropy"] = anisotropy
        computed_metrics["Geom_Planarity"] = planarity
        computed_metrics["Geom_Opacity_Purity"] = calculate_opacity_purity(gaussians)
        

        pred_points = gaussians.get_xyz.detach().cpu().numpy()
        pred_pcd_raw = o3d.geometry.PointCloud()
        pred_pcd_raw.points = o3d.utility.Vector3dVector(pred_points)
        if len(pred_pcd_raw.points) > 100:
            try:
                _, ind = pred_pcd_raw.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
                computed_metrics["Geom_Outlier_Ratio"] = (len(pred_pcd_raw.points) - len(ind)) / len(pred_pcd_raw.points)
            except Exception: pass
        try:
            scales = gaussians.get_scaling.detach().cpu().numpy()
            log_scales = np.log(scales + 1e-8)
            computed_metrics["Geom_Compactness"] = np.exp(np.mean(log_scales))
        except Exception: pass

        if not args.skip_2d_eval:
            pipe_params = pipe_params_def.extract(args_defaults)
            model_params.eval = True
            scene = Scene(model_params, GaussianModel(model_params.sh_degree), load_iteration=None, shuffle=False)
            scene.gaussians = gaussians
            test_cameras = scene.getTestCameras() or scene.getTrainCameras()
            if not test_cameras: return computed_metrics
            
            lpips_fn = lpips.LPIPS(net='alex').to("cuda")

            iqa_model = None
            if pyiqa:
                try: iqa_model = pyiqa.create_metric('brisque', device=torch.device("cuda"))
                except Exception as e: print(f"警告: 初始化 BRISQUE 模型失败: {e}")

            torch.cuda.reset_peak_memory_stats("cuda")
            

            psnr_list, ssim_list, lpips_list, render_time_list = [], [], [], []
            normal_smoothness_list, psnr_hf_list = [], []
            depth_smoothness_list = []
            sharpness_list, brisque_list = [], []
            
            background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device="cuda")
            
            for camera in tqdm(test_cameras, desc=f"  渲染评估 {exp_name}", leave=False, ncols=100):
                camera.to_device("cuda")
                
                torch.cuda.synchronize(); start_time = time.time()
                render_pkg = render(camera, gaussians, pipe_params, background)
                torch.cuda.synchronize(); render_time_list.append((time.time() - start_time) * 1000)

                rendered_img = render_pkg["render"].clamp(0.0, 1.0)
                gt_img = camera.original_image.clamp(0.0, 1.0).to("cuda")
                
                # ... (PSNR, SSIM, LPIPS, HF-PSNR 计算无需修改) ...
                rendered_np = (rendered_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                psnr_list.append(psnr_sk(gt_np, rendered_np, data_range=255))
                ssim_list.append(ssim_sk(gt_np, rendered_np, channel_axis=-1, data_range=255))
                try:
                    rendered_lpips = rendered_img * 2.0 - 1.0; gt_lpips = gt_img * 2.0 - 1.0
                    lpips_list.append(lpips_fn(rendered_lpips, gt_lpips).item())
                except Exception: pass
                try:
                    gt_np_gray = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY)
                    laplacian_gt = cv2.Laplacian(gt_np_gray, cv2.CV_64F)
                    mask = np.abs(laplacian_gt) > np.percentile(np.abs(laplacian_gt), 80)
                    if mask.sum() > 0: psnr_hf_list.append(psnr_sk(gt_np[mask], rendered_np[mask], data_range=255))
                except Exception: pass

                # [新增] 计算新的2D指标
                try: sharpness_list.append(calculate_perceptual_sharpness(rendered_img))
                except Exception: pass
                if iqa_model:
                    try: brisque_list.append(iqa_model(rendered_img.unsqueeze(0)).item())
                    except Exception: pass

                # ... (深度/法线平滑度计算无需修改) ...
                try: depth_smoothness_list.append(get_depth_smoothness(render_pkg["depth"]))
                except Exception: pass
                try:
                    normals = (gaussians.get_normals.detach() + 1.0) / 2.0
                    normal_render_pkg = render(camera, gaussians, pipe_params, background, override_color=normals)
                    normal_smoothness_list.append(get_normal_smoothness(normal_render_pkg["render"]))
                except Exception: pass

            # ... (聚合所有指标) ...
            computed_metrics["Efficiency_Peak_VRAM_MB"] = torch.cuda.max_memory_allocated("cuda") / (1024 ** 2)
            if psnr_list:
                computed_metrics["Test_PSNR"] = np.mean(psnr_list)
                computed_metrics["Test_PSNR_Std"] = np.std(psnr_list)
            if ssim_list: computed_metrics["Test_SSIM"] = np.mean(ssim_list)
            if lpips_list: computed_metrics["Test_LPIPS"] = np.mean(lpips_list)
            if psnr_hf_list: computed_metrics["Test_PSNR_High_Freq"] = np.mean(psnr_hf_list)
            if render_time_list: computed_metrics["Efficiency_Render_Time_ms"] = np.mean(render_time_list)
            if depth_smoothness_list: computed_metrics["Geom_Depth_Smoothness"] = np.mean(depth_smoothness_list)
            if normal_smoothness_list: computed_metrics["Geom_Normal_Smoothness"] = np.mean(normal_smoothness_list)
            # [新增] 聚合新指标
            if sharpness_list: computed_metrics["Test_Sharpness"] = np.mean(sharpness_list)
            if brisque_list: computed_metrics["Test_BRISQUE"] = np.mean(brisque_list)
            
            if computed_metrics["Test_PSNR"] and computed_metrics["Total_Points"]:
                efficiency = (computed_metrics["Test_PSNR"] / (computed_metrics["Total_Points"] / 100000.0))
                computed_metrics["Efficiency_PSNR_per_100k_pts"] = efficiency
    except Exception as e:
        print(f"  -> ❌ 在评估 {exp_name} 时出错: {e}")
    finally:
        if 'gaussians' in locals(): del gaussians
        if 'scene' in locals(): del scene
        if 'lpips_fn' in locals(): del lpips_fn
        if 'iqa_model' in locals(): del iqa_model
        gc.collect()
        torch.cuda.empty_cache()
    
    return computed_metrics

def plot_combined_figure(all_progress_data: dict, layout: list, output_path: str):

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

    scenes_to_analyze = []
    base_path = ""
    dir_content = [d for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]

    is_single_scene_folder = any("cfg_args" in os.listdir(os.path.join(args.input_path, d)) for d in dir_content if os.path.isdir(os.path.join(args.input_path, d)))
    
    if not is_single_scene_folder and os.path.exists(os.path.join(args.input_path, "cfg_args")):
         # 处理输入是单个实验文件夹的情况
        scenes_to_analyze = [os.path.basename(os.path.dirname(args.input_path))]
        base_path = os.path.dirname(os.path.dirname(args.input_path))
    elif is_single_scene_folder:
        # 输入是包含多个场景的文件夹
        scenes_to_analyze = [d for d in dir_content if os.path.isdir(os.path.join(args.input_path, d))]
        base_path = args.input_path
    else:
        # 输入是包含多个实验的单个场景文件夹
        scenes_to_analyze = [os.path.basename(args.input_path)]
        base_path = os.path.dirname(args.input_path)

    parent_folder_name = os.path.basename(os.path.abspath(base_path))


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
            

            table_path = os.path.join(OUTPUT_DIR, f"table_{parent_folder_name}_{scene_name}.csv")
            figure_path = os.path.join(OUTPUT_DIR, f"figure_{parent_folder_name}_{scene_name}.png")


            df_scene[display_columns].to_csv(table_path)
            print(f"--- 场景 [{scene_name}] 指标对比表 (已保存至 {table_path}) ---\n", df_scene[display_columns])
            
            plot_combined_figure(scene_progress_data, args.plot_layout, figure_path)
            master_results.extend(scene_final_metrics)

    if master_results:
        df_master = pd.DataFrame(master_results)
        df_master.to_csv(os.path.join(OUTPUT_DIR, "master_raw_data.csv"), index=False)
        print(f"\n✅ 全局原始数据已保存至: {os.path.join(OUTPUT_DIR, 'master_raw_data.csv')}")

if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS 基准测试分析脚本 (v5.0 - 全指标版)")
    parser.add_argument("input_path", type=str, help="要分析的顶级实验目录或单个场景目录。")
    parser.add_argument("-o", "--output_dir", type=str, default="analysis_output_final", help="保存所有分析结果的输出目录。")
    parser.add_argument("--skip_2d_eval", action="store_true", help="跳过渲染和计算 2D 指标。")
    cli_args = parser.parse_args()

    class ScriptConfig(Namespace):
        csv_log_name = "training_log.csv"
        
        # [新增] 定义全新的、更全面的绘图布局
        plot_layout = [
            # 核心渲染质量
            {'ax_idx': 0, 'column': 'Test_PSNR', 'title': '核心渲染质量: PSNR (越高越好)'},
            {'ax_idx': 1, 'column': 'Test_LPIPS', 'title': '核心渲染质量: 感知损失 LPIPS (越低越好)'},
            # 几何纯净度与平滑度
            {'ax_idx': 2, 'column': 'Geom_Local_Normal_Consistency', 'title': '几何质量: 局部法线一致性 (越高越好)'},
            {'ax_idx': 3, 'column': 'Geom_Opacity_Purity', 'title': '几何质量: 透明度纯净度 (越高越好)'},
            {'ax_idx': 4, 'column': 'Geom_Depth_Smoothness', 'title': '几何质量: 深度图平滑度 (越低越好)'},
            # 几何形状表达
            {'ax_idx': 5, 'column': 'Geom_Anisotropy', 'title': '几何形状: 各向异性 (越高越好)'},
            {'ax_idx': 6, 'column': 'Geom_Planarity', 'title': '几何形状: 平面度 (越接近1越好)'},
            # 无参考渲染细节
            {'ax_idx': 7, 'column': 'Test_Sharpness', 'title': '无参考渲染质量: 感知锐度 (越高越好)'},
            {'ax_idx': 8, 'column': 'Test_BRISQUE', 'title': '无参考渲染质量: BRISQUE 自然度 (越低越好)'},
            # 模型复杂度
            {'ax_idx': 9, 'column': 'Total_Points', 'title': '模型复杂度: 总点数', 'log_y': True},
        ]
        

        table_columns = [
            # 核心渲染指标
            'Test_PSNR', 'Test_SSIM', 'Test_LPIPS', 'Test_PSNR_High_Freq', 'Test_PSNR_Std', 
            # 新增无参考渲染指标
            'Test_Sharpness', 'Test_BRISQUE',
            # 核心几何指标
            'Geom_Local_Normal_Consistency', 'Geom_Depth_Smoothness', 'Geom_Normal_Smoothness', 
            # 新增几何指标
            'Geom_Anisotropy', 'Geom_Planarity', 'Geom_Opacity_Purity',
            'Geom_Compactness', 'Geom_Outlier_Ratio', 
            # 效率与复杂度指标
            'Total_Points', 'Efficiency_PSNR_per_100k_pts',
            'Efficiency_Render_Time_ms', 'Efficiency_Peak_VRAM_MB',
        ]
    
    args = ScriptConfig(**vars(cli_args))


    def camera_to_device(self, device):
        for attr in ['world_view_transform', 'full_proj_transform', 'camera_center']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
    Camera.to_device = camera_to_device

    main(args)
    print("\n 分析完毕！")