# ==============================================================================
#                      IMPORTS & SETUP
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
from PIL import Image

try:
    import matplotlib_zh
    matplotlib_zh.use_zh()
    print("✅ 已启用matplotlib_zh以支持中文显示。")
except ImportError:
    print("⚠️ 警告: 未找到 matplotlib_zh 库，图表中的中文可能无法正常显示。请运行 'pip install matplotlib-zh'")

# 假设此脚本位于项目根目录
project_root = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, project_root)
try:
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.system_utils import searchForMaxIteration
except ImportError as e:
    print(f"❌ 导入模块失败，请确保此脚本位于gaussian-splatting项目根目录: {e}")
    sys.exit(1)

# ==============================================================================
#               1. 数据解析与指标计算函数
# ==============================================================================
def load_config(cfg_path: str) -> dict:
    """加载并解析 cfg_args 文件，兼容JSON和Namespace格式。"""
    try:
        with open(cfg_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        try:
            with open(cfg_path, 'r') as f: content = f.read().strip()
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
        except Exception as e: raise IOError(f"无法将 '{os.path.basename(cfg_path)}' 解析为JSON或Namespace字符串。错误: {e}")

def parse_csv_log(exp_path: str, log_filename: str):
    """解析训练日志CSV文件。"""
    log_path = os.path.join(exp_path, log_filename)
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0: return None
    try:
        df = pd.read_csv(log_path)
        return df if not df.empty else None
    except Exception as e:
        print(f"    - ❌ 读取 {log_path} 时出错: {e}")
        return None

# ==============================================================================
#           新增：无监督几何质量评估函数
# ==============================================================================
def evaluate_geometry_without_gt(pcd: o3d.geometry.PointCloud):
    """
    在没有真值点云的情况下，评估预测点云的内在几何质量。
    """
    metrics = {}
    if not pcd.has_points() or len(pcd.points) < 100:
        print("    -> [WARNING] 点云点数过少，跳过无监督几何评估。")
        return metrics

    print("    -> [INFO] 正在进行无监督几何质量评估...")
    
    try:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
        num_outliers = len(pcd.points) - len(ind)
        outlier_ratio = num_outliers / len(pcd.points)
        metrics["Geom_Outlier_Ratio"] = outlier_ratio
    except Exception as e:
        print(f"    -> [WARNING] 离群点分析失败: {e}")

    try:
        scene_extent = pcd.get_max_bound() - pcd.get_min_bound()
        radius_normal = np.linalg.norm(scene_extent) * 0.01 
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        if not pcd.has_normals(): raise RuntimeError("法线计算失败。")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        normals = np.asarray(pcd.normals)
        local_variances = []
        sample_indices = np.random.choice(len(pcd.points), size=min(10000, len(pcd.points)), replace=False)
        for idx in sample_indices:
            [k, indices, _] = pcd_tree.search_knn_vector_3d(pcd.points[idx], 20)
            if k < 5: continue
            neighbor_normals = normals[indices, :]
            covariance_matrix = np.cov(neighbor_normals, rowvar=False)
            variance = np.trace(covariance_matrix)
            local_variances.append(variance)
        if local_variances:
            metrics["Geom_Normal_Variance"] = np.mean(local_variances)
    except Exception as e:
        print(f"    -> [WARNING] 表面法线分析失败: {e}")

    return metrics

# ==============================================================================
#                       2. 核心分析流程
# ==============================================================================

@torch.no_grad()
def evaluate_final_model(model_path: str, args: Namespace):
    """加载单个实验的模型，并计算所有2D和3D（无监督）指标。"""
    exp_name = os.path.basename(model_path)
    print(f"\n{'─'*20} [INFO] 开始评估模型: {exp_name} {'─'*20}")
    
    computed_metrics = {}
    try:
        print("  -> 步骤1: 正在构建模型参数...")
        parser = ArgumentParser(description="评估参数加载器")
        model_params_def = ModelParams(parser)
        opt_params_def = OptimizationParams(parser)
        
        args_defaults = parser.parse_args([])
        saved_cfg_dict = load_config(os.path.join(model_path, "cfg_args"))
        
        for key, value in saved_cfg_dict.items():
            if hasattr(args_defaults, key): setattr(args_defaults, key, value)
        
        args_defaults.model_path = model_path
        model_params = model_params_def.extract(args_defaults)
        
        print("  -> 步骤2: 正在加载高斯点云...")
        
        iteration = "best"
        ply_path = os.path.join(model_path, "point_cloud", "best", "point_cloud.ply")
        
        if not os.path.exists(ply_path):
            print("    -> [INFO] 未找到 'best' 模型，回退到查找最大迭代次数模型...")
            iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            if iteration is None: raise FileNotFoundError("未找到任何 'best' 或 'iteration_*' 点云模型")
            ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        else:
            print("    -> [INFO] 发现并优先加载 'best' 模型。")
        
        gaussians = GaussianModel(sh_degree=model_params.sh_degree)
        gaussians.load_ply(ply_path)
        
        # ==================== 调试探针 1: 检查加载后的模型数据 ====================
        print("\n" + "─"*15 + " [DEBUG-PROBE 1: 模型数据检查] " + "─"*15)
        if gaussians.get_xyz is not None and gaussians.get_xyz.shape[0] > 0:
            print(f"    -> [DEBUG-PROBE] 模型点数: {gaussians.get_xyz.shape[0]}")
            print(f"    -> [DEBUG-PROBE] 模型坐标是否包含无效值(NaN): {torch.isnan(gaussians.get_xyz).any().item()}")
            print(f"    -> [DEBUG-PROBE] 模型坐标是否包含无穷大(inf): {torch.isinf(gaussians.get_xyz).any().item()}")
            print(f"    -> [DEBUG-PROBE] 模型不透明度(opacity)均值: {torch.mean(gaussians.get_opacity).item():.4f}")
            print(f"    -> [DEBUG-PROBE] 模型缩放(scale)均值: {torch.mean(gaussians.get_scaling).item():.4f}")
        else:
            print("    -> [DEBUG-PROBE] ⚠️ 警告: 高斯模型为空或未加载任何点！")
        print("─"*60 + "\n")
        # ========================================================================
        
        print(f"    -> [DEBUG] 模型加载成功 (来源: {iteration}, 点数: {gaussians.get_xyz.shape[0]})")
        computed_metrics["Total_Points"] = gaussians.get_xyz.shape[0]

        print("  -> 步骤3: 计算无监督3D几何指标...")
        pred_points = gaussians.get_xyz.detach().cpu().numpy()
        pred_pcd_raw = o3d.geometry.PointCloud()
        pred_pcd_raw.points = o3d.utility.Vector3dVector(pred_points)
        
        metrics_3d_no_ref = evaluate_geometry_without_gt(pred_pcd_raw)
        computed_metrics.update(metrics_3d_no_ref)
        
        debug_pcd_dir = os.path.join(args.parent_dir, "_debug_pcds")
        os.makedirs(debug_pcd_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(debug_pcd_dir, f"{exp_name}_pred_raw.ply"), pred_pcd_raw)

        if not args.skip_2d_eval:
            print("  -> 步骤4: 计算2D渲染指标...")
            
            temp_parser = ArgumentParser(description="Temp parser for pipeline")
            pipe_params = PipelineParams(temp_parser).extract(args_defaults)
            
            if not hasattr(pipe_params, 'debug'):
                print("    -> [FIX] 手动为 pipe_params 添加缺失的 .debug 属性")
                pipe_params.debug = False
            
            scene = Scene(model_params, gaussians, shuffle=False)
            test_cameras = scene.getTestCameras()
            if not test_cameras: test_cameras = scene.getTrainCameras()
            if not test_cameras:
                print("    - ❌ 错误: 找不到任何相机用于评估。")
                return computed_metrics

            psnr_list, ssim_list = [], []
            background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device="cuda")
            
            # ==================== 调试探针 2: 检查渲染前置参数 ====================
            print("\n" + "─"*15 + " [DEBUG-PROBE 2: 渲染前置检查] " + "─"*15)
            print(f"    -> [DEBUG-PROBE] 使用的背景色 (R,G,B): {background.cpu().numpy()}")
            print(f"    -> [DEBUG-PROBE] 测试相机数量: {len(test_cameras)}")
            print("─"*60 + "\n")
            # ========================================================================

            debug_img_dir = os.path.join(args.parent_dir, "_debug_images", exp_name)
            os.makedirs(debug_img_dir, exist_ok=True)
            img_idx = 0

            for camera in tqdm(test_cameras, desc=f"  渲染 {exp_name}", leave=False):
                # ==================== 调试探针 3: 检查单个相机参数 ====================
                if img_idx < 1: # 只对第一张图打印一次，避免刷屏
                    print("\n" + "─"*15 + " [DEBUG-PROBE 3: 首个相机检查] " + "─"*15)
                    print(f"    -> [DEBUG-PROBE] 相机视图矩阵是否包含无效值(NaN): {torch.isnan(camera.world_view_transform).any().item()}")
                    print(f"    -> [DEBUG-PROBE] 相机投影矩阵是否包含无效值(NaN): {torch.isnan(camera.full_proj_transform).any().item()}")
                    print(f"    -> [DEBUG-PROBE] 相机视场角 (FoV Y): {camera.FoVy}")
                    print("─"*60 + "\n")
                # ========================================================================
                
                render_pkg = render(camera, gaussians, pipe_params, background)
                rendered_img = render_pkg["render"].clamp(0.0, 1.0)
                gt_img = camera.original_image.clamp(0.0, 1.0).to("cuda")
                
                rendered_np = (rendered_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                if img_idx < 5:
                    try:
                        Image.fromarray(rendered_np).save(os.path.join(debug_img_dir, f"{img_idx:02d}_render.png"))
                        Image.fromarray(gt_np).save(os.path.join(debug_img_dir, f"{img_idx:02d}_gt.png"))
                    except Exception as e:
                        print(f"    - ⚠️ 保存调试图像失败: {e}")
                img_idx += 1

                psnr_list.append(psnr_sk(gt_np, rendered_np, data_range=255))
                ssim_list.append(ssim_sk(gt_np, rendered_np, channel_axis=-1, data_range=255))
            
            if psnr_list: computed_metrics["Test_PSNR"] = np.mean(psnr_list)
            if ssim_list: computed_metrics["Test_SSIM"] = np.mean(ssim_list)
            
    except Exception as e:
        print(f"  -> ❌ 在评估 {exp_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        
    return computed_metrics

# ==============================================================================
#                       3. 绘图与主控函数
# ==============================================================================
def plot_combined_figure(all_progress_data: dict, layout: list, output_path: str):
    """将所有实验的训练过程数据绘制到一张图中。"""
    print(f"\n🎨 正在生成多合一对比图...")
    if not any(df is not None and not df.empty for df in all_progress_data.values()):
        print("  -> ⚠️ 没有任何有效的训练日志数据，跳过绘图。")
        return
    num_plots = max(config['ax_idx'] for config in layout) + 1
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(20, 7 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]
    handles, labels = None, None
    for config in layout:
        ax = axes[config['ax_idx']]
        column_name = config['column']
        combined_df_list = []
        for exp_name, df in all_progress_data.items():
            if df is not None and column_name in df.columns:
                temp_df = df[['Iteration', column_name]].copy()
                temp_df.rename(columns={column_name: 'Value'}, inplace=True)
                temp_df['Experiment'] = exp_name
                combined_df_list.append(temp_df)
        if not combined_df_list: continue
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        sns.lineplot(data=combined_df, x='Iteration', y='Value', hue='Experiment', ax=ax, lw=1.5)
        ax.set_title(config.get('title', column_name), fontsize=18, pad=15)
        ax.set_ylabel(config.get('ylabel', 'Value'), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if config.get('log_y', False): ax.set_yscale('log')
        if config.get('ylim'): ax.set_ylim(config['ylim'])
        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
    if handles and labels:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=16, frameon=True)
    axes[-1].set_xlabel('迭代次数 (Iteration)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.9, 1.0])
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 多合一对比图已保存: {output_path}")

def main(args):
    """主函数，负责发现实验、循环评估、汇总结果。"""
    parent_dir = args.parent_dir
    print(f"🔍 开始在父目录中搜索子实验: {parent_dir}")
    sub_experiments = sorted([
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
        if os.path.isdir(os.path.join(parent_dir, d)) and os.path.exists(os.path.join(parent_dir, d, "cfg_args"))
    ])
    
    if not sub_experiments:
        print(f"❌ 在 '{parent_dir}' 中未找到任何有效的子实验。请检查路径或确保实验包含 'cfg_args' 文件。")
        return
    print(f"✅ 找到 {len(sub_experiments)} 个实验: {[os.path.basename(p) for p in sub_experiments]}")
    
    print("\n✅ 已配置为无监督几何评估模式，将不加载外部真值点云。")

    all_final_metrics = []
    all_progress_data = {}
    
    for exp_path in sub_experiments:
        exp_name = os.path.basename(exp_path)
        print(f"\n{'='*30} 主循环: 开始处理 {exp_name} {'='*30}")
        
        progress_df = parse_csv_log(exp_path, args.csv_log_name)
        all_progress_data[exp_name] = progress_df
        
        computed_metrics = evaluate_final_model(exp_path, args)
        
        print(f"    -> [DEBUG] {exp_name} 的评估计算结果: {computed_metrics}")

        final_csv_metrics = progress_df.iloc[-1].to_dict() if (progress_df is not None and not progress_df.empty) else {}
        combined_metrics = {"Experiment": exp_name, **final_csv_metrics, **computed_metrics}
        all_final_metrics.append(combined_metrics)

    if all_final_metrics:
        df = pd.DataFrame(all_final_metrics).set_index('Experiment')
        display_columns = [col for col in args.table_columns if col in df.columns]
        df_display = df[display_columns].copy()
        
        for col in df_display.columns:
            if pd.api.types.is_numeric_dtype(df_display[col]) and df_display[col].notna().any():
                    df_display.loc[:, col] = df_display[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        
        table_string = df_display.to_string()
        
        print("\n" + "="*80)
        print("                            📊 最终指标对比总表 📊")
        print("="*80)
        print(table_string)
        print("="*80)
        
        table_path = os.path.join(parent_dir, "comparison_table.txt")
        try:
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write("📊 最终指标对比总表 📊\n")
                f.write("="*80 + "\n")
                f.write(table_string)
                f.write("\n" + "="*80)
            print(f"📄 格式化总表已保存至: {table_path}")
        except Exception as e:
            print(f"❌ 保存格式化总表失败: {e}")

        summary_path = os.path.join(parent_dir, "comparison_summary.json")
        df.to_json(summary_path, orient='index', indent=4)
        print(f"📄 原始指标数据(JSON)已保存至: {summary_path}")
            
    plot_combined_figure(all_progress_data, args.plot_layout, os.path.join(parent_dir, "comparison_figure.png"))

if __name__ == "__main__":
    parser = ArgumentParser(description="用于3DGS实验的最终对比分析脚本 (集成无监督几何评估)。")
    parser.add_argument("parent_dir", type=str, help="包含单个场景下多个子实验的父目录路径 (例如 NORMAL_EXPERIMENTS/courtyard)。")
    parser.add_argument("--skip_2d_eval", action="store_true", help="跳过耗时的2D渲染评估，仅进行3D几何评估。")
    parser.add_argument("--csv_log_name", type=str, default="training_log.csv", help="训练日志CSV文件名。")
    
    args = parser.parse_args()
    
    args.plot_layout = [
        {'ax_idx': 0, 'column': 'Train_PSNR', 'title': '训练过程 PSNR', 'ylabel': 'PSNR (dB)'},
        {'ax_idx': 1, 'column': 'Total_Loss', 'title': '总损失', 'ylabel': 'Loss (log scale)', 'log_y': True},
    ]

    args.table_columns = [
        'Test_PSNR', 
        'Test_SSIM', 
        'Geom_Outlier_Ratio',
        'Geom_Normal_Variance',
        'Train_PSNR', 
        'Total_Points'
    ]
    
    main(args)