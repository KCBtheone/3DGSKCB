#
# ==============================================================================
#                 3DGS 基准测试结果分析脚本 (Best模型专用版)
# ==============================================================================
# 核心逻辑变更:
# - 本脚本现在只评估由早停机制保存的 "best" 模型。
# - 不再查找最终迭代次数，而是直接定位 ".../point_cloud/best/point_cloud.ply"。
# - 如果某个实验没有 "best" 模型，它将被安全地跳过。
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
import copy # <<< 确保导入 copy 模块 >>>

# --- [新增] 尝试导入 torchvision 用于保存调试图像 ---
try:
    import torchvision.utils as vutils
    print("✅ 已成功导入 torchvision.utils 用于保存调试图像。")
except ImportError:
    print("⚠️ 警告: 未找到 torchvision 库，无法保存调试图像。请运行 'pip install torchvision'")
    vutils = None
# --- [新增结束] ---

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
    from utils.system_utils import searchForMaxIteration # 尽管不再主动使用，但保留导入以防万一
except ImportError as e:
    print(f"❌ 导入模块失败，请确保此脚本位于gaussian-splatting项目根目录: {e}")
    sys.exit(1)

# ==============================================================================
#               1. 数据解析与指标计算函数 (保持不变)
# ==============================================================================
def load_config(cfg_path: str) -> dict:
    """加载并解析 cfg_args 文件，兼容JSON和Namespace格式。"""
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
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

def evaluate_geometry_without_gt(pcd: o3d.geometry.PointCloud):
    """在没有真值点云的情况下，评估预测点云的内在几何质量。"""
    metrics = {}
    if not pcd.has_points() or len(pcd.points) < 100:
        print("    -> [警告] 点云点数过少，跳过无监督几何评估。")
        return metrics
    try:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
        num_outliers = len(pcd.points) - len(ind)
        outlier_ratio = num_outliers / len(pcd.points)
        metrics["Geom_Outlier_Ratio"] = outlier_ratio
    except Exception as e:
        print(f"    -> [警告] 离群点分析失败: {e}")
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
        print(f"    -> [警告] 表面法线分析失败: {e}")
    return metrics

# ==============================================================================
#                       2. 核心分析流程 (已按新逻辑重构)
# ==============================================================================
@torch.no_grad()
def evaluate_best_model(model_path: str, args: Namespace):
    """加载单个实验的'best'模型，并计算所有2D和3D（无监督）指标。"""
    exp_name = os.path.basename(model_path)
    
    computed_metrics = {
        "Total_Points": None, "Geom_Outlier_Ratio": None, 
        "Geom_Normal_Variance": None, "Test_PSNR": None, "Test_SSIM": None,
    }
    
    try:
        # --- 步骤 1: 加载 'best' 模型和计算点云指标 ---
        # ============================ [ ⭐⭐⭐ 核心修改处 ⭐⭐⭐ ] ============================
        print(f"  -> [INFO] 正在查找 'best' 模型...")
        best_model_ply_path = os.path.join(model_path, "point_cloud", "best", "point_cloud.ply")

        if not os.path.exists(best_model_ply_path):
            print(f"  -> [警告] 在 {exp_name} 中未找到 'point_cloud/best/point_cloud.ply'。根据要求，跳过此实验的评估。")
            return computed_metrics # 返回空的 metrics 字典

        print(f"  -> [INFO] 找到了 'best' 模型，正在加载...")
        # =================================================================================

        parser = ArgumentParser(description="评估参数加载器")
        model_params_def = ModelParams(parser)
        OptimizationParams(parser)
        args_defaults = parser.parse_args([])
        
        saved_cfg_dict = load_config(os.path.join(model_path, "cfg_args"))
        for key, value in saved_cfg_dict.items():
            if hasattr(args_defaults, key): setattr(args_defaults, key, value)
        args_defaults.model_path = model_path
        model_params = model_params_def.extract(args_defaults)
        
        gaussians = GaussianModel(sh_degree=model_params.sh_degree)
        gaussians.load_ply(best_model_ply_path)
        
        computed_metrics["Total_Points"] = gaussians.get_xyz.shape[0]
        
        pred_points = gaussians.get_xyz.detach().cpu().numpy()
        pred_pcd_raw = o3d.geometry.PointCloud()
        pred_pcd_raw.points = o3d.utility.Vector3dVector(pred_points)
        metrics_3d_no_ref = evaluate_geometry_without_gt(pred_pcd_raw)
        computed_metrics.update(metrics_3d_no_ref)


        # --- 步骤 2: 2D 渲染评估 ---
        if not args.skip_2d_eval:
            temp_parser = ArgumentParser(description="Temp parser for pipeline")
            pipe_params = PipelineParams(temp_parser).extract(args_defaults)
            if not hasattr(pipe_params, 'debug'): pipe_params.debug = False
            
            scene = Scene(model_params, copy.deepcopy(gaussians), shuffle=False) 
            test_cameras = scene.getTestCameras()
            
            if not test_cameras:
                print(f"  -> [警告] 在 {exp_name} 中未找到测试相机，将使用训练相机进行评估。")
                test_cameras = scene.getTrainCameras()
            if not test_cameras:
                print(f"  -> [错误] 在 {exp_name} 中也未找到训练相机，跳过2D评估。")
                return computed_metrics
            
            psnr_list, ssim_list = [], []
            background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device="cuda")
            
            for idx, camera in enumerate(tqdm(test_cameras, desc=f"  渲染评估 {exp_name}", leave=False, ncols=80)):
                render_pkg = render(camera, gaussians, pipe_params, background)
                rendered_img = render_pkg["render"].clamp(0.0, 1.0)
                gt_img = camera.original_image.clamp(0.0, 1.0).to("cuda")

                if vutils and hasattr(args, 'debug_save_path') and args.debug_save_path:
                    scene_name_safe = os.path.basename(os.path.dirname(model_path))
                    current_debug_dir = os.path.join(args.debug_save_path, scene_name_safe)
                    os.makedirs(current_debug_dir, exist_ok=True)
                    cam_id = getattr(camera, 'uid', idx)
                    exp_tag = exp_name.replace("exp", "E")
                    render_save_path = os.path.join(current_debug_dir, f"{exp_tag}__render__cam_{cam_id}.png")
                    gt_save_path = os.path.join(current_debug_dir, f"{exp_tag}__gt__cam_{cam_id}.png")
                    vutils.save_image(rendered_img, render_save_path)
                    vutils.save_image(gt_img.to("cpu"), gt_save_path)

                rendered_np = (rendered_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                psnr_list.append(psnr_sk(gt_np, rendered_np, data_range=255))
                ssim_list.append(ssim_sk(gt_np, rendered_np, channel_axis=-1, data_range=255))

            if psnr_list: computed_metrics["Test_PSNR"] = np.mean(psnr_list)
            if ssim_list: computed_metrics["Test_SSIM"] = np.mean(ssim_list)
            
    except Exception as e:
        import traceback
        print(f"  -> ❌ 在评估 {exp_name} 时出错: {e}")
        traceback.print_exc()
        
    return computed_metrics

# ==============================================================================
#                       3. 绘图函数 (保持不变)
# ==============================================================================
def plot_combined_figure(all_progress_data: dict, layout: list, output_path: str):
    """将单个场景下所有实验的训练过程数据绘制到一张图中。"""
    if not any(df is not None and not df.empty for df in all_progress_data.values()):
        return
    print(f"🎨 正在为场景 {os.path.basename(output_path).replace('comparison_figure_','').replace('.png','')} 生成对比图...")
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
    print(f"  ✅ 对比图已保存: {output_path}")

# ==============================================================================
#                       4. 全局主控函数 (配置保持不变)
# ==============================================================================
def main(args):
    """
    主函数，自动发现并分析所有预定义场景下的多个实验组结果，
    并生成每个场景的独立报告以及最终的全局对比总表。
    """
    SCENES = [
        "electro", "delivery_area", "pipes", "courtyard", "facade", "kicker", "meadow", "office",
        "nerf_360/bicycle", "nerf_360/bonsai", "nerf_360/counter", "nerf_360/garden",
        "nerf_360/kitchen", "nerf_360/room", "nerf_360/stump",
        "dtu/scan1", "dtu/scan4", "dtu/scan9", "dtu/scan10", "dtu/scan11", "dtu/scan12", "dtu/scan13", "dtu/scan15", "dtu/scan23", "dtu/scan24",
        "dtu/scan29", "dtu/scan32", "dtu/scan33", "dtu/scan34", "dtu/scan48", "dtu/scan49", "dtu/scan62", "dtu/scan75", "dtu/scan77", "dtu/scan110",
        "dtu/scan114", "dtu/scan118",
        "nerf_synthetic/chair", "nerf_synthetic/drums", "nerf_synthetic/ficus", "nerf_synthetic/hotdog",
        "nerf_synthetic/lego", "nerf_synthetic/materials", "nerf_synthetic/mic", "nerf_synthetic/ship"
    ]
    EXPERIMENT_GROUPS = {
        "Normal_MultiSave": "PUBLICATION_NORMAL_EXPERIMENTS_MULTISAVE",
    }
    OUTPUT_DIR = "MASTER_ANALYSIS_OUTPUT_BEST_ONLY" # 新的输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📊 所有分析结果将保存至: {OUTPUT_DIR}")

    debug_output_dir = os.path.join(OUTPUT_DIR, "debug_renders")
    os.makedirs(debug_output_dir, exist_ok=True)
    print(f"🖼️ 调试渲染图像将保存至: {debug_output_dir}")
    args.debug_save_path = debug_output_dir

    master_results = []

    for scene_name in tqdm(SCENES, desc="总进度：处理所有场景"):
        scene_name_safe = scene_name.replace('/', '_')
        print(f"\n{'='*80}\n场景: {scene_name}\n{'='*80}")
        
        scene_final_metrics = []
        scene_progress_data = {}

        for group_alias, group_base_dir in EXPERIMENT_GROUPS.items():
            scene_parent_dir = os.path.join(group_base_dir, scene_name_safe)
            if not os.path.exists(scene_parent_dir):
                print(f"  -> [跳过] 在 '{group_base_dir}' 中未找到场景 '{scene_name_safe}' 的目录。")
                continue

            sub_experiments = sorted([
                os.path.join(scene_parent_dir, d) for d in os.listdir(scene_parent_dir)
                if os.path.isdir(os.path.join(scene_parent_dir, d)) and os.path.exists(os.path.join(scene_parent_dir, d, "cfg_args"))
            ])
            if not sub_experiments:
                print(f"  -> [警告] 在 '{scene_parent_dir}' 中未找到任何有效的子实验。")
                continue
            
            print(f"  -> 发现 {len(sub_experiments)} 个来自 '{group_alias}' 的实验: {[os.path.basename(p) for p in sub_experiments]}")

            for exp_path in sub_experiments:
                exp_name = os.path.basename(exp_path)
                progress_df = parse_csv_log(exp_path, args.csv_log_name)
                unique_exp_name = f"{group_alias}_{exp_name}"
                scene_progress_data[unique_exp_name] = progress_df
                
                args.parent_dir = scene_parent_dir
                # [核心修改] 调用新函数
                computed_metrics = evaluate_best_model(exp_path, args) 
                
                # 从日志中找到最佳PSNR对应的行
                best_psnr_row = {}
                if progress_df is not None and not progress_df.empty and 'Test_PSNR' in progress_df.columns:
                    best_idx = progress_df['Test_PSNR'].idxmax()
                    best_psnr_row = progress_df.loc[best_idx].to_dict()

                combined_metrics = {
                    "Scene": scene_name, "Method_Group": group_alias, "Experiment": exp_name,
                    "Best_Iter": best_psnr_row.get("Iteration"),
                    **computed_metrics
                }
                scene_final_metrics.append(combined_metrics)

        if scene_final_metrics:
            df_scene = pd.DataFrame(scene_final_metrics).set_index('Experiment')
            display_columns = [col for col in args.table_columns if col in df_scene.columns]
            df_display = df_scene[display_columns].copy()
            for col in df_display.columns:
                if pd.api.types.is_numeric_dtype(df_display[col]) and df_display[col].notna().any():
                    df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            
            table_path = os.path.join(OUTPUT_DIR, f"comparison_table_{scene_name_safe}.txt")
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(f"📊 场景 [{scene_name}] 指标对比表 ('best' 模型) 📊\n{'='*80}\n")
                f.write(df_display.to_string())
                f.write(f"\n{'='*80}")
            print(f"  📄 单场景对比表已保存: {table_path}")

            plot_path = os.path.join(OUTPUT_DIR, f"comparison_figure_{scene_name_safe}.png")
            plot_combined_figure(scene_progress_data, args.plot_layout, plot_path)
            master_results.extend(scene_final_metrics)
        else:
            print(f"  -> [信息] 场景 '{scene_name}' 没有找到任何可供分析的实验结果。")

    if not master_results:
        print("\n❌ 所有场景均未找到任何有效的实验结果，无法生成最终总表。")
        return

    print(f"\n{'='*80}\n🏆 开始生成全局对比总表 ('best' 模型)...\n{'='*80}")
    df_master = pd.DataFrame(master_results)
    df_master['Display_Name'] = df_master.apply(lambda row: f"{row['Method_Group']}_{row['Experiment']}", axis=1)
    pivot_metrics = ['Test_PSNR', 'Test_SSIM', 'Geom_Outlier_Ratio', 'Geom_Normal_Variance', 'Total_Points', 'Best_Iter']
    pivot_metrics = [m for m in pivot_metrics if m in df_master.columns]

    if not pivot_metrics:
        print("❌ 找不到任何可用于生成透视表的关键指标。")
        return

    df_pivot = df_master.pivot_table(index='Scene', columns='Display_Name', values=pivot_metrics)
    df_pivot = df_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)

    for col in df_pivot.columns:
        if pd.api.types.is_numeric_dtype(df_pivot[col]):
            if 'Ratio' in col[1] or 'SSIM' in col[1] or 'Variance' in col[1]:
                df_pivot[col] = df_pivot[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            elif 'PSNR' in col[1]:
                df_pivot[col] = df_pivot[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
            elif 'Points' in col[1] or 'Iter' in col[1]:
                 df_pivot[col] = df_pivot[col].apply(lambda x: f'{int(x):,}' if pd.notna(x) else 'N/A')

    master_table_string = df_pivot.to_string()
    
    print("\n" + "="*120)
    print(" " * 45 + "🏆 全局实验结果对比总表 ('best' 模型) 🏆")
    print("="*120)
    print(master_table_string)
    print("="*120)

    master_table_path = os.path.join(OUTPUT_DIR, "master_comparison_table_best_only.txt")
    with open(master_table_path, 'w', encoding='utf-8') as f:
        f.write(f"🏆 全局实验结果对比总表 ('best' 模型) 🏆\n{'='*120}\n{master_table_string}")
    print(f"📄 全局对比总表 (文本格式) 已保存至: {OUTPUT_DIR}")

    df_master.to_csv(os.path.join(OUTPUT_DIR, "master_raw_data_best_only.csv"), index=False, encoding='utf-8-sig')
    df_pivot.to_csv(os.path.join(OUTPUT_DIR, "master_pivot_table_best_only.csv"), encoding='utf-8-sig')
    print(f"📄 全局原始数据和透视表 (CSV格式) 已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    args = Namespace()
    args.skip_2d_eval = False
    args.csv_log_name = "training_log.csv"
    args.plot_layout = [
        {'ax_idx': 0, 'column': 'Test_PSNR', 'title': '测试集 PSNR', 'ylabel': 'PSNR (dB)'},
        {'ax_idx': 1, 'column': 'Total_Loss', 'title': '总损失', 'ylabel': 'Loss (log scale)', 'log_y': True},
    ]
    args.table_columns = [
        'Test_PSNR', 'Test_SSIM', 'Geom_Outlier_Ratio',
        'Geom_Normal_Variance', 'Total_Points', 'Best_Iter',
    ]
    main(args)
    print("\n🎉🎉🎉 所有场景分析完毕！🎉🎉🎉")