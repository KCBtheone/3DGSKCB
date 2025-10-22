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
#                      1. 数据解析与指标计算函数
# ==============================================================================
def load_config(cfg_path: str) -> dict:
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
    log_path = os.path.join(exp_path, log_filename)
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0: return None
    try:
        df = pd.read_csv(log_path)
        return df if not df.empty else None
    except Exception as e:
        print(f"    - ❌ 读取 {log_path} 时出错: {e}")
        return None

def load_gt_pcd(gt_path, voxel_size):
    try:
        pcd = o3d.io.read_point_cloud(gt_path)
        if not pcd.has_points(): return None
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        print(f"    -> [DEBUG] 真值点云 '{os.path.basename(gt_path)}' 加载成功: 共 {len(downsampled_pcd.points)} 个点 (下采样后)")
        return downsampled_pcd
    except Exception as e:
        print(f"    -> ❌ 加载真值点云 '{gt_path}' 时出错: {e}")
        return None

def align_pcd_with_icp(source_pcd, target_pcd, threshold, initial_transform=np.identity(4), max_iteration=200):
    print("    -> [INFO] 正在使用ICP算法进行点云精细对齐...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    fitness = reg_p2p.fitness
    inlier_rmse = reg_p2p.inlier_rmse
    print(f"    -> [DEBUG] ICP对齐结果: Fitness = {fitness:.4f}, Inlier RMSE = {inlier_rmse:.4f}")
    if fitness < 0.6:
        print("    -> [WARNING] ICP Fitness值较低，精细对齐效果可能不佳！")
    return reg_p2p.transformation

def calculate_3d_metrics(pred_pcd, gt_pcd, f1_threshold):
    dists_pred_to_gt = pred_pcd.compute_point_cloud_distance(gt_pcd)
    dists_gt_to_pred = gt_pcd.compute_point_cloud_distance(pred_pcd)
    
    # [DEBUG] 打印一些距离统计，检查尺度
    mean_dist_pred_to_gt = np.mean(np.asarray(dists_pred_to_gt))
    mean_dist_gt_to_pred = np.mean(np.asarray(dists_gt_to_pred))
    print(f"    -> [DEBUG] Pred->GT 平均距离: {mean_dist_pred_to_gt:.6f} 米")
    print(f"    -> [DEBUG] GT->Pred 平均距离: {mean_dist_gt_to_pred:.6f} 米")
    
    chamfer_dist_l2 = (np.mean(np.asarray(dists_pred_to_gt)**2) + np.mean(np.asarray(dists_gt_to_pred)**2))
    precision = np.sum(np.asarray(dists_pred_to_gt) < f1_threshold) / len(dists_pred_to_gt)
    recall = np.sum(np.asarray(dists_gt_to_pred) < f1_threshold) / len(dists_gt_to_pred)
    print(f"    -> [DEBUG] F1-Score阈值 = {f1_threshold} 米")
    print(f"    -> [DEBUG] F1-Score组成: Precision = {precision:.4f}, Recall = {recall:.4f}")
    f1_score = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    return {"3D_Chamfer_L2": chamfer_dist_l2, "3D_F1-Score": f1_score}

# ==============================================================================
#                      2. 核心分析流程
# ==============================================================================

@torch.no_grad()
def evaluate_final_model(model_path: str, args: Namespace, gt_pcd_downsampled, alignment_transform: np.ndarray):
    exp_name = os.path.basename(model_path)
    print(f"\n{'─'*20} [INFO] 开始评估模型: {exp_name} {'─'*20}")
    
    computed_metrics = {}
    debug_pcd_dir = os.path.join(args.parent_dir, "_debug_pcds")
    debug_2d_dir = os.path.join(args.parent_dir, "_debug_2d_eval")
    os.makedirs(debug_pcd_dir, exist_ok=True)
    os.makedirs(debug_2d_dir, exist_ok=True)
    
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
        iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        if iteration is None: raise FileNotFoundError("未找到任何迭代点云")
        
        gaussians = GaussianModel(sh_degree=model_params.sh_degree)
        ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        gaussians.load_ply(ply_path)
        print(f"    -> [DEBUG] 模型加载成功 (迭代次数: {iteration}, 点数: {gaussians.get_xyz.shape[0]})")
        
        pred_points_colmap = gaussians.get_xyz.detach().cpu().numpy()
        pred_pcd_raw = o3d.geometry.PointCloud()
        pred_pcd_raw.points = o3d.utility.Vector3dVector(pred_points_colmap)
        
        if gt_pcd_downsampled is not None and alignment_transform is not None:
            print("  -> 步骤3: 计算3D几何指标...")
            
            # 1. 粗略对齐矩阵 (Colmap/GS -> World)
            initial_transform_inv = alignment_transform
            
            # [DEBUG SAVE] 保存粗略对齐后的点云用于肉眼检查
            pred_pcd_coarse_aligned = copy.deepcopy(pred_pcd_raw)
            pred_pcd_coarse_aligned.transform(initial_transform_inv)
            o3d.io.write_point_cloud(os.path.join(debug_pcd_dir, f"{exp_name}_pred_coarse_aligned.ply"), pred_pcd_coarse_aligned)
            print(f"    -> [DEBUG] 粗略对齐点云已保存 (检查是否与GT重叠): {exp_name}_pred_coarse_aligned.ply")
            
            # 2. 对预测点云进行下采样 (使用原始未对齐的点云进行下采样)
            pred_pcd_down_raw = pred_pcd_raw.voxel_down_sample(voxel_size=args.voxel_size)

            # 3. ICP 精细对齐 (更鲁棒的流程: 使用 coarse transform 作为 initial guess)
            print("    -> [INFO] 正在使用粗略对齐作为初始位姿，运行ICP精细对齐...")
            final_transformation_total = align_pcd_with_icp(
                copy.deepcopy(pred_pcd_down_raw), gt_pcd_downsampled, 
                threshold=args.icp_threshold,
                initial_transform=initial_transform_inv  # <-- 关键更改: 使用粗略对齐矩阵作为初始猜测
            )
            
            # 4. 将最终的总变换 (粗略+精细) 应用到原始高分辨率点云上
            pred_pcd_raw.transform(final_transformation_total)
            
            # [DEBUG SAVE] 保存最终对齐的点云用于指标计算和可视化
            o3d.io.write_point_cloud(os.path.join(debug_pcd_dir, f"{exp_name}_pred_final_aligned.ply"), pred_pcd_raw)
            print(f"    -> [DEBUG] 用于可视化的最终对齐点云已保存至: {exp_name}_pred_final_aligned.ply")
            
            metrics_3d = calculate_3d_metrics(pred_pcd_raw, gt_pcd_downsampled, args.f1_threshold)
            computed_metrics.update(metrics_3d)

        else:
            print("  -> 步骤3: 跳过3D几何指标计算 (缺少真值点云或对齐矩阵)。")

        if not args.skip_2d_eval:
            print("  -> 步骤4: 计算2D渲染指标...")
            
            # --- [FIX] 修正 PipelineParams 初始化 ---
            temp_parser = ArgumentParser(description="Temp parser for pipeline")
            pipe_params = PipelineParams(temp_parser).extract(args_defaults)
            
            scene = Scene(model_params, gaussians, shuffle=False)
            test_cameras = scene.getTestCameras()
            if not test_cameras: test_cameras = scene.getTrainCameras()
            if not test_cameras:
                print("    - ❌ 错误: 找不到任何相机用于评估。")
                return computed_metrics

            psnr_list, ssim_list = [], []
            background = torch.tensor([1,1,1] if model_params.white_background else [0,0,0], dtype=torch.float32, device="cuda")
            
            debug_2d_save_done = False # 用于控制只保存一张调试图
            
            for camera in tqdm(test_cameras, desc=f"  渲染 {exp_name}", leave=False):
                render_pkg = render(camera, gaussians, pipe_params, background)
                rendered_img = render_pkg["render"].clamp(0.0, 1.0)
                gt_img = camera.original_image.clamp(0.0, 1.0).to("cuda")
                
                rendered_np = (rendered_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # [DEBUG SAVE] 只保存第一张图进行检查
                if not debug_2d_save_done:
                    plt.imsave(os.path.join(debug_2d_dir, f"{exp_name}_rendered_{camera.fid}.png"), rendered_np / 255.0)
                    plt.imsave(os.path.join(debug_2d_dir, f"{exp_name}_gt_{camera.fid}.png"), gt_np / 255.0)
                    print(f"    -> [DEBUG] 渲染图和真值图已保存至: {exp_name}_rendered/gt_{camera.fid}.png (检查是否全黑或全白)")
                    debug_2d_save_done = True
                
                psnr_list.append(psnr_sk(gt_np, rendered_np, data_range=255))
                ssim_list.append(ssim_sk(gt_np, rendered_np, channel_axis=2, data_range=255))
            
            if psnr_list: computed_metrics["Test_PSNR"] = np.mean(psnr_list)
            if ssim_list: computed_metrics["Test_SSIM"] = np.mean(ssim_list)
            
    except Exception as e:
        print(f"  -> ❌ 在评估 {exp_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        
    return computed_metrics

# ==============================================================================
#                      3. 绘图与主控函数
# ==============================================================================
def plot_combined_figure(all_progress_data: dict, layout: list, output_path: str):
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
    parent_dir = args.parent_dir
    # --- 从父目录正确推断场景名 ---
    scene_name_for_all_experiments = os.path.basename(parent_dir.rstrip('/'))
    print(f"ℹ️ 从父目录推断出当前所有实验均属于场景: [{scene_name_for_all_experiments}]")
    
    print(f"🔍 开始在父目录中搜索子实验: {parent_dir}")
    sub_experiments = sorted([
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
        if os.path.isdir(os.path.join(parent_dir, d)) and os.path.exists(os.path.join(parent_dir, d, "cfg_args"))
    ])
    
    if not sub_experiments:
        print(f"❌ 在 '{parent_dir}' 中未找到任何有效的子实验。")
        return
    print(f"✅ 找到 {len(sub_experiments)} 个实验: {[os.path.basename(p) for p in sub_experiments]}")
    
    print(f"🔄 正在从 '{args.alignment_file}' 加载对齐矩阵...")
    if not os.path.exists(args.alignment_file):
        print(f"❌ 错误: 对齐文件 '{args.alignment_file}' 不存在！")
        sys.exit(1)
    with open(args.alignment_file, 'r') as f:
        alignments = json.load(f)
    print("    -> [INFO] 对齐文件加载成功。")

    gt_pcd_downsampled = load_gt_pcd(args.gt_pcd_path, voxel_size=args.voxel_size)
    
    debug_pcd_dir = os.path.join(parent_dir, "_debug_pcds")
    os.makedirs(debug_pcd_dir, exist_ok=True)
    
    if gt_pcd_downsampled is not None:
        o3d.io.write_point_cloud(os.path.join(debug_pcd_dir, "ground_truth_world_frame.ply"), gt_pcd_downsampled)
        print(f"    -> [DEBUG] 真值点云 (world frame) 已保存至: {debug_pcd_dir}/ground_truth_world_frame.ply")
    else:
        print(f"⚠️ 警告: 无法加载真值点云，3D评估将被跳过。")

    all_final_metrics = []
    all_progress_data = {}
    
    for exp_path in sub_experiments:
        exp_name = os.path.basename(exp_path)
        # --- 使用正确的场景名 ---
        scene_name = scene_name_for_all_experiments
        print(f"\n{'='*30} 主循环: 开始处理 {exp_name} (场景: {scene_name}) {'='*30}")
        
        progress_df = parse_csv_log(exp_path, args.csv_log_name)
        all_progress_data[exp_name] = progress_df
        
        alignment_transform = None
        if scene_name not in alignments:
            print(f"    - ⚠️ 警告: 在 '{args.alignment_file}' 中找不到场景 '{scene_name}' 的对齐矩阵，将跳过此实验的3D评估。")
        else:
            alignment_transform = np.array(alignments[scene_name])
            print(f"    -> [INFO] 成功为场景 '{scene_name}' 找到对齐矩阵。")
            
            # [DEBUG PRINT] 打印对齐矩阵
            print("    -> [DEBUG] 对齐矩阵 (4x4):")
            print(alignment_transform)

        computed_metrics = evaluate_final_model(exp_path, args, gt_pcd_downsampled, alignment_transform)
        
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
                 df_display.loc[:, col] = df_display[col].map('{:.4f}'.format)
        
        print("\n" + "="*80)
        print("                    📊 最终指标对比总表 (V-Final) 📊")
        print("="*80)
        print(df_display.to_string())
        print("="*80)
        
        summary_path = os.path.join(parent_dir, "comparison_summary_final.json")
        df.to_json(summary_path, orient='index', indent=4)
        print(f"📄 最终指标数据已保存至: {summary_path}")
            
    plot_combined_figure(all_progress_data, args.plot_layout, os.path.join(parent_dir, "comparison_figure_final.png"))

if __name__ == "__main__":
    parser = ArgumentParser(description="用于3DGS实验的最终对比分析脚本 (集成精确位姿对齐, 修复多项错误)。")
    parser.add_argument("parent_dir", type=str, help="包含单个场景下多个子实验的父目录路径 (例如 NORMAL_EXPERIMENTS/electro)。")
    parser.add_argument("--gt_pcd_path", type=str, required=True, help="该场景对应的、已合并的单一真值点云(.ply)的文件路径。")
    parser.add_argument("--alignment_file", type=str, required=True, help="包含所有场景对齐变换矩阵的JSON文件 (例如 eth3d_alignments_v2.json)。")
    parser.add_argument("--skip_2d_eval", action="store_true", help="跳过耗时的2D渲染评估。")
    parser.add_argument("--f1_threshold", type=float, default=0.05, help="3D F1-score的距离阈值(米)。")
    parser.add_argument("--csv_log_name", type=str, default="training_log.csv", help="训练日志CSV文件名。")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="用于点云下采样的大小(米)。")
    parser.add_argument("--icp_threshold", type=float, default=0.1, help="ICP配准的最大对应距离(米)。")
    
    args = parser.parse_args()
    
    args.plot_layout = [
        {'ax_idx': 0, 'column': 'Train_PSNR',  'title': '训练过程 PSNR', 'ylabel': 'PSNR (dB)'},
        {'ax_idx': 1, 'column': 'Total_Loss',  'title': '总损失', 'ylabel': 'Loss (log scale)', 'log_y': True},
    ]
    args.table_columns = [
        'Test_PSNR', 'Test_SSIM', '3D_Chamfer_L2', '3D_F1-Score', 'Train_PSNR', 'Total_Points'
    ]
    
    main(args)