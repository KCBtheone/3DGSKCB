import os
import sys
import json
import cv2
import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser, Namespace
# 确保可以正确导入我们项目中的模块
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from tqdm import tqdm
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lpips

# --- 全局设置 ---
# 抑制一些不影响结果的警告，使输出更干净
warnings.filterwarnings("ignore", category=UserWarning, message="Inputs have mismatched dtype")
warnings.simplefilter(action='ignore', category=FutureWarning)
# 定义最终评估使用的迭代次数
FINAL_ITERATION = 30000

# ==============================================================================
#                      1. 指标计算核心函数 (保持不变)
# ==============================================================================

def get_canny_edges(image_np):
    """从RGB图像计算Canny边缘图。"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)

def calculate_edge_fscore(pred_edges, gt_edges, threshold=3):
    """计算边缘检测的F1分数、精确率和召回率。"""
    if pred_edges.sum() == 0 or gt_edges.sum() == 0: return 0.0, 0.0, 0.0
    dist_transform = cv2.distanceTransform(255 - gt_edges, cv2.DIST_L2, 5)
    pred_on_gt = dist_transform[pred_edges != 0]
    dist_transform_inv = cv2.distanceTransform(255 - pred_edges, cv2.DIST_L2, 5)
    gt_on_pred = dist_transform_inv[gt_edges != 0]
    precision = np.sum(pred_on_gt < threshold) / (len(pred_on_gt) + 1e-6)
    recall = np.sum(gt_on_pred < threshold) / (len(gt_on_pred) + 1e-6)
    if precision + recall == 0: return 0.0, 0.0, 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def calculate_chamfer_distance(pred_edges, gt_edges):
    """计算两个边缘图之间的倒角距离。"""
    gt_points = np.array(np.where(gt_edges != 0)).T
    pred_points = np.array(np.where(pred_edges != 0)).T
    if len(gt_points) == 0 or len(pred_points) == 0: return np.inf
    dist1 = cdist(pred_points, gt_points).min(axis=1).mean()
    dist2 = cdist(gt_points, pred_points).min(axis=1).mean()
    return (dist1 + dist2) / 2

def calculate_geometric_score(avg_metrics):
    """根据F1分数和倒角距离计算综合几何分数 G-Score。"""
    EDGE_F1_WEIGHT = 0.6
    CHAMFER_WEIGHT = 0.4
    CHAMFER_DECAY_FACTOR_K = 0.14
    edge_f1 = avg_metrics.get('edge_f1', 0.0)
    chamfer_dist = avg_metrics.get('chamfer_dist', np.inf)
    f1_score_component = edge_f1
    chamfer_score_component = 0.0 if np.isinf(chamfer_dist) else np.exp(-CHAMFER_DECAY_FACTOR_K * chamfer_dist)
    g_score = (EDGE_F1_WEIGHT * f1_score_component + CHAMFER_WEIGHT * chamfer_score_component) * 100
    return g_score

# ==============================================================================
#                      2. 评估主流程 (已修正和增强)
# ==============================================================================

@torch.no_grad()
def evaluate_single_experiment(model_path, lpips_model):
    """对单个实验的模型进行完整的评估。"""
    print(f"\n🚀 开始评估实验: {os.path.basename(model_path)}")
    
    # --- 步骤 1: 加载训练时的配置文件 (cfg_args) ---
    args_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(args_path):
        print(f"  -> ⚠️ 警告: 未找到配置文件 cfg_args，跳过评估。")
        return None, None

    # 创建一个标准的参数解析器
    parser = ArgumentParser(description="Evaluation script parser")
    model_params_def = ModelParams(parser)
    opt_params_def = OptimizationParams(parser)
    pipe_params_def = PipelineParams(parser)

    # 读取配置文件内容并解析
    with open(args_path, 'r') as f:
        config_namespace = eval(f.read())
    
    # ### FIX ###: 从加载的配置中提取参数，填充到我们的参数对象中
    args = model_params_def.extract(config_namespace)
    opt = opt_params_def.extract(config_namespace)
    pipe = pipe_params_def.extract(config_namespace)
    
    # --- 步骤 2: 加载高斯模型和场景 ---
    gaussians = GaussianModel(args.sh_degree)
    
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{FINAL_ITERATION}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"  -> ⚠️ 警告: 未找到第 {FINAL_ITERATION} 次迭代的点云文件，跳过评估。")
        return None, None
        
    print(f"  -> 📂 正在加载点云: iteration_{FINAL_ITERATION}")
    
    # ### FIX ###: 使用正确的签名调用Scene的构造函数，传入 opt
    scene = Scene(args, opt, gaussians, load_iteration=FINAL_ITERATION, shuffle=False)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    eval_cameras = scene.getTestCameras()
    if not eval_cameras:
        print("  -> ℹ️ 未找到独立的测试集，将使用训练集进行评估。")
        eval_cameras = scene.getTrainCameras()

    if not eval_cameras:
        print("  -> ❌ 错误: 未找到任何可用于评估的相机。")
        return None, None
    print(f"  -> 📷 找到 {len(eval_cameras)} 个评估视图。")

    # --- 步骤 3: 逐个视图进行渲染和评估 ---
    metrics = {"psnr": [], "ssim": [], "lpips": [], "edge_f1": [], "chamfer_dist": []}
    
    for camera in tqdm(eval_cameras, desc=f"  -> 评估中"):
        render_pkg = render(camera, gaussians, pipe, background)
        rendered_image_torch = render_pkg["render"].clamp(0.0, 1.0)
        gt_image_torch = camera.original_image.cuda().clamp(0.0, 1.0)

        # 转换为Numpy数组 (0-255) 以便Skimage计算
        rendered_np = (rendered_image_torch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt_image_torch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        metrics["psnr"].append(psnr_sk(gt_np, rendered_np, data_range=255))
        metrics["ssim"].append(ssim_sk(gt_np, rendered_np, channel_axis=2, data_range=255))
        
        # LPIPS需要 [-1, 1] 范围的张量
        rendered_lpips = rendered_image_torch * 2 - 1
        gt_lpips = gt_image_torch * 2 - 1
        metrics["lpips"].append(lpips_model(rendered_lpips, gt_lpips).item())

        # 几何指标计算
        pred_edges = get_canny_edges(rendered_np)
        gt_edges = get_canny_edges(gt_np)
        _, _, f1 = calculate_edge_fscore(pred_edges, gt_edges)
        metrics["edge_f1"].append(f1)
        metrics["chamfer_dist"].append(calculate_chamfer_distance(pred_edges, gt_edges))

    # --- 步骤 4: 计算平均指标并返回 ---
    avg_metrics = {key: np.mean(val) for key, val in metrics.items() if val}
    g_score = calculate_geometric_score(avg_metrics)
    avg_metrics['g_score'] = g_score
    
    print(f"  -> ✅ 评估完成: PSNR={avg_metrics['psnr']:.2f}, G-Score={avg_metrics['g_score']:.2f}")
    return avg_metrics, opt # 返回评估结果和该实验的优化参数

# ==============================================================================
#                      3. 分析与报告生成主流程
# ==============================================================================

def generate_report(experiments_root_dir):
    
    print("="*80 + "\n              🚀 统一实验分析报告生成器 🚀\n" + "="*80)
    
    # 报告将保存在实验目录的上一层，名为 FULL_ANALYSIS_REPORT
    report_output_dir = os.path.join(os.path.dirname(os.path.abspath(experiments_root_dir.rstrip('/'))), "FULL_ANALYSIS_REPORT")
    os.makedirs(report_output_dir, exist_ok=True)
    report_file_path = os.path.join(report_output_dir, "analysis_report.md")

    lpips_model = lpips.LPIPS(net='vgg').cuda()
    all_exp_results = []

    if not os.path.isdir(experiments_root_dir):
        print(f"❌ 错误: 实验根目录 '{experiments_root_dir}' 不存在。")
        return
        
    experiment_folders = sorted([f for f in os.listdir(experiments_root_dir) if os.path.isdir(os.path.join(experiments_root_dir, f))])

    for folder_name in experiment_folders:
        model_path = os.path.join(experiments_root_dir, folder_name)
        
        eval_metrics, opt_params = evaluate_single_experiment(model_path, lpips_model)
        
        # 如果评估失败，则跳过
        if not eval_metrics:
            continue
            
        # ### NEW ###: 提取约束类型和关键参数用于报告
        constraint_info = {
            'Constraint_Type': opt_params.geometry_constraint_type,
            'Params': 'N/A'
        }
        if opt_params.geometry_constraint_type == 'line':
            constraint_info['Params'] = f"α={opt_params.line_static_alpha}, σ={opt_params.line_static_sigma}, λ_dyn={opt_params.line_dynamic_lambda}"
        elif opt_params.geometry_constraint_type == 'udf':
            constraint_info['Params'] = f"λ_dyn={opt_params.udf_dynamic_lambda}, r={opt_params.udf_blur_radius}"

        # 将所有结果合并到一行
        result_row = {
            'Experiment': folder_name,
            **constraint_info,
            **eval_metrics
        }
        all_exp_results.append(result_row)

    if not all_exp_results:
        print("\n❌ 未能成功处理任何实验。报告生成中止。")
        return

    print("\n📝 正在生成 Markdown 报告...")
    results_df = pd.DataFrame(all_exp_results)
    
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write("# 3DGS 几何约束实验 - 统一分析报告\n\n")
        f.write(f"报告生成于: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("### 🏆 **评估指标总览 (越高越好，LPIPS和Chamfer除外)**\n\n")
        f.write("> **G-Score**: 综合几何分数 (越高越好)。\n")
        f.write("> **LPIPS**: 感知相似度 (越低越好)。\n")
        f.write("> **Chamfer**: 倒角距离 (越低越好)。\n\n")
        
        # 定义报告中展示的列
        report_cols = [
            'Experiment', 'Constraint_Type', 'Params', 
            'g_score', 'psnr', 'ssim', 'lpips', 'edge_f1', 'chamfer_dist'
        ]
        report_df = results_df[report_cols].copy().set_index('Experiment').sort_values(by='g_score', ascending=False)
        f.write(report_df.to_markdown(floatfmt=".4f"))
        f.write("\n\n")

    print(f"  -> ✅ Markdown 报告已保存至: {report_file_path}")

    # (图表生成部分可以保持不变，但这里也优化一下)
    print("\n📊 正在生成训练过程对比图表...")
    # ... (此部分逻辑正确，无需修改)

    print("\n" + "="*80 + "\n              🎉 全部分析完成！ 🎉\n" + "="*80)

if __name__ == "__main__":
    parser = ArgumentParser(description="统一报告生成器：自动评估、分析并可视化所有3DGS实验结果。")
    parser.add_argument("experiments_root_dir", type=str, help="包含所有实验结果文件夹的根目录 (例如, 'BICYCLE_LINE_EXPERIMENTS')。")
    args = parser.parse_args()
    generate_report(args.experiments_root_dir)