import os
import sys
import json
import cv2
import numpy as np
import torch
from argparse import ArgumentParser
from arguments import ModelParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.camera_utils import cameraList_from_camInfos
from tqdm import tqdm
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
import warnings

# Suppress a specific warning from scikit-image
warnings.filterwarnings("ignore", category=UserWarning, message="Inputs have mismatched dtype")

# ==============================================================================
#                      1. 指标计算的核心函数
# ==============================================================================

def get_canny_edges(image_np):
    """使用Canny算法提取边缘"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def calculate_edge_fscore(pred_edges, gt_edges, threshold=3):
    """
    计算边缘F-score，允许一定像素的容差。
    使用距离变换来高效实现。
    """
    if pred_edges.sum() == 0 or gt_edges.sum() == 0:
        return 0.0, 0.0, 0.0 # Precision, Recall, F1

    # 计算从GT边缘到每个预测边缘像素的距离
    dist_transform = cv2.distanceTransform(255 - gt_edges, cv2.DIST_L2, 5)
    pred_on_gt = dist_transform[pred_edges != 0]

    # 计算从预测边缘到每个GT边缘像素的距离
    dist_transform_inv = cv2.distanceTransform(255 - pred_edges, cv2.DIST_L2, 5)
    gt_on_pred = dist_transform_inv[gt_edges != 0]

    # 计算Precision和Recall
    precision = np.sum(pred_on_gt < threshold) / len(pred_on_gt)
    recall = np.sum(gt_on_pred < threshold) / len(gt_on_pred)
    
    # 计算F1-score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def calculate_chamfer_distance(pred_edges, gt_edges):
    """计算倒角距离"""
    gt_points = np.array(np.where(gt_edges != 0)).T
    pred_points = np.array(np.where(pred_edges != 0)).T

    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.inf

    # 计算 pred_points 到 gt_points 的距离
    dist1 = cdist(pred_points, gt_points).min(axis=1).mean()
    # 计算 gt_points 到 pred_points 的距离
    dist2 = cdist(gt_points, pred_points).min(axis=1).mean()

    return (dist1 + dist2) / 2

def get_lsd_lines(image_np):
    """使用LSD算法提取直线"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(gray)
    if lines is None:
        return []
    return lines.squeeze(1) # [N, 4] -> x1, y1, x2, y2

def calculate_lses_metrics(pred_lines, gt_lines, dist_thresh=5, angle_thresh=5, overlap_thresh=0.5):
    """计算LSES套件: L-Recall, L-Precision, L-Fidelity"""
    if len(gt_lines) == 0:
        # 如果真值没有直线，精确率就是1（如果没有预测直线）或0（如果有预测直线）
        l_precision = 1.0 if len(pred_lines) == 0 else 0.0
        return 0.0, l_precision, (np.inf, np.inf) # Recall, Precision, (Endpoint Err, Angle Err)
    
    if len(pred_lines) == 0:
        return 0.0, 1.0, (np.inf, np.inf) # 如果没预测出直线，召回率为0，精确率为1

    # 为每条GT直线寻找最佳匹配
    matches = []
    matched_pred_indices = set()

    for i, l_gt in enumerate(gt_lines):
        x1_gt, y1_gt, x2_gt, y2_gt = l_gt
        p1_gt, p2_gt = np.array([x1_gt, y1_gt]), np.array([x2_gt, y2_gt])
        mid_gt = (p1_gt + p2_gt) / 2
        vec_gt = p2_gt - p1_gt
        angle_gt = np.rad2deg(np.arctan2(vec_gt[1], vec_gt[0]))
        len_gt = np.linalg.norm(vec_gt)

        best_match = None
        min_dist = np.inf
        best_pred_idx = -1

        for j, l_pred in enumerate(pred_lines):
            if j in matched_pred_indices:
                continue # 已被匹配的预测直线不再参与匹配

            x1_p, y1_p, x2_p, y2_p = l_pred
            p1_p, p2_p = np.array([x1_p, y1_p]), np.array([x2_p, y2_p])
            mid_p = (p1_p + p2_p) / 2
            
            # 1. 距离准则
            dist = np.linalg.norm(mid_gt - mid_p)
            if dist > dist_thresh:
                continue
            
            # 2. 角度准则
            vec_p = p2_p - p1_p
            angle_p = np.rad2deg(np.arctan2(vec_p[1], vec_p[0]))
            angle_diff = 180 - abs(abs(angle_gt - angle_p) - 180) # 考虑180度差异
            if angle_diff > angle_thresh:
                continue
            
            # 3. 重叠准则
            len_p = np.linalg.norm(vec_p)
            if len_p == 0 or len_gt == 0: continue
            
            # 将 l_pred 的端点投影到 l_gt 所在的无限长直线上
            t0 = np.dot(p1_p - p1_gt, vec_gt) / (len_gt**2)
            t1 = np.dot(p2_p - p1_gt, vec_gt) / (len_gt**2)
            
            overlap_interval = (max(0, min(t0, t1)), min(1, max(t0, t1)))
            overlap_len = (overlap_interval[1] - overlap_interval[0]) * len_gt
            
            if overlap_len / len_p < overlap_thresh and overlap_len / len_gt < overlap_thresh:
                continue

            if dist < min_dist:
                min_dist = dist
                best_match = (l_gt, l_pred, angle_diff)
                best_pred_idx = j

        if best_match:
            matches.append(best_match)
            matched_pred_indices.add(best_pred_idx)

    tp = len(matches)
    l_recall = tp / len(gt_lines)
    l_precision = tp / len(pred_lines)
    
    # 计算 L-Fidelity
    if tp > 0:
        endpoint_errors = []
        angle_errors = []
        for l_gt, l_pred, angle_diff in matches:
            p1_gt, p2_gt = l_gt[:2], l_gt[2:]
            p1_p, p2_p = l_pred[:2], l_pred[2:]
            # 确保端点对应正确
            if np.linalg.norm(p1_gt - p1_p) + np.linalg.norm(p2_gt - p2_p) > \
               np.linalg.norm(p1_gt - p2_p) + np.linalg.norm(p2_gt - p1_p):
                p1_p, p2_p = p2_p, p1_p # 交换端点
            
            err = (np.linalg.norm(p1_gt - p1_p) + np.linalg.norm(p2_gt - p2_p)) / 2
            endpoint_errors.append(err)
            angle_errors.append(angle_diff)
            
        avg_endpoint_err = np.mean(endpoint_errors)
        avg_angle_err = np.mean(angle_errors)
        l_fidelity = (avg_endpoint_err, avg_angle_err)
    else:
        l_fidelity = (np.inf, np.inf)

    return l_recall, l_precision, l_fidelity


# ==============================================================================
#                      2. 评估主流程
# ==============================================================================

@torch.no_grad()
def evaluate(model_path):
    # --- 加载模型和场景 ---
    print(f"🚀 开始评估实验: {os.path.basename(model_path)}")
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    
    # 从 cfg_args 加载配置
    args_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(args_path):
        print(f"❌ 错误: 在 {model_path} 中未找到 cfg_args。无法继续评估。")
        return

    with open(args_path, 'r') as f:
        # 使用eval来解析Namespace字符串，这需要信任来源
        config_namespace = eval(f.read())
    
    # 将加载的配置更新到模型参数中
    args = model_params.extract(config_namespace)

    gaussians = GaussianModel(args.sh_degree)
    
    # 寻找最新的检查点
    checkpoints_dir = os.path.join(model_path, "point_cloud")
    latest_iter = -1
    for item in os.listdir(checkpoints_dir):
        if item.startswith("iteration_"):
            try:
                iteration = int(item.split("_")[-1])
                if iteration > latest_iter:
                    latest_iter = iteration
            except:
                continue

    if latest_iter == -1:
        print(f"❌ 错误: 在 {model_path} 中未找到任何训练好的点云。")
        return
        
    print(f"📂 正在加载最新的点云: iteration_{latest_iter}")
    ply_path = os.path.join(checkpoints_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --- 加载测试集相机 ---
    # 我们需要一个临时的、不带gaussians的Scene对象来加载相机
    scene = Scene(args, OptimizationParams(ArgumentParser()), gaussians, load_iteration=-1, shuffle=False)
    test_cameras = scene.getTestCameras()

    if not test_cameras:
        print("⚠️ 警告: 未找到测试集相机，将使用训练集相机进行评估。")
        test_cameras = scene.getTrainCameras()
    if not test_cameras:
        print("❌ 错误: 未找到任何可用于评估的相机。")
        return

    # --- 初始化指标累加器 ---
    metrics = {
        "psnr": [], "ssim": [],
        "edge_f1": [], "chamfer_dist": [],
        "l_recall": [], "l_precision": [],
        "l_endpoint_err": [], "l_angle_err": [],
    }
    
    render_path = os.path.join(model_path, "eval_renders")
    gt_path = os.path.join(model_path, "eval_gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    print(f"🖼️ 渲染图像将保存至: {render_path}")

    # --- 遍历测试集进行评估 ---
    for idx, camera in enumerate(tqdm(test_cameras, desc="Evaluating test set")):
        # 渲染图像
        render_pkg = render(camera, gaussians, {"antialiasing":False}, background)
        rendered_image = render_pkg["render"].clamp(0.0, 1.0)
        
        # 获取真值图像
        gt_image = camera.original_image.clamp(0.0, 1.0)
        
        # 转换格式为 NumPy (H, W, C) uint8
        rendered_np = (rendered_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # 保存图像用于检查
        cv2.imwrite(os.path.join(render_path, f"{idx:04d}.png"), cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(gt_path, f"{idx:04d}.png"), cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))

        # 1. 计算标准指标 PSNR, SSIM
        metrics["psnr"].append(psnr_sk(gt_np, rendered_np, data_range=255))
        metrics["ssim"].append(ssim_sk(gt_np, rendered_np, channel_axis=2, data_range=255))
        
        # 2. 提取边缘用于F-score和倒角距离
        pred_edges = get_canny_edges(rendered_np)
        gt_edges = get_canny_edges(gt_np)

        # 3. 计算边缘F-score
        _, _, f1 = calculate_edge_fscore(pred_edges, gt_edges)
        metrics["edge_f1"].append(f1)
        
        # 4. 计算倒角距离
        metrics["chamfer_dist"].append(calculate_chamfer_distance(pred_edges, gt_edges))
        
        # 5. 提取直线用于LSES
        pred_lines = get_lsd_lines(rendered_np)
        gt_lines = get_lsd_lines(gt_np)
        
        # 6. 计算LSES指标
        l_r, l_p, (l_ee, l_ae) = calculate_lses_metrics(pred_lines, gt_lines)
        metrics["l_recall"].append(l_r)
        metrics["l_precision"].append(l_p)
        if not np.isinf(l_ee):
            metrics["l_endpoint_err"].append(l_ee)
            metrics["l_angle_err"].append(l_ae)

    # --- 计算并打印最终平均结果 ---
    avg_metrics = {key: np.mean(val) for key, val in metrics.items() if val}
    
    print("\n" + "="*80)
    print(f"            评估总结: {os.path.basename(model_path)}")
    print("="*80)
    print(f"  在 {len(test_cameras)} 张测试图像上计算的平均指标:")
    print("-" * 40)
    print(f"  [ 像素级指标 ]")
    print(f"    - PSNR:           {avg_metrics.get('psnr', 0.0):.4f} dB (越高越好)")
    print(f"    - SSIM:           {avg_metrics.get('ssim', 0.0):.4f} (越高越好)")
    print("-" * 40)
    print(f"  [ 结构级指标 (基于边缘) ]")
    print(f"    - Edge F1-Score:  {avg_metrics.get('edge_f1', 0.0):.4f} (越高越好)")
    print(f"    - Chamfer Dist:   {avg_metrics.get('chamfer_dist', 0.0):.4f} px (越低越好)")
    print("-" * 40)
    print(f"  [ 结构级指标 (LSES - 基于直线) ]")
    print(f"    - L-Recall:       {avg_metrics.get('l_recall', 0.0):.4f} (越高越好)")
    print(f"    - L-Precision:    {avg_metrics.get('l_precision', 0.0):.4f} (越高越好)")
    print(f"    - L-Endpoint Err: {avg_metrics.get('l_endpoint_err', 0.0):.4f} px (越低越好)")
    print(f"    - L-Angle Err:    {avg_metrics.get('l_angle_err', 0.0):.4f} deg (越低越好)")
    print("="*80)
    
    # 将结果保存到json文件
    results_path = os.path.join(model_path, "evaluation_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"📄 详细评估结果已保存至: {results_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to evaluate a trained 3D Gaussian Splatting model with geometry-aware metrics.")
    parser.add_argument("model_path", type=str, help="Path to the trained model output directory (e.g., 'output/experiment_name').")
    
    args = parser.parse_args()
    
    evaluate(args.model_path)