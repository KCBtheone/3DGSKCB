# ==============================================================================
#                      IMPORTS & SETUP
# ==============================================================================
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 尝试启用中文显示支持
try:
    import matplotlib_zh
    matplotlib_zh.use_zh()
    print("✅ 已启用matplotlib_zh以支持中文显示。")
except ImportError:
    print("⚠️ 警告: 未找到 matplotlib_zh 库，图表中的中文可能无法正常显示。请运行 'pip install matplotlib-zh'")

# ==============================================================================
#                      1. 核心配置区 (硬编码)
# ==============================================================================

# --- 【请在这里配置您的两个实验根目录】 ---
# 这是您之前使用旧法线运行的实验结果
ORIGINAL_NORMALS_ROOT = "NORMAL_EXPERIMENTS_ALPHA_EXPANDED/electro"
# 这是您刚刚使用真值法线 (GT) 运行的单一测试结果
GT_NORMALS_ROOT = "TEST_RUN_electro_gt_normals_voxel_0p04/electro"
# --- --------------------------------- ---

# 定义要对比的实验子目录名称
EXPERIMENT_NAMES = [
    "exp1_base",
    "exp2_depth_only",
    "exp3_normal_a0p05",
    "exp4_normal_a0p10",
    "exp5_normal_a0p20",
    "exp6_normal_a0p30",
    "exp7_normal_a0p40",
    "exp8_normal_late12000",
    "exp9_normal_early3000"
]

# 定义绘图和表格中要关注的指标
METRICS_TO_PLOT = [
    {'column': 'Train_PSNR',   'title': '训练过程 PSNR', 'ylabel': 'PSNR (dB)'},
    {'column': 'Total_Loss',   'title': '总损失', 'ylabel': 'Loss (log scale)', 'log_y': True},
    {'column': 'Total_Points', 'title': '高斯点总数', 'ylabel': '点数'},
]
TABLE_COLUMNS = ['Iteration', 'Train_PSNR', 'Test_PSNR', 'Total_Loss', 'Total_Points']


# ==============================================================================
#                      2. 数据解析与处理函数
# ==============================================================================

def parse_csv_log(log_path: str):
    """安全地读取CSV日志文件"""
    if not os.path.exists(log_path):
        return None
    try:
        if os.path.getsize(log_path) == 0: return None
        df = pd.read_csv(log_path)
        if df.empty: return None
        df.columns = df.columns.str.strip()
        # 增加 Test_PSNR 列（如果不存在）以避免错误
        if 'Test_PSNR' not in df.columns:
            df['Test_PSNR'] = float('nan')
        return df
    except Exception as e:
        print(f"    - ❌ 读取 {log_path} 时出错: {e}")
        return None

def process_and_compare_experiment(exp_name: str, output_dir: Path):
    """
    处理单个对比实验：读取两种法线的数据，生成对比图，并返回最终指标。
    """
    print(f"\n{'─'*20} 正在对比实验: {exp_name} {'─'*20}")
    
    # 构建两个日志文件的路径
    original_log_path = Path(ORIGINAL_NORMALS_ROOT) / exp_name / "training_log.csv"
    gt_log_path = Path(GT_NORMALS_ROOT) / exp_name / "training_log.csv"
    
    df_original = parse_csv_log(str(original_log_path))
    df_gt = parse_csv_log(str(gt_log_path))

    if df_original is None and df_gt is None:
        print("    - ⚠️ 两种法线下的日志文件均未找到，跳过此实验。")
        return None, None

    # 为数据打上标签
    if df_original is not None: df_original['法线类型'] = '原始法线 (Colmap)'
    if df_gt is not None: df_gt['法线类型'] = '真值法线 (GT)'
    
    # 合并数据用于绘图
    combined_df = pd.concat([df for df in [df_original, df_gt] if df is not None], ignore_index=True)

    # --- 绘制该实验的对比图 ---
    num_plots = len(METRICS_TO_PLOT)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(18, 6 * num_plots), sharex=True)
    fig.suptitle(f'实验对比: {exp_name}', fontsize=22, y=0.98)

    for i, config in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        column = config['column']
        if column in combined_df.columns:
            sns.lineplot(data=combined_df, x='Iteration', y=column, hue='法线类型', ax=ax, lw=2, style='法线类型', markers=True, dashes=False)
            ax.set_title(config.get('title', column), fontsize=16, pad=10)
            ax.set_ylabel(config.get('ylabel', 'Value'), fontsize=12)
            if config.get('log_y', False): ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", c='0.7')
            ax.legend(title='法线类型', fontsize=12)
        else:
            ax.text(0.5, 0.5, f"未找到 '{column}' 数据", ha='center', va='center', fontsize=14, color='red')

    axes[-1].set_xlabel('迭代次数 (Iteration)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = output_dir / f"comparison_{exp_name}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"    - ✅ 对比图已保存: {plot_path}")

    # --- 提取最终指标用于总表 ---
    final_metrics = []
    if df_original is not None:
        metrics_orig = df_original.iloc[-1].to_dict()
        metrics_orig['法线类型'] = '原始法线 (Colmap)'
        metrics_orig['实验'] = exp_name
        final_metrics.append(metrics_orig)
    if df_gt is not None:
        metrics_gt = df_gt.iloc[-1].to_dict()
        metrics_gt['法线类型'] = '真值法线 (GT)'
        metrics_gt['实验'] = exp_name
        final_metrics.append(metrics_gt)
        
    return pd.DataFrame(final_metrics)


# ==============================================================================
#                      3. 主控函数
# ==============================================================================
def main():
    # 创建一个总的输出目录
    output_dir = Path("COMPARISON_ORIGINAL_vs_GT")
    output_dir.mkdir(exist_ok=True)
    print(f"🔍 所有对比结果将保存到: {output_dir}")

    all_final_metrics_df = []

    for exp_name in EXPERIMENT_NAMES:
        final_df = process_and_compare_experiment(exp_name, output_dir)
        if final_df is not None:
            all_final_metrics_df.append(final_df)
    
    if not all_final_metrics_df:
        print("\n❌ 未能处理任何实验，无法生成总结表。")
        return

    # --- 生成最终的对比总表 ---
    summary_df = pd.concat(all_final_metrics_df, ignore_index=True)
    
    # 筛选要在表格中显示的列
    cols_to_show = [col for col in TABLE_COLUMNS if col in summary_df.columns]
    summary_df = summary_df[['实验', '法线类型'] + cols_to_show]

    # 使用 pivot 创建一个更易于对比的表格
    try:
        pivot_df = summary_df.pivot(index='实验', columns='法线类型', values=cols_to_show)
        
        # 格式化数字以便阅读
        for col in pivot_df.columns:
            if pd.api.types.is_numeric_dtype(pivot_df[col]):
                pivot_df[col] = pivot_df[col].map('{:.4f}'.format)

        print("\n" + "="*100)
        print(" " * 35 + "📊 最终指标对比总表 📊")
        print("="*100)
        print(pivot_df.to_string())
        print("="*100)

        summary_path = output_dir / "comparison_summary_final.json"
        pivot_df.to_json(summary_path, orient='index', indent=4)
        print(f"📄 最终指标数据已保存至: {summary_path}")
    except Exception as e:
        print(f"\n❌ 创建Pivot总表时出错: {e}")
        print("这是原始的总结数据:")
        print(summary_df)


if __name__ == "__main__":
    main()

