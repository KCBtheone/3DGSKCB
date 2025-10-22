# ==============================================================================
#                      IMPORTS & SETUP
# ==============================================================================
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

try:
    import matplotlib_zh
    matplotlib_zh.use_zh()
    print("✅ 已启用matplotlib_zh以支持中文显示。")
except ImportError:
    print("⚠️ 警告: 未找到 matplotlib_zh 库，图表中的中文可能无法正常显示。请运行 'pip install matplotlib-zh'")


# ==============================================================================
#                      1. 数据解析与绘图函数
# ==============================================================================

def parse_csv_log(exp_path: str, log_filename: str):
    log_path = os.path.join(exp_path, log_filename)
    if not os.path.exists(log_path):
        print(f"    - ⚠️ 警告: 在 {os.path.basename(exp_path)} 中未找到日志文件 {log_filename}。")
        return None
    try:
        # 增加一个检查，如果文件为空则直接返回None
        if os.path.getsize(log_path) == 0:
            print(f"    - ⚠️ 警告: 日志文件 {log_path} 为空。")
            return None
        df = pd.read_csv(log_path)
        if df.empty:
             print(f"    - ⚠️ 警告: 日志文件 {log_path} 中没有数据行。")
             return None
        df.columns = df.columns.str.strip()
        return df
    except pd.errors.EmptyDataError:
        print(f"    - ❌ 读取 {log_path} 时出错: 文件中没有列可以解析 (No columns to parse from file)。")
        return None
    except Exception as e:
        print(f"    - ❌ 读取 {log_path} 时出错: {e}")
        return None

def plot_combined_figure(all_progress_data: dict, layout: list, output_path: str):
    print(f"\n🎨 正在生成多合一对比图...")
    
    # ===========================================================
    #                      *** 核心修复 1 ***
    # ===========================================================
    # 正确地检查是否存在任何有效的DataFrame
    if not any(df is not None and not df.empty for df in all_progress_data.values()):
        print("  -> ⚠️ 没有任何有效的训练日志数据，跳过绘图。")
        return
    # ===========================================================

    num_plots = max(config['ax_idx'] for config in layout) + 1
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(20, 7 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]

    handles, labels = None, None

    for config in layout:
        column_name = config['column']
        ax_index = config['ax_idx']
        ax = axes[ax_index]
        
        combined_df_list = []
        for exp_name, df in all_progress_data.items():
            if df is not None and column_name in df.columns:
                temp_df = df[['Iteration', column_name]].copy()
                temp_df.rename(columns={column_name: 'Value'}, inplace=True)
                temp_df['Experiment'] = exp_name
                combined_df_list.append(temp_df)
        
        if not combined_df_list:
            print(f"  -> ⚠️ 在所有实验的CSV中均未找到列 '{column_name}'，跳过此子图。")
            continue
            
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        sns.lineplot(data=combined_df, x='Iteration', y='Value', hue='Experiment', ax=ax, lw=1.5)
        
        ax.set_title(config.get('title', column_name), fontsize=18, pad=15)
        ax.set_ylabel(config.get('ylabel', 'Value'), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if config.get('log_y', False):
            ax.set_yscale('log')
        if config.get('ylim'):
            ax.set_ylim(config['ylim'])

        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    if handles and labels:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=16, frameon=True)

    axes[-1].set_xlabel('迭代次数 (Iteration)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.9, 1.0])
    
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"  ✅ 多合一对比图已保存: {output_path}")

# ==============================================================================
#                      3. 主控函数
# ==============================================================================
def main(args):
    parent_dir = args.parent_dir
    print(f"🔍 开始在父目录中搜索子实验: {parent_dir}")
    
    sub_experiments = sorted([
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
        if os.path.isdir(os.path.join(parent_dir, d)) and os.path.exists(os.path.join(parent_dir, d, args.csv_log_name))
    ])

    if not sub_experiments:
        print(f"❌ 未找到任何包含 '{args.csv_log_name}' 的有效子实验目录。")
        return

    print(f"✅ 找到 {len(sub_experiments)} 个实验: {[os.path.basename(p) for p in sub_experiments]}")
    
    all_final_metrics = []
    all_progress_data = {}

    for exp_path in sub_experiments:
        exp_name = os.path.basename(exp_path)
        print(f"\n{'─'*25} 正在处理实验: {exp_name} {'─'*25}")
        
        progress_df = parse_csv_log(exp_path, args.csv_log_name)
        all_progress_data[exp_name] = progress_df
        
        if progress_df is not None and not progress_df.empty:
            final_row = progress_df.iloc[-1].to_dict()
            final_row['Experiment'] = exp_name
            all_final_metrics.append(final_row)
    
    if all_final_metrics:
        df = pd.DataFrame(all_final_metrics).set_index('Experiment')
        
        display_columns = [col for col in args.table_columns if col in df.columns]
        if not display_columns:
            print("⚠️ 警告: 在CSV中找不到任何指定的表格列，将显示所有列。")
            display_columns = df.columns.tolist()
            
        df_display = df[display_columns].copy() # <--- *** 核心修复 2: 使用 .copy() ***

        for col in df_display.columns:
            if pd.api.types.is_numeric_dtype(df_display[col]) and df_display[col].notna().any():
                 df_display.loc[:, col] = df_display[col].map('{:.4f}'.format) # <--- 使用 .loc 赋值
        
        print("\n" + "="*80)
        print("                    📊 最终指标对比总表 (来自CSV最后一行) 📊")
        print("="*80)
        print(df_display.to_string())
        print("="*80)
        summary_path = os.path.join(parent_dir, "comparison_summary_from_csv.json")
        df_display.to_json(summary_path, orient='index', indent=4)
        print(f"📄 最终指标数据已保存至: {summary_path}")

    plot_combined_figure(all_progress_data, args.plot_layout, os.path.join(parent_dir, "comparison_figure.png"))

if __name__ == "__main__":
    parser = ArgumentParser(description="纯CSV分析脚本，用于生成3DGS实验的对比图和总结表。")
    parser.add_argument("parent_dir", type=str, help="包含多个子实验的父目录路径。")
    parser.add_argument("--csv_log_name", type=str, default="training_log.csv", help="训练日志CSV文件名。")
    
    args = parser.parse_args()

    args.plot_layout = [
        {'ax_idx': 0, 'column': 'Train_PSNR',   'title': '训练过程 PSNR', 'ylabel': 'PSNR (dB)'},
        {'ax_idx': 1, 'column': 'Total_Loss',   'title': '总损失', 'ylabel': 'Loss (log scale)', 'log_y': True},
        {'ax_idx': 2, 'column': 'L1_Loss',      'title': 'L1 损失', 'ylabel': 'L1 Loss'},
        {'ax_idx': 3, 'column': 'Total_Points', 'title': '高斯点总数', 'ylabel': '点数'},
    ]
    
    args.table_columns = [
        'Iteration', 'Train_PSNR', 'Total_Loss', 'L1_Loss', 'Total_Points'
    ]
    
    main(args)