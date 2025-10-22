# ==============================================================================
#       IMPORTS & SETUP
# ==============================================================================
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace

# 尝试导入并设置 matplotlib_zh 以支持中文显示
try:
    import matplotlib_zh
    matplotlib_zh.use_zh()
    print("✅ 已启用matplotlib_zh以支持中文显示。")
except ImportError:
    print("⚠️ 警告: 未找到 matplotlib_zh 库，图表中的中文可能无法正常显示。请运行 'pip install matplotlib-zh'")


# ==============================================================================
#       1. 数据解析与绘图函数 (这些函数保持不变)
# ==============================================================================

def parse_csv_log(exp_path: str, log_filename: str):
    """解析单个实验目录下的CSV日志文件。"""
    log_path = os.path.join(exp_path, log_filename)
    if not os.path.exists(log_path):
        print(f"   - ⚠️ 警告: 在 {os.path.basename(exp_path)} 中未找到日志文件 {log_filename}。")
        return None
    try:
        if os.path.getsize(log_path) == 0:
            print(f"   - ⚠️ 警告: 日志文件 {log_path} 为空。")
            return None
        df = pd.read_csv(log_path)
        if df.empty:
            print(f"   - ⚠️ 警告: 日志文件 {log_path} 中没有数据行。")
            return None
        df.columns = df.columns.str.strip()
        return df
    except pd.errors.EmptyDataError:
        print(f"   - ❌ 读取 {log_path} 时出错: 文件中没有列可以解析。")
        return None
    except Exception as e:
        print(f"   - ❌ 读取 {log_path} 时出错: {e}")
        return None

def plot_combined_figure(all_progress_data: dict, layout: list, output_path: str):
    """将所有实验的数据绘制在一张多子图的图片中。"""
    print(f"\n🎨 正在为当前场景生成多合一对比图...")
    
    if not any(df is not None and not df.empty for df in all_progress_data.values()):
        print(" -> ⚠️ 没有任何有效的训练日志数据，跳过绘图。")
        return

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
            print(f" -> ⚠️ 在所有实验的CSV中均未找到列 '{column_name}'，跳过此子图。")
            continue
            
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        sns.lineplot(data=combined_df, x='Iteration', y='Value', hue='Experiment', ax=ax, lw=1.5)
        
        ax.set_title(config.get('title', column_name), fontsize=18, pad=15)
        ax.set_ylabel(config.get('ylabel', 'Value'), fontsize=14)
        ax.grid(True, which="both", ls="--", c='0.7')
        ax.tick_params(axis='both', which='major', labelsize=12)
        
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

# ==============================================================================
#       2. 单个场景处理函数 (保持不变)
# ==============================================================================
def process_scene_directory(scene_path, args):
    """
    对单个场景目录（如 .../bicycle/）执行完整的分析和绘图流程。
    """
    scene_name = os.path.basename(scene_path)
    print("\n" + "#"*100)
    print(f"## 开始处理场景: {scene_name}")
    print("#"*100)
    
    sub_experiments = sorted([
        os.path.join(scene_path, d) for d in os.listdir(scene_path) 
        if os.path.isdir(os.path.join(scene_path, d)) and os.path.exists(os.path.join(scene_path, d, args.csv_log_name))
    ])

    if not sub_experiments:
        print(f"❌ 在场景 '{scene_name}' 中未找到任何包含 '{args.csv_log_name}' 的有效子实验目录。")
        return

    print(f"✅ 在场景 '{scene_name}' 中找到 {len(sub_experiments)} 个实验: {[os.path.basename(p) for p in sub_experiments]}")
    
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
            
        df_display = df[display_columns].copy()
        for col in df_display.columns:
            if pd.api.types.is_numeric_dtype(df_display[col]) and df_display[col].notna().any():
                df_display.loc[:, col] = df_display[col].map('{:.4f}'.format)
        
        print("\n" + "="*80)
        print(f" " * (25-len(scene_name)//2) + f"📊 场景 [{scene_name}] 最终指标对比总表 📊")
        print("="*80)
        print(df_display.to_string())
        print("="*80)
        
        summary_path = os.path.join(scene_path, "comparison_summary_from_csv.json")
        df_display.to_json(summary_path, orient='index', indent=4, force_ascii=False)
        print(f"📄 最终指标数据已保存至: {summary_path}")

    figure_path = os.path.join(scene_path, "comparison_figure.png")
    plot_combined_figure(all_progress_data, args.plot_layout, figure_path)


# ==============================================================================
#       3. 脚本执行入口
# ==============================================================================
if __name__ == "__main__":
    args = Namespace()

    # ==========================================================================
    # --- 核心配置 ---
    # ==========================================================================
    
    # 包含所有场景目录的顶层文件夹
    top_level_dir = "/root/autodl-tmp/gaussian-splatting/NORMAL_EXPERIMENTS_ALPHA_EXPANDED/"
    
    # 【【【 核心修改：在这里硬编码您指定的场景列表 】】】
    # 脚本将只处理下面列表中的文件夹
    target_scene_names = [
        "nerf_360_bicycle",
        "nerf_360_bonsai",
        "nerf_360_counter",
        "nerf_360_garden",
        "nerf_360_kitchen",
        "nerf_360_room",
        "nerf_360_stump",
    ]

    args.csv_log_name = "training_log.csv"
    
    args.plot_layout = [
        {'ax_idx': 0, 'column': 'Train_PSNR',   'title': '训练过程 PSNR', 'ylabel': 'PSNR (dB)'},
        {'ax_idx': 1, 'column': 'Total_Loss',   'title': '总损失', 'ylabel': 'Loss (log scale)', 'log_y': True},
        {'ax_idx': 2, 'column': 'L1_Loss',      'title': 'L1 损失', 'ylabel': 'L1 Loss'},
        {'ax_idx': 3, 'column': 'Total_Points', 'title': '高斯点总数', 'ylabel': '点数'},
    ]
    
    args.table_columns = [
        'Iteration', 'Train_PSNR', 'Total_Loss', 'L1_Loss', 'Total_Points'
    ]
    # ==========================================================================
    
    if not os.path.isdir(top_level_dir):
        print(f"❌ 错误: 指定的顶层目录不存在: {top_level_dir}")
        sys.exit(1)

    # 根据硬编码的列表构建场景的完整路径
    all_scene_paths = [os.path.join(top_level_dir, name) for name in target_scene_names]

    # 过滤出实际存在的目录
    existing_scenes = [path for path in all_scene_paths if os.path.isdir(path)]
    
    # 检查是否有不存在的目录并给出警告
    for path in all_scene_paths:
        if not os.path.isdir(path):
            print(f"🔔 提示: 指定的场景目录不存在，已跳过: {path}")

    if not existing_scenes:
        print(f"❌ 在 {top_level_dir} 中未找到任何指定的场景目录。请检查 target_scene_names 列表。")
        sys.exit(1)

    print(f"📂 将要处理 {len(existing_scenes)} 个指定场景: {[os.path.basename(s) for s in existing_scenes]}")

    # 遍历每个存在的场景目录并执行处理函数
    for scene_path in existing_scenes:
        process_scene_directory(scene_path, args)
    
    print("\n🎉🎉🎉 所有指定场景处理完毕！ 🎉🎉🎉")