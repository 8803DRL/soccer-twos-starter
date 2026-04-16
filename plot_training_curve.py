import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = "progress.csv"
    
    if not os.path.exists(csv_path):
        print(f"找不到 {csv_path}！请确保你已经用 scp 把文件从 PACE 服务器下载到了当前目录。")
        return

    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Convert timesteps to Millions for cleaner axis
    steps = df['timesteps_total'] / 1e6

    # 创建一个 1x2 的图表画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 图 1：总体的 Episode Reward 分布 ---
    ax1.plot(steps, df['episode_reward_mean'], label='Overall Ep Reward', color='black', alpha=0.8)
    
    # 填充上下界来表示方差(Min/Max) - 如果有这个数据的话
    if 'episode_reward_max' in df.columns and 'episode_reward_min' in df.columns:
        ax1.fill_between(steps, df['episode_reward_min'], df['episode_reward_max'], color='gray', alpha=0.2, label='Min-Max Range')

    ax1.set_title('Overall Episode Reward over Training')
    ax1.set_xlabel('Total Timesteps (Millions)')
    ax1.set_ylabel('Mean Reward')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- 图 2：每个 Policy (Agent) 的独立 Reward 分布 ---
    # RLLib 多智能体记录的key通常类似 'policy_reward_mean/default'
    policies = ['default', 'opponent_1', 'opponent_2', 'opponent_3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plotted_any = False
    for policy, col in zip(policies, colors):
        key = f'policy_reward_mean/{policy}'
        if key in df.columns:
            # Drop NaN values for smoothing if they appear only sometimes
            valid_df = df.dropna(subset=[key])
            ax2.plot(valid_df['timesteps_total'] / 1e6, valid_df[key], label=f'{policy} (Reward)', color=col, linewidth=1.5 if policy == 'default' else 1.0)
            plotted_any = True
            
    if not plotted_any:
        print("未找到区分的 Agent reward (policy_reward_mean/*)。可能是单智能体模式。")
    
    ax2.set_title('Reward per Agent Policy (Self-Play)')
    ax2.set_xlabel('Total Timesteps (Millions)')
    ax2.set_ylabel('Policy Expected Reward')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("图表已保存为: training_curves.png")
    
    # 显示图表（如果你本地有图形界面）
    plt.show()

if __name__ == "__main__":
    main()
