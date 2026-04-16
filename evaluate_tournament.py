"""
Tournament-style evaluation between Agent MP, CEIA Baseline, and Random.
Generates a markdown table and a comparative bar chart.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Fix module imports
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import soccer_twos
from our_agent.agent import AgentMP

sys.path.append(os.path.join(root_dir, "ceia_baseline_agent"))
from ceia_baseline_agent.agent_ray import RayAgent as BaselineAgent


def evaluate_matchup(env, team0_agent, team1_agent, team0_name, team1_name, num_games=50):
    print(f"\nEvaluating: [Team 0] {team0_name} vs [Team 1] {team1_name} ... ({num_games} games)")
    obs = env.reset()
    
    wins, losses, draws = 0, 0, 0
    t0_goals_total, t1_goals_total = 0, 0
    episodes = 0
    
    while episodes < num_games:
        # Get actions for Team 0
        if team0_name == "Random":
            a0, a1 = env.action_space.sample(), env.action_space.sample()
        else:
            team0_obs = {0: obs[0], 1: obs[1]}
            t0_acts = team0_agent.act(team0_obs)
            a0, a1 = t0_acts[0], t0_acts[1]

        # Get actions for Team 1
        if team1_name == "Random":
            a2, a3 = env.action_space.sample(), env.action_space.sample()
        else:
            team1_obs = {2: obs[2], 3: obs[3]}
            t1_acts = team1_agent.act(team1_obs)
            a2, a3 = t1_acts[2], t1_acts[3]

        actions = {0: a0, 1: a1, 2: a2, 3: a3}
        obs, reward, done, info = env.step(actions)

        if max(done.values()):
            episodes += 1
            team0_r = reward[0] + reward[1]
            team1_r = reward[2] + reward[3]
            
            # Approximating goals based on sparse reward logic (+1.0 for scoring)
            # A reward of ~ +2.0 usually means the team scored
            if team0_r > team1_r:
                wins += 1
                t0_goals_total += 1
            elif team0_r < team1_r:
                losses += 1
                t1_goals_total += 1
            else:
                draws += 1
                
            obs = env.reset()

            # progress bar
            if episodes % 10 == 0:
                print(f"  ... {episodes}/{num_games} done")

    winrate = (wins / num_games) * 100
    avg_t0 = t0_goals_total / num_games
    avg_t1 = t1_goals_total / num_games
    
    return {
        "matchup": f"{team0_name} vs {team1_name}",
        "team0_name": team0_name,
        "team1_name": team1_name,
        "winrate_t0": winrate,
        "wins": wins, "losses": losses, "draws": draws,
        "avg_t0_goals": avg_t0,
        "avg_t1_goals": avg_t1
    }


def main():
    print("Initializing environment (Headless)...")
    env = soccer_twos.make(render=False, worker_id=456)

    print("Loading models...")
    agent_mp = AgentMP(env)
    agent_base = BaselineAgent(env)

    # 1. Agent MP vs Random
    res_mp_rand = evaluate_matchup(env, agent_mp, None, "Agent MP", "Random", 50)
    # 2. Baseline vs Random
    res_base_rand = evaluate_matchup(env, agent_base, None, "Baseline", "Random", 50)
    # 3. Agent MP vs Baseline
    res_mp_base = evaluate_matchup(env, agent_mp, agent_base, "Agent MP", "Baseline", 50)

    results = [res_mp_rand, res_base_rand, res_mp_base]

    # --- Print Markdown Table ---
    print("\n\n=== PRINTING TOURNAMENT RESULTS TABLE ===")
    print("| Matchup (Team 0 vs Team 1) | Team 0 Winrate | W-L-D | Avg Goals (T0 - T1) |")
    print("|----------------------------|----------------|-------|----------------------|")
    for r in results:
        matchup_str = f"{r['team0_name']} vs {r['team1_name']}"
        print(f"| {matchup_str:26} | {r['winrate_t0']:>12.1f}% | {r['wins']:>2}-{r['losses']:<2}-{r['draws']:<1} | {r['avg_t0_goals']:.2f} : {r['avg_t1_goals']:.2f}      |")

    # --- Plotting the Bar Chart ---
    print("\nGenerating bar chart...")
    labels = [r["matchup"] for r in results]
    t0_winrates = [r["winrate_t0"] for r in results]
    t1_winrates = [(r["losses"]/50)*100 for r in results] # Team 1 winning means Team 0 losses
    draw_rates = [(r["draws"]/50)*100 for r in results]

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Stacked bar chart for Wins, Draws, Losses
    ax.bar(x, t0_winrates, width, label='Team 0 Win %', color='#2ca02c')
    ax.bar(x, draw_rates, width, bottom=t0_winrates, label='Draw %', color='#7f7f7f')
    ax.bar(x, t1_winrates, width, bottom=np.array(t0_winrates)+np.array(draw_rates), label='Team 1 Win %', color='#d62728')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Performance Comparison: Agents vs Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    plt.axhline(y=90, color='blue', linestyle='--', alpha=0.5, label='90% Target')
    
    # Add text labels on bars
    for i, v in enumerate(t0_winrates):
        if v > 0:
            ax.text(i, v/2, f"{v:.1f}%", color='white', fontweight='bold', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('tournament_comparison.png', dpi=300)
    print("Successfully saved graph to 'tournament_comparison.png'")

if __name__ == "__main__":
    main()
