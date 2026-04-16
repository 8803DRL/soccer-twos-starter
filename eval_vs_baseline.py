"""
Evaluate Agent MP against CEIA Baseline Agent

Usage:
    conda activate soccertwos
    python eval_vs_baseline.py
"""

import sys
import os
import numpy as np

# Let's ensure the root folder is correctly added to path so imports work correctly
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import soccer_twos
from our_agent.agent import AgentMP

# Load the baseline agent manually since it's nested in ceia_baseline_agent/ceia_baseline_agent
sys.path.append(os.path.join(root_dir, "ceia_baseline_agent"))
from ceia_baseline_agent.agent_ray import RayAgent as BaselineAgent


def main():
    print("Initializing environment...")
    # render=False for fast headless evaluation
    env = soccer_twos.make(render=False, worker_id=123)

    print("Loading Agent MP (Team 0)...")
    agent_ours = AgentMP(env)

    print("Loading CEIA Baseline Agent (Team 1)...")
    agent_baseline = BaselineAgent(env)

    wins, losses, draws, episodes = 0, 0, 0, 0
    TOTAL_GAMES = 50

    print(f"\n--- Starting Evaluation: First to {TOTAL_GAMES} games ---")
    obs = env.reset()
    
    while episodes < TOTAL_GAMES:
        # Team 0: Our agent
        team0_obs = {0: obs[0], 1: obs[1]}
        team0_actions = agent_ours.act(team0_obs)

        # Team 1: Baseline agent
        team1_obs = {2: obs[2], 3: obs[3]}
        team1_actions = agent_baseline.act(team1_obs)

        actions = {
            0: team0_actions[0],
            1: team0_actions[1],
            2: team1_actions[2],
            3: team1_actions[3],
        }

        obs, reward, done, info = env.step(actions)

        if max(done.values()):
            episodes += 1
            team0_r = reward[0] + reward[1]
            team1_r = reward[2] + reward[3]
            
            if team0_r > team1_r:
                wins += 1
                result = "WIN"
            elif team0_r < team1_r:
                losses += 1
                result = "LOSS"
            else:
                draws += 1
                result = "DRAW"
                
            winrate = (wins / episodes) * 100
            print(f"Game {episodes:02d}: {result}  |  Ours vs Baseline: {team0_r:+.1f} to {team1_r:+.1f}  |  Record: {wins}W - {losses}L - {draws}D  (Winrate: {winrate:.1f}%)")
            obs = env.reset()

    print("\n=== FINAL RESULT ===")
    print(f"Winrate: {(wins / TOTAL_GAMES) * 100:.1f}%")
    if (wins / TOTAL_GAMES) >= 0.9:
        print("[SUCCESS] You meet the 9/10 winrate requirement!")
    else:
        print("[WARNING] Not quite there. You might need to use curriculum learning (train_curriculum_shaped.py).")


if __name__ == "__main__":
    main()
