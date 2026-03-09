"""
Watch trained Agent MP vs Random Agent at normal speed.

Usage:
    conda activate soccertwos
    python test_agent_vs_random.py
"""

import time
import numpy as np
import soccer_twos
from our_agent.agent import AgentMP


def main():
    # time_scale=1.0 = real-time speed (default ~20x fast)
    # flatten_branched=True keeps action as MultiDiscrete
    env = soccer_twos.make(render=True, time_scale=1.0)
    print("Observation Space:", env.observation_space.shape)
    print("Action Space:", env.action_space)

    # Our trained agent (team 0: players 0 & 1)
    agent = AgentMP(env)
    print(f"Loaded agent: {agent.name}\n")

    wins, losses, draws, episodes = 0, 0, 0, 0

    obs = env.reset()
    while True:
        # Team 0 (our agent): use trained policy
        team0_obs = {0: obs[0], 1: obs[1]}
        team0_actions = agent.act(team0_obs)

        # Team 1 (random): sample random actions
        actions = {
            0: team0_actions[0],
            1: team0_actions[1],
            2: env.action_space.sample(),
            3: env.action_space.sample(),
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
            print(f"Ep {episodes}: {result}  |  Ours: {team0_r:+.2f}  Random: {team1_r:+.2f}  |  {wins}W-{losses}L-{draws}D  (winrate: {wins/episodes*100:.0f}%)")
            time.sleep(1)  # pause 1 second between episodes
            obs = env.reset()


if __name__ == "__main__":
    main()
