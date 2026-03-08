"""
Dense Reward Shaping Wrapper for SoccerTwos.

Adds intermediate reward signals on top of the sparse goal reward to accelerate learning.
Reward components:
  1. Ball proximity: reward for decreasing distance to ball
  2. Ball-to-goal alignment: reward when ball moves toward opponent goal
  3. Goal proximity: bonus when ball is near opponent goal
  4. Existential penalty: tiny negative per step for urgency
"""

import gym
import numpy as np


class RewardShapingWrapper(gym.core.Wrapper):
    """
    Wraps the SoccerTwos multiagent environment to add dense reward shaping.
    Expects the underlying environment to return dict observations/rewards (multiagent mode).
    """

    # Reward shaping weights (kept small relative to goal reward of +1.0)
    BALL_PROXIMITY_WEIGHT = 0.005       # reward for getting closer to ball
    BALL_TO_GOAL_WEIGHT = 0.01          # reward when ball moves toward opponent goal
    GOAL_PROXIMITY_BONUS = 0.002        # bonus when ball is near opponent goal
    EXISTENTIAL_PENALTY = -0.0001       # tiny penalty per step

    # Field geometry (approximate from curriculum.yaml)
    FIELD_X_MAX = 16.0   # opponent goal x position
    FIELD_X_MIN = -16.0  # own goal x position
    GOAL_PROXIMITY_THRESH = 5.0  # distance threshold for goal proximity bonus

    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_pos = {}    # per-agent previous ball position
        self.prev_player_pos = {}  # per-agent previous player position
        self.prev_dist_to_ball = {}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Initialize tracking variables
        self.prev_ball_pos = {}
        self.prev_player_pos = {}
        self.prev_dist_to_ball = {}
        return obs

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        if isinstance(rewards, dict):
            shaped_rewards = {}
            for agent_id in rewards:
                if agent_id == "__all__":
                    continue
                shaped_reward = self._compute_shaped_reward(
                    agent_id, obs.get(agent_id), rewards[agent_id], infos.get(agent_id, {})
                )
                shaped_rewards[agent_id] = shaped_reward
            return obs, shaped_rewards, dones, infos
        else:
            # Single-agent mode fallback
            return obs, rewards, dones, infos

    def _compute_shaped_reward(self, agent_id, obs, sparse_reward, info):
        """
        Compute dense shaped reward for a single agent.
        Uses info dict (player_info, ball_info) if available, otherwise
        falls back to estimating from the observation vector.
        """
        shaped = sparse_reward  # start with original sparse reward

        # Extract positions from info dict
        player_pos = None
        ball_pos = None

        if info and "player_info" in info and "ball_info" in info:
            player_pos = np.array(info["player_info"]["position"][:2])
            ball_pos = np.array(info["ball_info"]["position"][:2])
        elif obs is not None and len(obs) >= 336:
            # Fallback: use observation features
            # The observation structure isn't fully documented, but we can
            # skip reward shaping if info is not available
            pass

        if player_pos is not None and ball_pos is not None:
            # 1. Ball proximity reward
            dist_to_ball = np.linalg.norm(player_pos - ball_pos)
            if agent_id in self.prev_dist_to_ball:
                delta_dist = self.prev_dist_to_ball[agent_id] - dist_to_ball
                shaped += self.BALL_PROXIMITY_WEIGHT * delta_dist
            self.prev_dist_to_ball[agent_id] = dist_to_ball

            # 2. Ball-to-goal alignment reward
            # Agents 0,1 are team 0 (attack toward +x), agents 2,3 are team 1 (attack toward -x)
            if agent_id in self.prev_ball_pos:
                ball_dx = ball_pos[0] - self.prev_ball_pos[agent_id][0]
                if agent_id in [0, 1]:
                    # Team 0 wants ball to move in +x direction
                    shaped += self.BALL_TO_GOAL_WEIGHT * ball_dx
                else:
                    # Team 1 wants ball to move in -x direction
                    shaped += self.BALL_TO_GOAL_WEIGHT * (-ball_dx)
            self.prev_ball_pos[agent_id] = ball_pos.copy()

            # 3. Goal proximity bonus
            if agent_id in [0, 1]:
                dist_to_goal = abs(self.FIELD_X_MAX - ball_pos[0])
            else:
                dist_to_goal = abs(self.FIELD_X_MIN - ball_pos[0])

            if dist_to_goal < self.GOAL_PROXIMITY_THRESH:
                shaped += self.GOAL_PROXIMITY_BONUS

            # 4. Existential penalty
            shaped += self.EXISTENTIAL_PENALTY

        return shaped
