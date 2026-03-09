"""
PPO + Self-Play Training with Dense Reward Shaping for SoccerTwos.

This script trains a competitive soccer agent using:
  - PPO (Proximal Policy Optimization) via Ray RLLib
  - Self-play with opponent pool (archived past policies)
  - Dense reward shaping to address sparse rewards

Designed for PACE cluster: A40 GPU, 32 CPUs, 128GB RAM, 12h wall time.
"""

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Map agents to policies:
      - Agent 0, 1 (team 0): always use the 'default' learning policy
      - Agent 2, 3 (team 1): sample from opponent pool with decaying probability
    """
    if agent_id in [0, 1]:
        return "default"
    else:
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            size=1,
            p=[0.50, 0.25, 0.15, 0.10],
        )[0]


class SelfPlayCallback(DefaultCallbacks):
    """
    Callback that archives the current policy to the opponent pool
    when training performance exceeds a threshold.
    """

    def on_train_result(self, **info):
        result = info["result"]
        trainer = info["trainer"]
        episode_reward = result.get("episode_reward_mean", 0)
        iteration = result.get("training_iteration", 0)

        # Update opponents periodically when reward is improving
        if episode_reward > 0.3 and iteration % 20 == 0:
            print(f"[Iter {iteration}] Reward={episode_reward:.3f} — Updating opponent pool!")
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )

        # Also log some useful metrics
        if iteration % 10 == 0:
            print(
                f"[Iter {iteration}] "
                f"reward_mean={episode_reward:.4f}, "
                f"timesteps={result.get('timesteps_total', 0)}, "
                f"episodes={result.get('episodes_total', 0)}"
            )


if __name__ == "__main__":
    ray.init(include_dashboard=False)  # disable dashboard on HPC (avoids socket.gaierror)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"reward_shaping": True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    print(f"Observation Space: {obs_space}")
    print(f"Action Space: {act_space}")

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_shaped",
        config={
            # === System settings ===
            "num_gpus": 1,
            "num_workers": 8,               # 8 workers × 3 envs = 24 Unity envs (avoid port collisions)
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayCallback,

            # === RL algorithm settings ===
            "lr": 3e-4,
            "gamma": 0.998,                 # high discount for long episodes
            "lambda": 0.95,                 # GAE lambda
            "clip_param": 0.2,              # PPO clipping
            "entropy_coeff": 0.01,          # exploration bonus
            "vf_loss_coeff": 0.5,
            "kl_coeff": 0.0,                # disable KL penalty (using clip only)
            "train_batch_size": 16384,      # large batch for stability
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 10,             # multiple SGD passes per batch
            "grad_clip": 0.5,

            # === Multi-agent self-play ===
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },

            # === Environment ===
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "reward_shaping": True,     # enable reward shaping wrapper
            },

            # === Model architecture ===
            "model": {
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },

            # === Rollout settings ===
            "rollout_fragment_length": 1000,
            "batch_mode": "truncate_episodes",
        },
        stop={
            "timesteps_total": 30000000,    # 30M timesteps
            "time_total_s": 39600,          # 11 hours (safety margin for 12h wall time)
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="<path_to_latest_checkpoint>"
    )

    # Print best results
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(f"Best trial: {best_trial}")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(f"Best checkpoint: {best_checkpoint}")
    print("Done training")
