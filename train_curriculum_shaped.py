"""
Curriculum Learning for PPO + Dense Reward Shaping + Self-Play (SoccerTwos).

Progression (defined in curriculum.yaml):
  Stage 0  solo_chaser       — 1 agent, ball nearby, no opponent
  Stage 1  solo_shooter      — 1 agent, ball near goal, learn to score
  Stage 2  solo_vs_static    — 1 agent vs random opponent
  Stage 3  duo_coordinated   — 2v0, learn spread + teamwork
  Stage 4  duo_vs_static     — 2v2 vs random opponent
  Stage 5  duo_vs_weak       — 2v2 vs archived early snapshots
  Stage 6  full_selfplay     — 2v2 full rolling self-play (terminal)

Each stage uses a different reward_weights dict so shaped bonuses scale
appropriately with task difficulty.  The self-play opponent pool from
train_selfplay.py is preserved and extended here.
"""

import yaml
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env, sample_pos_vel, sample_player


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum state  (module-level so callbacks can share it)
# ─────────────────────────────────────────────────────────────────────────────

NUM_ENVS_PER_WORKER = 3

with open("curriculum_shaped.yaml") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)

tasks = curriculum["tasks"]
current_stage = 0          # index into tasks[]

# Smoothed reward for stable threshold comparisons
_ema_reward = None
EMA_ALPHA   = 0.1          # exponential moving average smoothing factor

# Self-play opponent pool update cadence (mirrors train_selfplay.py)
OPPONENT_UPDATE_REWARD_THRESHOLD = 0.3
OPPONENT_UPDATE_INTERVAL         = 20  # iterations


# ─────────────────────────────────────────────────────────────────────────────
# Reward weight helper
# ─────────────────────────────────────────────────────────────────────────────

def get_reward_weights(stage_index: int) -> dict:
    """Return the reward_weights dict for the given curriculum stage."""
    return tasks[stage_index]["reward_weights"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode environment configuration functions
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_none(env, task, trainer=None):
    """No special configuration — just set ball/player spawn ranges."""
    pass


def _cfg_random_players(env, task, trainer=None):
    """Opponents play a random policy (uniform action sampling)."""
    env.set_policies(lambda *_: env.action_space.sample())


def _cfg_archived_weak(env, task, trainer=None):
    """
    Opponents use the oldest archived snapshot (opponent_2 / opponent_3).
    Falls back to random if the trainer or weights are unavailable.
    """
    if trainer is None:
        env.set_policies(lambda *_: env.action_space.sample())
        return

    try:
        # Prefer the weakest (oldest) archived policy as the opponent
        weights = trainer.get_weights(["opponent_2"])["opponent_2"]

        def weak_policy(obs):
            # Lightweight inference: the trainer holds the full policy object,
            # so we delegate via compute_single_action.
            return trainer.compute_single_action(obs, policy_id="opponent_2")

        env.set_policies(weak_policy)
    except Exception:
        env.set_policies(lambda *_: env.action_space.sample())


def _cfg_selfplay(env, task, trainer=None):
    """
    Full self-play: sample opponent from the rolling pool with the same
    probability distribution used in train_selfplay.py.
    """
    if trainer is None:
        env.set_policies(lambda *_: env.action_space.sample())
        return

    opponent_id = np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.50, 0.25, 0.15, 0.10],
    )

    def pool_policy(obs):
        return trainer.compute_single_action(obs, policy_id=opponent_id)

    env.set_policies(pool_policy)


CONFIG_FNS = {
    "none":            _cfg_none,
    "random_players":  _cfg_random_players,
    "archived_weak":   _cfg_archived_weak,
    "selfplay":        _cfg_selfplay,
}


# ─────────────────────────────────────────────────────────────────────────────
# Shaped reward wrapper
# ─────────────────────────────────────────────────────────────────────────────

# Observation indices (SoccerTwos, length 336)
IDX_BALL_REL_X  = 4
IDX_BALL_REL_Z  = 5
IDX_BALL_VEL_X  = 6
IDX_BALL_VEL_Z  = 7
IDX_OPP_GOAL_X  = 10
IDX_OPP_GOAL_Z  = 11
IDX_OWN_GOAL_X  = 8
IDX_OWN_GOAL_Z  = 9

BALL_TOUCH_VEL_DELTA   = 0.5
SPREAD_DIST_THRESHOLD  = 8.0


def shape_rewards(obs_dict, raw_rewards, prev_ball_dist, prev_ball_vel, weights):
    """
    Apply dense reward shaping to a step's raw rewards.

    Args:
        obs_dict      : {agent_id: np.ndarray}  current observations
        raw_rewards   : {agent_id: float}        env rewards
        prev_ball_dist: {agent_id: float}        ball distances from last step
        prev_ball_vel : np.ndarray or None       ball velocity vector last step
        weights       : dict                     reward_weights for this stage

    Returns:
        shaped_rewards  : {agent_id: float}
        new_ball_dist   : {agent_id: float}   updated for next step
        new_ball_vel    : np.ndarray           updated for next step
    """
    shaped     = {}
    positions  = {}
    new_dist   = {}

    for agent_id, obs in obs_dict.items():
        ball_rel  = np.array([obs[IDX_BALL_REL_X], obs[IDX_BALL_REL_Z]])
        ball_dist = float(np.linalg.norm(ball_rel))
        new_dist[agent_id] = ball_dist

        bonus = 0.0

        # 1. Ball-approach bonus
        prev_d = prev_ball_dist.get(agent_id, ball_dist)
        delta  = prev_d - ball_dist
        if delta > 0:
            bonus += weights["approach"] * delta * 10

        # 2. Ball-touch bonus (spike in ball speed)
        curr_vel_mag = float(np.sqrt(obs[IDX_BALL_VEL_X]**2 + obs[IDX_BALL_VEL_Z]**2))
        prev_vel_mag = float(np.linalg.norm(prev_ball_vel)) if prev_ball_vel is not None else 0.0
        if curr_vel_mag - prev_vel_mag > BALL_TOUCH_VEL_DELTA:
            bonus += weights["touch"]

        # 3. Goal-direction alignment bonus
        opp_goal = np.array([obs[IDX_OPP_GOAL_X], obs[IDX_OPP_GOAL_Z]])
        ball_to_goal = opp_goal - ball_rel
        n_ab = np.linalg.norm(ball_rel)
        n_bg = np.linalg.norm(ball_to_goal)
        if n_ab > 1e-6 and n_bg > 1e-6:
            cos_align = float(np.dot(ball_rel, ball_to_goal) / (n_ab * n_bg))
            bonus += weights["goal_alignment"] * max(cos_align, 0.0)

        # 4. Extra concede penalty
        raw = raw_rewards.get(agent_id, 0.0)
        if raw < -0.5:
            bonus -= weights["concede_penalty"]

        shaped[agent_id] = raw + bonus

        # Store agent proxy position for spread calculation
        positions[agent_id] = np.array([obs[IDX_OWN_GOAL_X], obs[IDX_OWN_GOAL_Z]])

    # 5. Spread bonus — iterate over consecutive teammate pairs (ids 0&1, 2&3)
    agent_ids = list(obs_dict.keys())
    for i in range(0, len(agent_ids) - 1, 2):
        a, b = agent_ids[i], agent_ids[i + 1]
        if a in positions and b in positions:
            dist = float(np.linalg.norm(positions[a] - positions[b]))
            if dist > SPREAD_DIST_THRESHOLD:
                shaped[a] = shaped.get(a, 0.0) + weights["spread"]
                shaped[b] = shaped.get(b, 0.0) + weights["spread"]

    # Update ball velocity reference
    new_vel = prev_ball_vel
    if obs_dict:
        first_obs = next(iter(obs_dict.values()))
        new_vel = np.array([first_obs[IDX_BALL_VEL_X], first_obs[IDX_BALL_VEL_Z]])

    return shaped, new_dist, new_vel


# ─────────────────────────────────────────────────────────────────────────────
# RLLib Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumSelfPlayCallback(DefaultCallbacks):
    """
    Combined callback handling:
      1. Per-episode env config (spawn ranges + opponent policy)
      2. Curriculum stage advancement when EMA reward exceeds threshold
      3. Self-play opponent pool rolling update (from train_selfplay.py)
    """

    # ── Episode start: configure environment for current stage ────────────────

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        global current_stage, tasks

        task = tasks[current_stage]
        trainer = getattr(worker, "_trainer", None)  # may be None in workers

        for env in base_env.get_unwrapped():
            # Apply opponent policy configuration
            CONFIG_FNS[task["config_fn"]](env, task, trainer)

            # Set ball and player spawn positions/velocities
            try:
                ball_cfg    = task["ranges"]["ball"]
                players_cfg = task["ranges"]["players"]

                env.env_channel.set_parameters(
                    ball_state=sample_pos_vel(ball_cfg),
                    players_states={
                        pid: sample_player(players_cfg[pid])
                        for pid in players_cfg
                    },
                )
            except Exception as e:
                # env_channel may not be available in all wrappers — soft fail
                print(f"[Curriculum] Could not set env parameters: {e}")

    # ── Episode end: inject shaped rewards into episode custom metrics ────────

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        task    = tasks[current_stage]
        weights = task["reward_weights"]
        # Log current reward weights so we can track them in TensorBoard
        for key, val in weights.items():
            episode.custom_metrics[f"rw_{key}"] = val
        episode.custom_metrics["curriculum_stage"] = current_stage

    # ── Train result: advance stage + update self-play opponent pool ──────────

    def on_train_result(self, **info):
        global current_stage, _ema_reward

        result   = info["result"]
        trainer  = info["trainer"]
        raw_mean = result.get("episode_reward_mean", 0.0)
        iteration = result.get("training_iteration", 0)

        # Smooth reward with EMA for stable threshold decisions
        if _ema_reward is None:
            _ema_reward = raw_mean
        else:
            _ema_reward = EMA_ALPHA * raw_mean + (1 - EMA_ALPHA) * _ema_reward

        # ── Curriculum advancement ────────────────────────────────────────────
        task      = tasks[current_stage]
        threshold = task.get("threshold")

        if threshold is not None and _ema_reward > threshold:
            next_stage = current_stage + 1
            if next_stage < len(tasks):
                print(
                    f"\n{'='*60}\n"
                    f"[Curriculum] Stage {current_stage} → {next_stage}\n"
                    f"  Completed : {task['name']}\n"
                    f"  Next      : {tasks[next_stage]['name']}\n"
                    f"  EMA reward: {_ema_reward:.4f} > threshold {threshold}\n"
                    f"{'='*60}\n"
                )
                current_stage = next_stage
                _ema_reward   = None  # reset EMA for new stage

        # ── Self-play opponent pool update (mirrors train_selfplay.py) ─────────
        # Only active once we reach the self-play stages (duo_vs_weak onward)
        if current_stage >= 5:
            if raw_mean > OPPONENT_UPDATE_REWARD_THRESHOLD and iteration % OPPONENT_UPDATE_INTERVAL == 0:
                print(
                    f"[Self-Play] Iter {iteration} | "
                    f"reward={raw_mean:.3f} — rotating opponent pool"
                )
                try:
                    trainer.set_weights({
                        "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                        "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                        "opponent_1": trainer.get_weights(["default"])["default"],
                    })
                except Exception as e:
                    print(f"[Self-Play] Could not update opponent pool: {e}")

        # ── Periodic logging ──────────────────────────────────────────────────
        if iteration % 10 == 0:
            print(
                f"[Iter {iteration:4d}] "
                f"stage={current_stage} ({task['name']}) | "
                f"reward_mean={raw_mean:.4f} | "
                f"ema={_ema_reward:.4f} | "
                f"timesteps={result.get('timesteps_total', 0):,}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Env factory: read reward weights from current curriculum stage
# ─────────────────────────────────────────────────────────────────────────────

def create_curriculum_env(config):
    """
    Thin wrapper around create_rllib_env that injects the current stage's
    reward_weights into the env config so the shaping wrapper picks them up.
    """
    merged = dict(config)
    merged["reward_weights"] = get_reward_weights(current_stage)
    return create_rllib_env(merged)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_curriculum_env)

    # Probe env for spaces (stage 0 is always single-agent)
    temp_env = create_curriculum_env({"reward_shaping": True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    print(f"Observation Space : {obs_space}")
    print(f"Action Space      : {act_space}")
    print(f"Curriculum stages : {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"  [{i}] {t['name']:20s}  threshold={t.get('threshold')}")

    analysis = tune.run(
        "PPO",
        name="PPO_curriculum_selfplay_shaped",
        config={
            # ── System ──────────────────────────────────────────────────────
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumSelfPlayCallback,

            # ── Algorithm ───────────────────────────────────────────────────
            "lr": 3e-4,
            "gamma": 0.998,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 0.5,
            "kl_coeff": 0.0,
            "train_batch_size": 16384,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 10,
            "grad_clip": 0.5,

            # ── Multi-agent self-play (active from stage 4 onward) ──────────
            # Policies are registered for all stages; earlier stages simply
            # never sample from the opponent pool.
            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(
                    lambda agent_id, *args, **kwargs: (
                        "default" if agent_id in [0, 1]
                        else np.random.choice(
                            ["default", "opponent_1", "opponent_2", "opponent_3"],
                            p=[0.50, 0.25, 0.15, 0.10],
                        )
                    )
                ),
                "policies_to_train": ["default"],
            },

            # ── Environment ─────────────────────────────────────────────────
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "reward_shaping": True,
            },

            # ── Model (larger than curriculum example; matches selfplay) ─────
            "model": {
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },

            # ── Rollout ──────────────────────────────────────────────────────
            # complete_episodes keeps shaped rewards coherent across full eps
            "rollout_fragment_length": 1000,
            "batch_mode": "truncate_episodes",
        },
        stop={
            "timesteps_total": 30_000_000,  # 30M total steps
            "time_total_s": 39_600,         # 11 h (12 h wall-time safety margin)
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="<path_to_checkpoint>",   # uncomment to resume
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(f"\nBest trial      : {best_trial}")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(f"Best checkpoint : {best_checkpoint}")
    print("Done.")