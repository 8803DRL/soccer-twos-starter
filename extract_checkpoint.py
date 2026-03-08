"""
Utility script to extract policy weights from a Ray RLLib checkpoint
and save them as a PyTorch state dict for the Agent MP.

Usage:
    python extract_checkpoint.py <path_to_rllib_checkpoint>

This will create 'our_agent/checkpoint.pth' with the policy weights.
"""

import os
import sys
import pickle

import numpy as np
import torch
import ray
from ray.rllib.agents.ppo import PPOTrainer

from utils import create_rllib_env


def extract_from_rllib_checkpoint(checkpoint_path):
    """
    Load an RLLib checkpoint and extract the 'default' policy weights
    into a PyTorch state dict compatible with our PolicyNetwork.
    """
    ray.init(ignore_reinit_error=True)

    from ray import tune
    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    # Create a trainer with the same config used during training
    config = {
        "num_gpus": 0,
        "num_workers": 0,
        "framework": "torch",
        "multiagent": {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda _: "default",
            "policies_to_train": ["default"],
        },
        "env": "Soccer",
        "env_config": {},
        "model": {
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "relu",
            "vf_share_layers": True,
        },
    }

    trainer = PPOTrainer(config=config, env="Soccer")
    trainer.restore(checkpoint_path)

    # Get the default policy
    policy = trainer.get_policy("default")
    model = policy.model

    print("=== RLLib Model Architecture ===")
    print(model)

    # Extract weights
    state_dict = model.state_dict()
    print("\n=== Weight keys ===")
    for k, v in state_dict.items():
        print(f"  {k}: {v.shape}")

    # Save mapping info for debugging
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "our_agent")
    os.makedirs(out_dir, exist_ok=True)

    # Save the raw RLLib state dict
    rllib_path = os.path.join(out_dir, "rllib_state_dict.pth")
    torch.save(state_dict, rllib_path)
    print(f"\nSaved RLLib state dict to: {rllib_path}")

    # Also save weights in a format our PolicyNetwork can load
    # We need to map RLLib's naming to ours
    our_state_dict = {}
    for k, v in state_dict.items():
        # RLLib FullyConnectedNetwork naming convention:
        # _hidden_layers.0._model.0.weight -> fc1.weight
        # _hidden_layers.0._model.0.bias -> fc1.bias
        # _hidden_layers.1._model.0.weight -> fc2.weight
        # _hidden_layers.1._model.0.bias -> fc2.bias
        # _logits._model.0.weight -> action_head.weight
        # _logits._model.0.bias -> action_head.bias
        # _value_branch._model.0.weight -> value_head.weight
        # _value_branch._model.0.bias -> value_head.bias
        mapping = {
            "_hidden_layers.0._model.0.weight": "fc1.weight",
            "_hidden_layers.0._model.0.bias": "fc1.bias",
            "_hidden_layers.1._model.0.weight": "fc2.weight",
            "_hidden_layers.1._model.0.bias": "fc2.bias",
            "_logits._model.0.weight": "action_head.weight",
            "_logits._model.0.bias": "action_head.bias",
            "_value_branch._model.0.weight": "value_head.weight",
            "_value_branch._model.0.bias": "value_head.bias",
        }
        if k in mapping:
            our_state_dict[mapping[k]] = v
            print(f"  Mapped {k} -> {mapping[k]}")

    pth_path = os.path.join(out_dir, "checkpoint.pth")
    torch.save(our_state_dict, pth_path)
    print(f"\nSaved PolicyNetwork state dict to: {pth_path}")

    trainer.stop()
    ray.shutdown()

    return pth_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_checkpoint.py <path_to_rllib_checkpoint>")
        print("Example: python extract_checkpoint.py ./ray_results/PPO_selfplay_shaped/PPO_Soccer_.../checkpoint_000100/checkpoint-100")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Extracting weights from: {checkpoint_path}")
    output_path = extract_from_rllib_checkpoint(checkpoint_path)
    print(f"\nDone! Agent weights saved to: {output_path}")
    print("You can now test the agent with: python -m soccer_twos.watch -m our_agent")
