"""
Agent MP - PPO Self-Play Agent for SoccerTwos.

Loads a trained RLLib PPO checkpoint and provides inference via the AgentInterface.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from soccer_twos import AgentInterface


class PolicyNetwork(nn.Module):
    """
    Mirrors the RLLib default fully-connected network architecture.
    Must match the model config used during training: [512, 512] + relu.
    """

    def __init__(self, obs_size, act_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.action_head = nn.Linear(512, act_size)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.action_head(x), self.value_head(x)


class AgentMP(AgentInterface):
    """
    Agent MP: Trained with PPO + Self-Play + Dense Reward Shaping.
    Implements the AgentInterface for evaluation.
    """

    def __init__(self, env):
        super().__init__()
        self.name = "Agent MP"

        # Environment spaces
        self.obs_size = env.observation_space.shape[0]  # 336
        self.act_space = env.action_space  # MultiDiscrete([3, 3, 3])
        total_act_size = sum(self.act_space.nvec)  # 3+3+3 = 9

        # Build policy network
        self.model = PolicyNetwork(self.obs_size, total_act_size)
        self.model.eval()

        # Try to load checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load model weights from the checkpoint file."""
        checkpoint_dir = os.path.dirname(os.path.abspath(__file__))

        # Try loading PyTorch state dict first
        pth_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if os.path.isfile(pth_path):
            print(f"[Agent MP] Loading checkpoint from {pth_path}")
            self.model.load_state_dict(torch.load(pth_path, map_location="cpu"))
            return

        # Try loading from RLLib pickle checkpoint
        pkl_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
        if os.path.isfile(pkl_path):
            print(f"[Agent MP] Loading RLLib checkpoint from {pkl_path}")
            with open(pkl_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            # Extract 'default' policy weights
            if "default" in checkpoint_data:
                weights = checkpoint_data["default"]
                self._load_rllib_weights(weights)
            return

        print("[Agent MP] WARNING: No checkpoint found! Using random initialization.")

    def _load_rllib_weights(self, weights_dict):
        """
        Load weights from RLLib checkpoint format into our PolicyNetwork.
        RLLib stores weights as numpy arrays with specific naming conventions.
        """
        try:
            state_dict = {}
            # Map RLLib weight names to our model's parameter names
            rllib_to_ours = {
                "default_model.fc1.weight": "fc1.weight",
                "default_model.fc1.bias": "fc1.bias",
                "default_model.fc2.weight": "fc2.weight",
                "default_model.fc2.bias": "fc2.bias",
                "default_model.action_head.weight": "action_head.weight",
                "default_model.action_head.bias": "action_head.bias",
                "default_model.value_head.weight": "value_head.weight",
                "default_model.value_head.bias": "value_head.bias",
            }
            for rllib_name, our_name in rllib_to_ours.items():
                if rllib_name in weights_dict:
                    state_dict[our_name] = torch.tensor(weights_dict[rllib_name])

            if state_dict:
                self.model.load_state_dict(state_dict, strict=False)
                print("[Agent MP] Successfully loaded RLLib weights.")
            else:
                print("[Agent MP] WARNING: Could not map RLLib weight names.")
        except Exception as e:
            print(f"[Agent MP] Error loading RLLib weights: {e}")

    def act(self, observation):
        """
        The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        with torch.no_grad():
            for player_id in observation:
                obs = observation[player_id]
                state = torch.from_numpy(obs).float().unsqueeze(0)
                logits, _ = self.model(state)

                # Split logits into action branches [3, 3, 3]
                branch_logits = torch.split(logits[0], list(self.act_space.nvec))
                action = []
                for branch in branch_logits:
                    probs = F.softmax(branch, dim=-1)
                    act = torch.argmax(probs).item()
                    action.append(act)
                actions[player_id] = np.array(action)

        return actions
