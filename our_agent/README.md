# Agent MP

## Agent Information
- **Agent Name**: Agent MP
- **Authors**: (to be filled)
- **Emails**: (to be filled)

## Description

Agent MP is trained using **PPO (Proximal Policy Optimization)** with:
- **Self-Play**: Opponent pool with archived past policies to prevent overfitting
- **Dense Reward Shaping**: Custom reward wrapper that adds intermediate signals:
  - Ball proximity reward (encouraging approaching the ball)
  - Ball-to-goal alignment (encouraging kicking toward opponent goal)
  - Goal proximity bonus (rewarding ball near opponent goal)
  - Existential penalty (encouraging urgency)

### Architecture
- Fully connected network: [512, 512] with ReLU activation
- Shared value/policy layers
- GAE (λ=0.95) with high discount (γ=0.998) for long-horizon credit assignment

### Training
- Algorithm: PPO via Ray RLLib
- Framework: PyTorch
- Environment: SoccerTwos (multiagent_player, 4 individual agents)
- Hardware: NVIDIA A40 GPU, 32 CPUs, 128GB RAM on PACE cluster

## Requirements
- torch
- numpy
- soccer-twos
