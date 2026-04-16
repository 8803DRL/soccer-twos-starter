## 整体目标

这是一个用于训练 **SoccerTwos（2v2 足球游戏）** AI 的强化学习系统，核心思路是：**课程学习（Curriculum Learning）+ 密集奖励塑形（Dense Reward Shaping）+ 自我博弈（Self-Play）** 三者结合。

---

## 课程学习：7个阶段渐进训练

训练从简单到复杂，分7个阶段，每个阶段达到奖励阈值后自动晋级：

| 阶段 | 名称 | 场景 | 阈值 |
|------|------|------|------|
| 0 | solo_chaser | 1个agent，球在附近，无对手 | 0.6 |
| 1 | solo_shooter | 1个agent，球在对方球门附近，学习射门 | 1.0 |
| 2 | solo_vs_static | 1个agent vs 随机对手 | 1.2 |
| 3 | duo_coordinated | 2v0，两个队友学习配合 | 1.4 |
| 4 | duo_vs_static | 2v2，对手随机策略 | 1.5 |
| 5 | duo_vs_weak | 2v2，对手是早期存档策略 | 1.6 |
| 6 | full_selfplay | 2v2，完整滚动自我博弈 | 无（终点） |

阶段晋级使用 **EMA（指数移动平均）** 平滑奖励，避免单次噪声触发晋级：

```python
_ema_reward = EMA_ALPHA * raw_mean + (1 - EMA_ALPHA) * _ema_reward
```

---

## 密集奖励塑形：`shape_rewards()`

原始环境只在进球/失球时给奖励（稀疏），这里叠加了5种密集奖励信号：

```
1. approach        → 每步向球靠近就给奖励（引导追球）
2. touch           → 球速突然加快说明踢到了球，给奖励
3. goal_alignment  → agent、球、对方球门三点对齐时给奖励（引导射门方向）
4. concede_penalty → 失球时额外惩罚（比原始奖励更强的信号）
5. spread          → 两个队友距离超过阈值时各自获奖（防止扎堆）
```

随着阶段推进，`approach/touch` 权重逐渐降低，`concede_penalty` 逐渐升高——前期鼓励探索，后期强调竞技。

---

## 自我博弈：对手池滚动更新

维护4个策略槽：`default / opponent_1 / opponent_2 / opponent_3`

**对手采样概率**：`[50%, 25%, 15%, 10%]`，主要对战当前最强版本的自己，同时保留历史弱版本增加多样性。

**滚动更新逻辑**（每20轮、奖励超过0.3时触发）：
```
opponent_3 ← opponent_2   # 最老的被淘汰
opponent_2 ← opponent_1
opponent_1 ← default      # 当前策略存入历史
```

---

## 回调类 `CurriculumSelfPlayCallback`

这是整个系统的调度核心，挂载在RLLib训练循环上：

- `on_episode_start` → 根据当前阶段配置对手策略和球/球员初始位置
- `on_episode_end` → 记录当前阶段的奖励权重到 TensorBoard
- `on_train_result` → 判断是否晋级 + 触发对手池更新 + 周期性日志

---