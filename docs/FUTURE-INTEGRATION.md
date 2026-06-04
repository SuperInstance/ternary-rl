# Future Integration: ternary-rl

## Current State
Implements reinforcement learning with ternary action space {-1, 0, +1}: `TernaryEnvironment` with shaped rewards, `QTable` with greedy/epsilon-greedy policies, `TernaryAgent` with configurable exploration, SARSA and Q-learning update rules, and episode tracking.

## Integration Opportunities

### With ternary-cell / construct-core
Every `TernaryCell` is an RL agent. The cell's `tick()` cycle maps to: observe state → `QTable::best_action()` → execute action → receive reward (surprise signal) → `q_update()`. The ternary action space matches the cell's state transitions perfectly: Neg = decrease, Zero = maintain, Pos = increase. Over time, cells learn which actions maximize surprise (exploration) vs. stability (exploitation).

### With ternary-attention
Replace `QTable`'s flat state representation with attention-weighted state encoding. The agent attends to the most relevant features of the room state rather than treating the raw state as a monolithic key. This creates `TernaryAttentionRL` — a cell that learns *what to pay attention to* in its environment.

### With ternary-curriculum
The `TernaryEnvironment` can be progressively complexified using `ternary-curriculum`. Start with simple single-room environments, then multi-room, then multi-agent. The `step_count` and `max_steps` from the environment directly control curriculum pacing.

## Potential in Mature Systems
In room-as-codespace, ensign specialists are trained via RL. A kitchen ensign starts with a `QTable` trained in simulation (`TernaryEnvironment` with cooking-related states). When deployed to a real room, it continues learning via online Q-updates. The `TernaryReward` signal comes from `ternary-sensor` classifications: Negative = problem detected, Neutral = nominal, Positive = goal achieved. At Layer 0, the `QTable` compiles to a lookup table — no learning, just recall.

## Cross-Pollination Ideas
**Music × RL:** Learn to compose. State = current harmony, action = next chord (resolve/hold/tension), reward = voice-leading smoothness + harmonic surprise. The agent learns to compose ternary music that balances predictability and novelty. Connects to `ternary-music` and `flux-algebra-rs`.

**Evolution × RL:** `evolution-ternary` evolves the reward function while RL optimizes the policy. The co-evolution of reward and policy produces emergent behaviors that neither fixed-reward RL nor pure evolution would discover alone.

## Dependencies for Next Steps
- `ternary-cell` tick integration: RL agent as a cell behavior trait
- Multi-agent RL: multiple cells learning simultaneously with shared/different reward
- Function approximation for large state spaces (attention-based Q-network)
