# ternary-rl

Reinforcement learning with ternary actions — Q-learning, SARSA, epsilon-greedy policies, reward shaping, and convergence detection over action space {-1, 0, +1}.

## Why This Exists

Most RL frameworks target continuous or large discrete action spaces. But many control and decision problems naturally have three actions: retreat / hold / advance, sell / hold / buy, decrease / maintain / increase. This crate provides a lean, self-contained RL toolkit for exactly that setting — ternary action spaces with ternary reward signals. Includes tabular Q-learning, SARSA, epsilon-greedy exploration, reward shaping, episode tracking, and convergence detection. No external dependencies, no unsafe code.

## Core Concepts

- **TritAction**: Ternary action — `Negative` (-1), `Zero` (0), `Positive` (+1).
- **TernaryReward**: Ternary feedback — `Negative`, `Neutral`, `Positive`.
- **TernaryEnvironment**: Simple discrete-state environment with a goal state, max steps, and distance-based reward shaping.
- **QTable**: State-action value table with `get`, `set`, `best_action`, `best_value`.
- **SARSA update**: On-policy temporal-difference learning: `Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]`.
- **Q-learning update**: Off-policy: `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]`.
- **TernaryAgent**: Combines Q-table, learning rate, discount factor, and epsilon-greedy policy.
- **Episode tracking**: Record (state, action, reward) tuples with total reward and success flag.
- **Convergence detection**: Checks if the variance of recent episode rewards falls below a threshold.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-rl = "0.1"
```

```rust
use ternary_rl::{
    TritAction, TernaryReward, TernaryEnvironment, TernaryAgent,
    Episode, detect_convergence, shape_reward,
};

fn main() {
    let mut env = TernaryEnvironment::new(3, 50);
    let mut agent = TernaryAgent::new(0.1, 0.9, 0.2);
    let mut episodes: Vec<Episode> = Vec::new();

    for _ in 0..200 {
        let mut state = env.reset();
        let mut episode = Episode::new();
        let mut action = agent.choose_action(state);

        loop {
            let (next_state, reward, done) = env.step(action);
            episode.record(state, action, reward.value());

            if done {
                episode.success = reward == TernaryReward::Positive;
                agent.learn_q(state, action, reward.value(), next_state);
                break;
            }

            let next_action = agent.choose_action(next_state);
            agent.learn_sarsa(state, action, reward.value(), next_state, next_action);
            state = next_state;
            action = next_action;
        }
        episodes.push(episode);
    }

    println!("Converged: {}", detect_convergence(&episodes, 10, 0.5));

    // Reward shaping: combine base reward with bonus signal
    let shaped = shape_reward(TernaryReward::Positive, TernaryReward::Positive); // 1.5
}
```

## API Overview

| Type / Function | Description |
|---|---|
| `TritAction` | Ternary action: `Negative`, `Zero`, `Positive` |
| `TernaryReward` | Ternary reward signal |
| `TernaryEnvironment` | Simple env with `reset()`, `step()`, goal state, max steps |
| `QTable` | State-action value table with `best_action()`, `best_value()` |
| `epsilon_greedy` | Policy: explore with probability ε, exploit otherwise |
| `sarsa_update` | On-policy TD update |
| `q_learning_update` | Off-policy TD update |
| `shape_reward` | Combine base + bonus ternary rewards |
| `Episode` | Step-by-step episode record with total reward and success flag |
| `detect_convergence` | Check reward variance over a sliding window |
| `TernaryAgent` | Bundles Q-table + hyperparameters + learning methods |

## How It Works

The **TernaryEnvironment** maintains a scalar state. Each `step` adds the action's value (-1, 0, or +1) to the state. A positive reward is given for reaching the goal, a negative reward for exceeding max steps, and distance-based shaping rewards for intermediate steps.

**Q-learning** updates the Q-value toward the best possible next value (off-policy), while **SARSA** uses the actual next action chosen by the policy (on-policy). Both use the standard TD update rule.

**Epsilon-greedy** exploration uses a deterministic pseudo-random number generator (no external RNG dependency) to decide between random exploration and greedy exploitation.

**Convergence detection** computes the variance of total rewards over the last `window` episodes. When variance drops below `threshold`, training is considered converged.

## Use Cases

- **Three-action game AI**: Train agents that choose between retreat / hold / advance.
- **Trading signal optimization**: Learn buy / hold / sell policies from simulated markets.
- **Robotics control**: Discrete three-valued motor control (backward / stop / forward) with reward-based learning.
- **Adaptive system tuning**: Learn to increase / maintain / decrease a parameter based on system performance feedback.

## Ecosystem

Part of the **SuperInstance** ternary computing suite:

- `ternary-lattice` — lattice structures for ternary values
- `ternary-codes` — error-correcting codes for ternary data
- `ternary-gradient` — gradient-free optimization on ternary landscapes
- `ternary-language` — ternary NLP and grammar processing
- `ternary-trees` — ternary decision trees and forests
- `ternary-transform` — wavelet, Fourier, and kernel transforms
- `ternary-planning` — planning and scheduling with ternary priorities
- `ternary-rl` — this crate
- `ternary-som` — self-organizing maps for ternary data
- `ternary-failure` — failure analysis with ternary classification

## License

MIT

## See Also
- **ternary-predict** — related
- **ternary-oracle** — related
- **ternary-ga** — related
- **ternary-fitness** — related
- **ternary-attention** — related

