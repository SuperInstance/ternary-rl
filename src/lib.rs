#![forbid(unsafe_code)]

//! Reinforcement learning with ternary actions.

use std::collections::HashMap;

/// Ternary action space: {-1, 0, +1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TritAction {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}

impl TritAction {
    pub fn value(&self) -> i32 {
        *self as i32
    }

    pub fn all() -> [TritAction; 3] {
        [TritAction::Negative, TritAction::Zero, TritAction::Positive]
    }

    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            -1 => Some(TritAction::Negative),
            0 => Some(TritAction::Zero),
            1 => Some(TritAction::Positive),
            _ => None,
        }
    }
}

/// Ternary reward feedback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TernaryReward {
    Negative = -1,
    Neutral = 0,
    Positive = 1,
}

impl TernaryReward {
    pub fn value(&self) -> f64 {
        *self as i32 as f64
    }
}

/// A simple ternary environment with discrete states.
pub struct TernaryEnvironment {
    pub state: i32,
    pub terminal: bool,
    pub goal_state: i32,
    pub step_count: usize,
    pub max_steps: usize,
}

impl TernaryEnvironment {
    pub fn new(goal_state: i32, max_steps: usize) -> Self {
        Self { state: 0, terminal: false, goal_state, step_count: 0, max_steps }
    }

    pub fn reset(&mut self) -> i32 {
        self.state = 0;
        self.terminal = false;
        self.step_count = 0;
        self.state
    }

    pub fn step(&mut self, action: TritAction) -> (i32, TernaryReward, bool) {
        if self.terminal {
            return (self.state, TernaryReward::Neutral, true);
        }
        self.state += action.value();
        self.step_count += 1;

        let reward = if self.state == self.goal_state {
            self.terminal = true;
            TernaryReward::Positive
        } else if self.step_count >= self.max_steps {
            self.terminal = true;
            TernaryReward::Negative
        } else {
            // Shaping: closer = more positive
            let dist_before = (self.state - action.value() - self.goal_state).abs();
            let dist_after = (self.state - self.goal_state).abs();
            if dist_after < dist_before {
                TernaryReward::Positive
            } else if dist_after > dist_before {
                TernaryReward::Negative
            } else {
                TernaryReward::Neutral
            }
        };

        (self.state, reward, self.terminal)
    }
}

/// Q-Table for ternary action space.
pub struct QTable {
    table: HashMap<(i32, TritAction), f64>,
    default: f64,
}

impl QTable {
    pub fn new(default: f64) -> Self {
        Self { table: HashMap::new(), default }
    }

    pub fn get(&self, state: i32, action: TritAction) -> f64 {
        *self.table.get(&(state, action)).unwrap_or(&self.default)
    }

    pub fn set(&mut self, state: i32, action: TritAction, value: f64) {
        self.table.insert((state, action), value);
    }

    pub fn best_action(&self, state: i32) -> TritAction {
        let mut best = TritAction::Zero;
        let mut best_val = self.get(state, TritAction::Zero);
        for &a in &TritAction::all() {
            let v = self.get(state, a);
            if v > best_val {
                best_val = v;
                best = a;
            }
        }
        best
    }

    pub fn best_value(&self, state: i32) -> f64 {
        self.get(state, self.best_action(state))
    }
}

/// Epsilon-greedy policy.
pub fn epsilon_greedy(q: &QTable, state: i32, epsilon: f64) -> TritAction {
    if rand_bool(epsilon) {
        random_action()
    } else {
        q.best_action(state)
    }
}

/// Use thread-local seed to avoid unsafe.
use std::cell::Cell;

thread_local! {
    static SEED: Cell<u64> = Cell::new(42);
    static COUNTER: Cell<u64> = Cell::new(0);
}

fn rand_bool(probability: f64) -> bool {
    SEED.with(|s| {
        let old = s.get();
        let new = old.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.set(new);
        ((new & 0xFFFF) as f64 / 65536.0) < probability
    })
}

fn random_action() -> TritAction {
    COUNTER.with(|c| {
        let old = c.get();
        c.set(old + 1);
        let v = (old.wrapping_add(1).wrapping_mul(2654435761)) % 3;
        match v {
            0 => TritAction::Negative,
            1 => TritAction::Zero,
            _ => TritAction::Positive,
        }
    })
}

/// SARSA update rule.
pub fn sarsa_update(
    q: &mut QTable,
    state: i32,
    action: TritAction,
    reward: f64,
    next_state: i32,
    next_action: TritAction,
    alpha: f64,
    gamma: f64,
) {
    let old = q.get(state, action);
    let next_q = q.get(next_state, next_action);
    let new = old + alpha * (reward + gamma * next_q - old);
    q.set(state, action, new);
}

/// Q-learning update (off-policy).
pub fn q_learning_update(
    q: &mut QTable,
    state: i32,
    action: TritAction,
    reward: f64,
    next_state: i32,
    alpha: f64,
    gamma: f64,
) {
    let old = q.get(state, action);
    let max_next = q.best_value(next_state);
    let new = old + alpha * (reward + gamma * max_next - old);
    q.set(state, action, new);
}

/// Reward shaping with ternary feedback.
pub fn shape_reward(base: TernaryReward, bonus: TernaryReward) -> f64 {
    base.value() + 0.5 * bonus.value()
}

/// Episode tracking.
#[derive(Debug, Clone)]
pub struct Episode {
    pub steps: Vec<(i32, TritAction, f64)>, // (state, action, reward)
    pub total_reward: f64,
    pub success: bool,
}

impl Episode {
    pub fn new() -> Self {
        Self { steps: Vec::new(), total_reward: 0.0, success: false }
    }

    pub fn record(&mut self, state: i32, action: TritAction, reward: f64) {
        self.total_reward += reward;
        self.steps.push((state, action, reward));
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }
}

/// Convergence detection: checks if last N episodes have similar average rewards.
pub fn detect_convergence(episodes: &[Episode], window: usize, threshold: f64) -> bool {
    if episodes.len() < window {
        return false;
    }
    let recent: Vec<f64> = episodes.iter().rev().take(window).map(|e| e.total_reward).collect();
    let mean = recent.iter().sum::<f64>() / recent.len() as f64;
    let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;
    variance < threshold
}

/// TernaryAgent: a basic RL agent.
pub struct TernaryAgent {
    pub q: QTable,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
}

impl TernaryAgent {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64) -> Self {
        Self { q: QTable::new(0.0), alpha, gamma, epsilon }
    }

    pub fn choose_action(&self, state: i32) -> TritAction {
        epsilon_greedy(&self.q, state, self.epsilon)
    }

    pub fn learn_sarsa(&mut self, s: i32, a: TritAction, r: f64, s2: i32, a2: TritAction) {
        sarsa_update(&mut self.q, s, a, r, s2, a2, self.alpha, self.gamma);
    }

    pub fn learn_q(&mut self, s: i32, a: TritAction, r: f64, s2: i32) {
        q_learning_update(&mut self.q, s, a, r, s2, self.alpha, self.gamma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_action_all() {
        assert_eq!(TritAction::all().len(), 3);
    }

    #[test]
    fn test_trit_from_i32_roundtrip() {
        for v in [-1i32, 0, 1] {
            assert_eq!(TritAction::from_i32(v).unwrap().value(), v);
        }
    }

    #[test]
    fn test_environment_reset() {
        let mut env = TernaryEnvironment::new(5, 100);
        env.step(TritAction::Positive);
        let s = env.reset();
        assert_eq!(s, 0);
        assert!(!env.terminal);
    }

    #[test]
    fn test_environment_goal() {
        let mut env = TernaryEnvironment::new(2, 100);
        let (_, reward, done) = env.step(TritAction::Positive);
        assert!(!done);
        let (state, reward, done) = env.step(TritAction::Positive);
        assert_eq!(state, 2);
        assert_eq!(reward, TernaryReward::Positive);
        assert!(done);
    }

    #[test]
    fn test_environment_max_steps() {
        let mut env = TernaryEnvironment::new(100, 2);
        env.step(TritAction::Zero);
        let (_, reward, done) = env.step(TritAction::Zero);
        assert_eq!(reward, TernaryReward::Negative);
        assert!(done);
    }

    #[test]
    fn test_environment_terminal_noop() {
        let mut env = TernaryEnvironment::new(1, 100);
        env.step(TritAction::Positive);
        let (s, r, done) = env.step(TritAction::Negative);
        assert!(done);
        assert_eq!(r, TernaryReward::Neutral);
    }

    #[test]
    fn test_qtable_default() {
        let q = QTable::new(5.0);
        assert_eq!(q.get(0, TritAction::Positive), 5.0);
    }

    #[test]
    fn test_qtable_set_get() {
        let mut q = QTable::new(0.0);
        q.set(1, TritAction::Positive, 10.0);
        assert_eq!(q.get(1, TritAction::Positive), 10.0);
    }

    #[test]
    fn test_qtable_best_action() {
        let mut q = QTable::new(0.0);
        q.set(0, TritAction::Negative, 1.0);
        q.set(0, TritAction::Zero, 5.0);
        q.set(0, TritAction::Positive, 3.0);
        assert_eq!(q.best_action(0), TritAction::Zero);
    }

    #[test]
    fn test_sarsa_update() {
        let mut q = QTable::new(0.0);
        q.set(0, TritAction::Positive, 1.0);
        q.set(1, TritAction::Zero, 2.0);
        sarsa_update(&mut q, 0, TritAction::Positive, 1.0, 1, TritAction::Zero, 0.1, 0.9);
        let updated = q.get(0, TritAction::Positive);
        // 1.0 + 0.1 * (1.0 + 0.9*2.0 - 1.0) = 1.0 + 0.1 * 1.8 = 1.18
        assert!((updated - 1.18).abs() < 1e-6);
    }

    #[test]
    fn test_q_learning_update() {
        let mut q = QTable::new(0.0);
        q.set(0, TritAction::Positive, 0.0);
        q.set(1, TritAction::Negative, 5.0);
        q_learning_update(&mut q, 0, TritAction::Positive, 1.0, 1, 0.5, 0.9);
        let updated = q.get(0, TritAction::Positive);
        // 0.0 + 0.5 * (1.0 + 0.9*5.0 - 0.0) = 0.5 * 5.5 = 2.75
        assert!((updated - 2.75).abs() < 1e-6);
    }

    #[test]
    fn test_reward_shaping() {
        let r = shape_reward(TernaryReward::Positive, TernaryReward::Positive);
        assert!((r - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_reward_shaping_negative() {
        let r = shape_reward(TernaryReward::Negative, TernaryReward::Negative);
        assert!((r - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn test_episode_tracking() {
        let mut ep = Episode::new();
        ep.record(0, TritAction::Positive, 1.0);
        ep.record(1, TritAction::Positive, 1.0);
        assert_eq!(ep.len(), 2);
        assert!((ep.total_reward - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_convergence_detected() {
        let episodes: Vec<Episode> = (0..10).map(|_| {
            let mut ep = Episode::new();
            ep.total_reward = 5.0;
            ep
        }).collect();
        assert!(detect_convergence(&episodes, 5, 1.0));
    }

    #[test]
    fn test_convergence_not_detected() {
        let episodes: Vec<Episode> = (0..10).map(|i| {
            let mut ep = Episode::new();
            ep.total_reward = i as f64 * 10.0;
            ep
        }).collect();
        assert!(!detect_convergence(&episodes, 5, 1.0));
    }

    #[test]
    fn test_convergence_too_few() {
        let episodes = vec![Episode::new()];
        assert!(!detect_convergence(&episodes, 5, 1.0));
    }

    #[test]
    fn test_agent_basic() {
        let mut agent = TernaryAgent::new(0.1, 0.9, 0.0);
        let action = agent.choose_action(0);
        agent.learn_q(0, action, 1.0, 1);
        assert_ne!(agent.q.get(0, action), 0.0);
    }

    #[test]
    fn test_ternary_reward_values() {
        assert_eq!(TernaryReward::Negative.value(), -1.0);
        assert_eq!(TernaryReward::Neutral.value(), 0.0);
        assert_eq!(TernaryReward::Positive.value(), 1.0);
    }

    #[test]
    fn test_agent_sarsa_learning() {
        let mut agent = TernaryAgent::new(0.1, 0.9, 0.0);
        agent.q.set(0, TritAction::Positive, 0.0);
        agent.q.set(1, TritAction::Zero, 0.0);
        agent.learn_sarsa(0, TritAction::Positive, 1.0, 1, TritAction::Zero);
        assert_ne!(agent.q.get(0, TritAction::Positive), 0.0);
    }

    #[test]
    fn test_episode_success_flag() {
        let mut ep = Episode::new();
        ep.success = true;
        assert!(ep.success);
    }
}
