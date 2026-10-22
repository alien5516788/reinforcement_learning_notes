pub mod dynamic_programming;
pub mod monte_carlo_learning;
pub mod temporal_difference_learning;


// Markov Decision Process
/*
 * M = (S, A, P, R, gamma)
 *
 * MDP only give the environment structure, not solution.
 */

// 1. State (S)
/*
 * S = {s1, s1, s3, ...}
 */
#[derive(Eq, PartialEq, Clone, Hash)]
struct State;

// 2. Action (A
/*
 * A = {a1, a2, a3, ...}
 */
#[derive(Eq, PartialEq, Clone, Hash)]
struct Action;

// 3. World Model (P)
/*
 * P(s' | s, a)
 */
fn world_model(_s: &State, _a: &Action, _s_next: &State) -> f32 {
    0.5
}

// 4. Reward (R)
/*
 * R(s, a) -> r
 */
fn reward(_s: &State, _a: &Action, _s_next: &State) -> f32 {
    -1.0
}

// 5. Gamma (γ)
/*
 * Discount factor
 */
const GAMMA: f32 = 0.9;

// 3, 4. Joint World Model
/*
 * P(s', r | s, a)
 */
fn _world_model(_s: &State, _a: &Action, _s_next: &State, _r: f32) -> f32 {
    0.5
}

// Bellman Equation
/*
 * Bellman equation define the fundamental learning flow using MDP
 *
 *   V_π(s) <- Eπ[R_t+1 + γV_π(S_t+1) | S_t = s]
 *
 *   This can be defined as recusive if it's calculated repeatedly on next state
 *   But most of the time V(s') the existing entry of a table or an nn
 *
 * The equation has variations based on the algorithm, but the basic structure is same
 */


fn main() {
    /*
     *   - Dynamic Programming (DP)
     *     - Policy Evaluation
     *     - Policy Iteration
     *     - Value Iteration
     *   - Monte Carlo Learning (MCL)
     *     - Monte Carlo Prediction
     *     - Monte Carlo Control
     *   - Temporal Dfference Learning (TD)
     *     - SARSA
     *     - Q-learning
     *   - DQN: TODO
     *   - Policy Gradient: TODO
     *   - Actor-Critic: TODO
     *   - PPO: TODO
     */
    println!("Reinforcement Learning")
}
