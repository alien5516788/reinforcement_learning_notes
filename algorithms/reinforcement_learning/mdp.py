# Markov Decision Process
#
#   M = (S, A, P, R, gamma)
#
#   MDP only give the environment structure, not solution.
#

# 1. State (S)
#
#   S = {s1, s1, s3, ...}
#
class State:
    pass


states: list[State] = []


# 2. Action (A)
#
#   A = {a1, a2, a3, ...}
#
class Action:
    pass


actions: list[Action] = []


# 3. World Model (P)
#
#   P(s' | s, a)
#
def world_model(s: State, a: Action, s_next: State) -> float:
    return 0.5


# 4. Reward (R)
#
#   R(s, a, s') -> r
#
def reward(s: State, a: Action, s_next: State) -> float:
    return -0.1


# 5. Gamma (γ)
#
#   Discount factor
#
GAMMA: float = 0.9


# 3, 4. Joint World Model
#
#   P(s', r | s, a)
#
def _world_model(s: State, a: Action, s_next: State, r: float) -> float:
    return 0.5


# Bellman Equation
#   Bellman equation define the fundamental learning flow using MDP
#
#     V_π(s) <- Eπ[R_t+1 + γV_π(S_t+1) | S_t = s]
#
#   This can be defined as recusive if it's calculated repeatedly on next state
#   But most of the time V(s') the existing entry of a table or an nn
#   The equation has variations based on the algorithm, but the basic structure is same
#


# Common Algorithms
#
#   - Dynamic Programming (DP)
#     - Policy Evaluation
#     - Policy Iteration
#     - Value Iteration
#   - Monte Carlo Learning (MCL)
#     - Monte Carlo Prediction
#     - Monte Carlo Control
#   - Temporal Dfference Learning (TD)
#     - SARSA
#     - Q-learning
#   - DQN: TODO
#   - Policy Gradient
#   - Actor-Critic: TODO
#   - PPO: TODO
#
