# Markov Decision Process (MDP)
#
#   M = (S, A, P, R, gamma)
#
#   - MDP only gives the environment model, not a solution.
#   - The RL agent may not always have access to the World Model (P) and Reward (R) directly.
#     It need to to interact with and Environment (E) to obtain the reward and learn the P.
#

# 1. State (S)
#
#   S = {s1, s1, s3, ...}
#
class State:
    pass


# 2. Action (A)
#
#   A = {a1, a2, a3, ...}
#
class Action:
    pass


# 3. World Model / Transition Probability (P)
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

# Other parameters that RL algorithms uses

# 6. Alpha (α)
#
#   Learning rate
#
ALPHA: float = 0.1
