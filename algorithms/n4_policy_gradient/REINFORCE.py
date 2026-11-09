import math

from ..reinforcement_learning.mdp import (
    GAMMA,
    Action,
    State,
    actions,
    reward,
)


# Step
# Episodic Step
#
class Step:
    def __init__(self, s: State, a: Action, r: float):
        self.state = s  # Current state
        self.action = a  # Action taken at current state
        self.reward = r  # Reward from next state


# Environemnt
#
#   E(s, a) -> (s', r)
#
class Environment:
    def __init__(self):
        self.current_state: State = State()  # Or initial
        self.terminal_state: State = State()

    def step(self, a: Action) -> tuple[State, float]:
        s_next = State()
        r = reward(self.current_state, a, s_next)
        self.current_state = s_next
        return (self.current_state, r)

    def reset(self):
        pass


# Policy Parameter (θ) - learnt
#
#  θ(s, a) -> n
#
# Each state-action pair has a parameter in this tabular implementation.
# These parameters determine the policy probabilities.
#
# In neural-network implementations, θ represents the network's
# trainable weights and biases, while π_θ is the policy represented
# by that network.
#
parameter: dict[tuple[State, Action], float] = {}


# Policy (π_θ) - softmax policy derived from θ
#
#   π_θ(a | s) -> e^θ(s, a) / Σ_a' e^θ(s, a')
#
def policy(s: State, a: Action) -> float:
    # e^θ(s, a)
    numerator = math.exp(parameter[(s, a)])

    # Σ_a' e^θ(s, a')
    denominator = 0.0

    for aa in actions:
        denominator += math.exp(parameter[(s, aa)])

    # e^θ(s, a) / Σ_b e^θ(s, a')
    return numerator / denominator


def action(s: State) -> Action:
    # An action is sampled from the probability distribution
    #   π(a | s) for the current state.
    for a in actions:
        _ = policy(s, a)

    return Action()


# Alpha (α)
# Learning rate
#
ALPHA: float = 0.1


# Policy Gradient Update
#
# REINFORCE updates the policy parameters using sampled returns.
#
#   G_t = R_t+1 + γG_t+1
#
#   θ <- θ + α γ^t G_t ∇θ log π_θ(a_t | s_t)
#
# For a tabular softmax policy:
#
#   ∇θ log π_θ(a_t | s_t) = ∂ log πθ(a_t | s_t) /∂ θ(s, a)
#                              after deriving
#                         = I(a_t == a) - π_θ(a | s), when s == s_t
#
# The gradient is accumulated over the episode before
# updating θ.
#
def reinforce(episode: list[Step]):
    returns = []

    # G_t+1
    g = 0.0
    episode.reverse()

    for t in range(len(episode)):
        # G_t = R_t+1 + γG_t+1
        g = episode[t].reward + GAMMA * g

        returns.insert(0, g)

    gradients: dict[tuple[State, Action], float] = {}

    episode.reverse()
    for t, step in enumerate(episode):
        # G_t
        g_t = returns[t]

        for a in actions:
            # I(a_t == a)
            if step.action == a:
                indicator = 1.0
            else:
                indicator = 0.0

            # ∇θ log π_θ(a_t | s_t)
            # ∂ log π_θ(a_t | s_t) /∂ θ(s_t, a)
            gradient = indicator - policy(step.state, a)

            # α γ^t G_t ∇θ log π_θ(a_t | s_t)
            update = ALPHA * pow(GAMMA, t) * g_t * gradient

            # Accumulate parameter update
            gradients[(step.state, a)] = gradients.get((step.state, a), 0.0) + update

    # Apply accumulated updates
    for key, value in gradients.items():
        # θ <- θ + α γ^t G_t ∇θ log π_θ(a_t | s_t)
        parameter[key] = parameter[key] + value


def train():
    environment = Environment()
    converged = False

    while not converged:
        environment.reset()

        # Generate one complete episode
        episode: list[Step] = []
        s = environment.current_state

        while s != environment.terminal_state:
            a = action(s)

            # E(s, a) -> (s', r)
            (s_next, r) = environment.step(a)

            # Add step
            episode.append(Step(s, a, r))

            s = s_next

        reinforce(episode)
