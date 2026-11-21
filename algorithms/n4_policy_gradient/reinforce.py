import math
import random

from find_the_dock import Action, Environment, State


# Policy (π_θ) - defined
#
#   softmax
#
def policy(s: State) -> Action:
    p = [policy_probability(s, a) for a in Action.all()]

    return random.choices(Action.all(), weights=p, k=1)[0]


def policy_probability(s: State, a: Action) -> float:
    # e^θ(s, a)
    numerator = math.exp(parameter.get((s, a), 0))

    # Σ_a' e^θ(s, a')
    denominator = 0.0

    for aa in Action.all():
        denominator += math.exp(parameter.get((s, aa), 0))

    # e^θ(s, a) / Σ_b e^θ(s, a')
    return numerator / denominator


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


# Alpha (α)
# Learning rate
#
ALPHA: float = 0.1


# Gamma (γ)
# Discount factor
#
GAMMA: float = 0.9


# Step
# Episodic Step
#
class Step:
    def __init__(self, s: State, a: Action, r: float):
        self.state = s  # Current state
        self.action = a  # Action taken at current state
        self.reward = r  # Reward from next state


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

        for a in Action.all():
            # I(a_t == a)
            if step.action == a:
                indicator = 1.0
            else:
                indicator = 0.0

            # ∇θ log π_θ(a_t | s_t)
            # ∂ log π_θ(a_t | s_t) /∂ θ(s_t, a)
            gradient = indicator - policy_probability(step.state, a)

            # α γ^t G_t ∇θ log π_θ(a_t | s_t)
            update = ALPHA * pow(GAMMA, t) * g_t * gradient

            # Accumulate parameter update
            gradients[(step.state, a)] = gradients.get((step.state, a), 0.0) + update

    # Apply accumulated updates
    for key, value in gradients.items():
        # θ <- θ + α γ^t G_t ∇θ log π_θ(a_t | s_t)
        parameter[key] = parameter.get(key, 0) + value


def train():
    environment = Environment(title="REINFORCE")

    MAX_EPISODES = 3000
    MAX_STEPS = 100

    for ep_no in range(MAX_EPISODES):
        s = environment.reset()

        # Generate one complete episode
        episode: list[Step] = []

        while s not in environment.terminal_states() and len(episode) < MAX_STEPS:
            a = policy(s)

            environment.debug_print(
                f"Episode: {ep_no + 1}"
                + f"\nAction: {a.name}"
                + f"\nProbability: {round(policy_probability(s, a), 4)}"
                + f"\n\nUp: {round(policy_probability(s, Action.UP), 4)}"
                + f"\nDown: {round(policy_probability(s, Action.DOWN), 4)}"
                + f"\nLeft: {round(policy_probability(s, Action.LEFT), 4)}"
                + f"\nRight: {round(policy_probability(s, Action.RIGHT), 4)}"
            )

            # E(s, a) -> (s', r)
            (s_next, r) = environment.step(a)

            # Add step
            episode.append(Step(s, a, r))

            s = s_next

        reinforce(episode)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
