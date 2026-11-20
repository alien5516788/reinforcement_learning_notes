import random

from find_the_dock import Action, Environment, State


# Policy (π) - defined
#
#   Uniform Random
#
def policy(s: State) -> Action:
    return random.choice(Action.all())


# Value (V) - learnt
#
#  V(s) -> v
#
value: dict[State, float] = {}


# Alpha (α)
# Learning rate
#
ALPHA: float = 0.1


# Gamma (γ)
# Discount factor
#
GAMMA: float = 0.9


# Step
class Step:
    def __init__(self, s: State, r: float):
        self.state = s  # Current state
        self.reward = r  # Reward from next state


# Monte carlo update using sampled return
#
#   V(S_t) <- V(S_t) + α[R_t+1 + γG_t+1 - V(S_t)]
#
# Steps
#   G_t = R_t+1 + γG_t+1
#   δ = error = G_t - V(S_t)
#   V(S_t) <- V(S_t) + α * δ
#
# Value is learnt by calculating discounted return backward from the end of the episode
#   then updating the value of each step state using the return
#
def mc_prediction(episode: list[Step]):
    # We start as t+1 = terminal step which is not included in the episode
    #   and t = last step in the episode

    # G_t+1
    g = 0.0

    episode.reverse()
    for step in episode:
        # G_t = R_t+1 + γG_t+1
        g = step.reward + GAMMA * g

        # δ = G_t - V(S_t)
        error = g - value.get(step.state, 0)

        # V(S_t) <- V(S_t) + α * δ
        value[step.state] = value.get(step.state, 0) + ALPHA * error


def train():
    environment = Environment(title="MC Prediction")

    MAX_EPISODES = 3000
    MAX_STEPS = 100

    for ep_no in range(MAX_EPISODES):
        # Generate one complete episode
        episode: list[Step] = []

        s = environment.reset()

        while s not in environment.terminal_states() and len(episode) < MAX_STEPS:
            a = policy(s)

            environment.debug_print(
                f"Episode: {ep_no + 1}\nAction: {a.name}\nV(s): {value.get(s, 0)}"
            )

            # E(s, a) -> (s', r)
            (s_next, r) = environment.step(a)

            # Add step
            episode.append(Step(s, r))

            s = s_next

        mc_prediction(episode)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
