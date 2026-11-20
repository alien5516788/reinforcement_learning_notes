import random

from find_the_dock import Action, Environment, State


# Policy (π) - defined
#
#   ε-greedy
#
def policy(s: State) -> Action:
    # Exploration
    EPSILON = 0.1

    if random.random() < EPSILON:
        return random.choice(Action.all())

    # Exploitation
    # ISSUE: max favors first action when all values are equal
    return max(Action.all(), key=lambda a: qvalue.get((s, a), 0.0))


# Q-value (Q) - learnt
#
#  Q(s, a) -> q
#
qvalue: dict[tuple[State, Action], float] = {}


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
    def __init__(self, s: State, a: Action, r: float):
        self.state = s  # Current state
        self.action = a  # Action taken at current state
        self.reward = r  # Reward from next state


# Monte carlo update using sampled return
#
#   Q(S_t, A_t) <- Q(S_t, A_t) + α[R_t+1 + γG_t+1 - Q(S_t, A_t)]
#
# Steps
#   G_t = R_t+1 + γG_t+1
#   δ = error = G_t - Q(S_t, A_t)
#   Q(S_t, A_t) <- Q(S_t, A_t) + α * δ
#
# Q-value is learnt by calculating discounted return backward from the end of the episode
#   then updating the q-value of each step state using the return
#
def mc_control(episode: list[Step]):
    # We start as t+1 = terminal step which is not included in the episode
    #   and t = last step in the episode

    # G_t+1
    g = 0.0

    episode.reverse()
    for step in episode:
        # G_t = R_t+1 + γG_t+1
        g = step.reward + GAMMA * g

        # δ = G_t - Q(S_t, A_t)
        error = g - qvalue.get((step.state, step.action), 0)

        # Q(S_t, A_t) <- Q(S_t) + α * δ
        qvalue[(step.state, step.action)] = (
            qvalue.get((step.state, step.action), 0) + ALPHA * error
        )


def train():
    environment = Environment(title="MC Control")

    MAX_EPISODES = 3000
    MAX_STEPS = 100

    for ep_no in range(MAX_EPISODES):
        # Generate one complete episode
        episode: list[Step] = []

        s = environment.reset()

        while s not in environment.terminal_states() and len(episode) < MAX_STEPS:
            a = policy(s)

            environment.debug_print(
                f"Episode: {ep_no + 1}\nAction: {a.name}\nQ(s, a): {qvalue.get((s, a), 0)}"
            )

            # E(s, a) -> (s', r)
            (s_next, r) = environment.step(a)

            # Add step
            episode.append(Step(s, a, r))

            s = s_next

        mc_control(episode)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
