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
#
class Step:
    def __init__(self, s: State, s_next: State, r: float):
        self.state = s  # Current state
        self.next_state = s_next  # Next state
        self.reward = r  # Reward from next state


# Bellman equation for td update
#
#   V(s) <- V(s) + α[R + γV(s') - V(s)]
#
# Steps
#
#   target = R + γV(s')
#   δ = error = target - V(s)
#   V(s) <- V(s) + α * δ
#
def td_update(step: Step):
    # target = R + γV(s')
    target = step.reward + GAMMA * value.get(step.next_state, 0)

    # δ = target - V(s)
    error = target - value.get(step.state, 0)

    # V(s) <- V(s) + α * δ
    value[step.state] = value.get(step.state, 0) + ALPHA * error


def train():
    environment = Environment(title="TD(0)")

    MAX_ITERATIONS = 3000
    MAX_STEPS = 100

    for i in range(MAX_ITERATIONS):
        s = environment.reset()

        step_no = 0

        while s not in environment.terminal_states() and step_no < MAX_STEPS:
            a = policy(s)

            environment.debug_print(
                f"Iteration: {i + 1}\nAction: {a.name}\nV(s): {value.get(s, 0)}"
            )

            # E(s, a) -> (s', r)
            (s_next, r) = environment.step(a)

            # TD update
            td_update(Step(s, s_next, r))

            s = s_next

            step_no += 1


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
