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
#
class Step:
    def __init__(self, s: State, a: Action, s_next: State, r: float):
        self.state = s  # Current state
        self.action = a  # Action taken at current state
        self.next_state = s_next  # Next state
        self.reward = r  # Reward from next state


# Bellman equation for td update
#
#   Q(s, a) <- Q(s, a) + α[R + γQ(s', a') - Q(s, a)]
#
# Steps
#
#   target = R + γQ(s', a')
#   δ = error = target - Q(s, a)
#   Q(s, a) <- Q(s, a) + α * δ
#
# Note
#   - On-policy, Model-free
#   - S -> A -> R -> S -> A
#   - SARSA calculate both current qvalue and next qvalue using the same policy
#
def td_update(step: Step, a_next: Action):
    # target = R + γQ(s', a')
    target = step.reward + GAMMA * qvalue.get((step.next_state, a_next), 0)

    # δ = target - Q(s, a)
    error = target - qvalue.get((step.state, step.action), 0)

    # Q(s, a) <- Q(s, a) + α * δ
    qvalue[(step.state, step.action)] = (
        qvalue.get((step.state, step.action), 0) + ALPHA * error
    )


def train():
    environment = Environment(title="SARSA")

    MAX_ITERATIONS = 3000
    MAX_STEPS = 100

    for i in range(MAX_ITERATIONS):
        s = environment.reset()
        a = policy(s)

        step_no = 0

        while s not in environment.terminal_states() and step_no < MAX_STEPS:
            environment.debug_print(
                f"Iteration: {i + 1}\nAction: {a.name}\nQ(s, a): {qvalue.get((s, a), 0)}"
            )

            # E(s, a) -> (s', r)
            (s_next, r) = environment.step(a)

            a_next = policy(s_next)

            # TD update
            td_update(Step(s, a, s_next, r), a_next)

            s = s_next
            a = a_next

            step_no += 1


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
