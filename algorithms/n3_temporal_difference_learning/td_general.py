from ..reinforcement_learning.mdp import (
    GAMMA,
    Action,
    State,
    actions,
    reward,
)


# Step
#
class Step:
    def __init__(self, s: State, a: Action, s_next: State, r: float):
        self.state = s  # Current state
        self.action = a  # Action taken at current state
        self.next_state = s_next  # Next state
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


# Q-value (Q) - learnt
#
#  Q(s, a) -> q
#
qvalue: dict[tuple[State, Action], float] = {}


# Alpha (α)
# Learning rate
#
ALPHA: float = 0.1


# Bellman equation for td update
#
#   Q(s, a) <- Q(s, a) + α[R + γQ(s', a') - Q(s, a)]
#
# Steps
#
#   target = R + γQ(s', a')
#   error = target - Q(s, a)
#
#   Q(s, a) <- Q(s, a) + α * error
#
# Q-values are updated from individual transitions
#
def td_update(step: Step, value_next: float):
    # target = R + γQ(s', a')
    target = step.reward + GAMMA * value_next
    # error = target - Q(s, a)
    error = target - qvalue[(step.state, step.action)]
    # Q(s, a) <- Q(s, a) + α * δ
    qvalue[(step.state, step.action)] = (
        qvalue[(step.state, step.action)] + ALPHA * error
    )


def action(s: State) -> Action:
    #  An action must be selected from a probability distribution
    #    based on Q(s, a) for each action
    for a in actions:
        _ = qvalue[(s, a)]

    return Action()


def train():
    environment = Environment()
    converged = False

    while not converged:
        environment.reset()

        s = environment.current_state

        while s != environment.terminal_state:
            a = action(s)

            # E(s, a) -> (s', r)
            (s_next, r) = environment.step(a)

            step = Step(s, a, s_next, r)

            # This value must be taken from Q table
            # Out of all possible actions one of the (s, a) pair must be selected
            # That part is handled in specific TD algorithm like SARSA, Expected SARSA or Q-learning
            value_next = 0.0
            td_update(step, value_next)
            s = s_next
