import random

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


# Q-value (Q) - learnt
#
#  Q(s, a) -> q
#
qvalue: dict[tuple[State, Action], float] = {}


# Alpha (α)
# Learning rate
#
ALPHA: float = 0.1


# Monte carlo update using sampled return
#
#   Q(S_t, A_t) <- Q(S_t, A_t) + α[R_t+1 + γG_t+1 - Q(S_t, A_t)]
#
# Steps
#   G_t = R_t+1 + γG_t+1
#   error = G_t - Q(A_t)
#   V(S_t) <- V(S_t) + α error
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

        # error = G_t - Q(S_t, A_t)
        error = g - qvalue[(step.state, step.action)]

        # Q(S_t, A_t) <- Q(S_t) + α error
        qvalue[(step.state, step.action)] = (
            qvalue[(step.state, step.action)] + ALPHA * error
        )


def action(s: State) -> Action:
    #  An action must be selected from a probability distribution
    #    based on π(a | s) for each action

    # Exploration
    EPSILON = 0.1

    if random.random() < EPSILON:
        return random.choice(actions)

    # Exploitation
    return max(actions, key=lambda a: qvalue.get((s, a), 0.0))


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

        mc_control(episode)
