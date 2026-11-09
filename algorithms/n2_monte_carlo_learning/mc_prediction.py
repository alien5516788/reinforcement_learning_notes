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
    def __init__(self, s: State, r: float):
        self.state = s  # Current state
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


# Policy (π) - defined
#
#   π(a | s)
#
def policy(s: State, a: Action) -> float:
    # Must decide the policy based on value
    return 0.9


# Value (V) - learnt
#
#  V(s) -> v
#
value: dict[State, float] = {}


# Alpha (α)
# Learning rate
#
ALPHA: float = 0.1


# Monte carlo update using sampled return
#
#   V(S_t) <- V(S_t) + α[R_t+1 + γG_t+1 - V(S_t)]
#
# Steps
#   G_t = R_t+1 + γG_t+1
#   error = G_t - V(S_t)
#   V(S_t) <- V(S_t) + α error
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

        # error = G_t - V(S_t)
        error = g - value[step.state]

        # V(S_t) <- V(S_t) + α error
        value[step.state] = value[step.state] + ALPHA * error


def action(s: State) -> Action:
    #  An action must be selected from a probability distribution
    #    based on π(a | s) for each action
    for a in actions:
        _ = policy(s, a)

    return Action()


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
            episode.append(Step(s, r))

            s = s_next

        mc_prediction(episode)
