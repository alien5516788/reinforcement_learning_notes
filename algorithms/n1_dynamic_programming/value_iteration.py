from math import inf

from ..reinforcement_learning.mdp import (
    GAMMA,
    Action,
    State,
    actions,
    reward,
    states,
    world_model,
)

# Value (V) - learnt
#
#  V(s) -> v
#
value: dict[State, float] = {}


# Bellman equation for value iteration
#
#   Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
#   V(s) <- max_a Q(s, a)
#
# Value is learnt by continuously improving the value of each state
#   by choosing the action with the highest expected return.
#
def value_iteration():
    for s in states:
        # Place holder until calculated
        v = -inf  # max_a Q(s, a)

        for a in actions:
            # Q(s, a)
            q = 0.0

            # Σ_s'
            for s_next in states:
                # Q(s, a) =+ Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
                q += world_model(s, a, s_next) * (
                    reward(s, a, s_next) + (GAMMA * value[s_next])
                )

            # max_a Q(s, a)
            v = max(q, v)

        # V(s) <- max_a Q(s, a)
        value[s] = v


def train():
    converged = False

    while not converged:
        value_iteration()


# ISSUE: Value iteration updates v in place. and next iterations uses new value instead of evaluating with old value
# TODO: Add a policy to use the learnt value
