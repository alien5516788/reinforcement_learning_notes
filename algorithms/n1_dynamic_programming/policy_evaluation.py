from ..reinforcement_learning.mdp import (
    GAMMA,
    Action,
    State,
    actions,
    reward,
    states,
    world_model,
)


# Policy (π) - defined
#
#   π(a | s)
#
def policy(s: State, a: Action) -> float:
    return 0.9


# Value (V) - learnt
#
#  V(s) -> v
#
value: dict[State, float] = {}


# Bellman equation for policy evaluation
#
#   V(s) <- Σ_a π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
#
# Value is improved by following a fixed policy
#
def policy_evaluation():
    for s in states:
        # v
        v = 0.0

        # Σ_a
        for a in actions:
            # π(a | s)
            p = policy(s, a)

            # Σ_s'
            for s_next in states:
                # v += π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
                v += (
                    p
                    * world_model(s, a, s_next)
                    * (reward(s, a, s_next) + (GAMMA * value[s_next]))
                )

        # V(s) <- v
        value[s] = v


def train():
    converged = False

    while not converged:
        policy_evaluation()
