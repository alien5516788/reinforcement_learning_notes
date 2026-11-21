from . import GAMMA, Action, State, print_v_table, reward, world_model


# Policy (π) - fixed
#
#   π(a | s)
#
def policy_probability(s: State, a: Action) -> float:
    # Each action have an equal probability
    return 1.0 / len(Action.all())


# Value (V) - learnt
#
#  V(s) -> v
#
value: dict[State, float] = {}

for s in State.all():
    value[s] = 0


# Bellman equation for policy evaluation
#
#   V(s) <- Σ_a π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
#
# Value is improved by following a fixed policy
#
def policy_evaluation():
    value_new = {}

    for s in State.all():
        # v
        v = 0.0

        # Σ_a
        for a in Action.all():
            # π(a | s)
            p = policy_probability(s, a)

            # Σ_s'
            for s_next in State.all():
                # v += π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
                v += (
                    p
                    * world_model(s, a, s_next)
                    * (reward(s, a, s_next) + (GAMMA * value[s_next]))
                )

        # V(s) <- v
        value_new[s] = v

    value.update(value_new)


def train():
    MAX_ITERATIONS = 1000
    DOCK = (0, 3)

    print("Initial value")
    print_v_table(value, DOCK)

    for i in range(MAX_ITERATIONS):
        policy_evaluation()

        print("Iteration: ", i + 1)
        print_v_table(value, DOCK)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
