from math import inf

from . import GAMMA, Action, State, print_v_table, reward, world_model

# Value (V) - learnt
#
#  V(s) -> v
#
value: dict[State, float] = {}

for s in State.all():
    value[s] = 0


# Bellman equation for value iteration
#
#   Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
#   V(s) <- max_a Q(s, a)
#
# Value is learnt by continuously improving the value of each state
#   by choosing the action with the highest expected return.
#
def value_iteration():
    value_new = {}

    for s in State.all():
        # Place holder until calculated
        v = -inf  # max_a Q(s, a)

        for a in Action.all():
            # Q(s, a)
            q = 0.0

            # Σ_s'
            for s_next in State.all():
                # Q(s, a) =+ Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
                q += world_model(s, a, s_next) * (
                    reward(s, a, s_next) + (GAMMA * value[s_next])
                )

            # max_a Q(s, a)
            v = max(q, v)

        # V(s) <- max_a Q(s, a)
        value_new[s] = v

    value.update(value_new)


def train():
    MAX_ITERATIONS = 1000
    DOCK = (0, 3)

    print("Initial value")
    print_v_table(value, DOCK)

    for i in range(MAX_ITERATIONS):
        value_iteration()

        print("Iteration: ", i + 1)
        print_v_table(value, DOCK)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
