from math import inf

from . import GAMMA, Action, State, print_v_table, reward, world_model

# Policy (π) - learnt
#
#   π(a | s)
#
policy_probability: dict[tuple[State, Action], float] = {}

for s in State.all():
    for a in Action.all():
        policy_probability[(s, a)] = 1.0 / len(Action.all())


# Value (V) - learnt
#
#  V(s) -> v
#
value: dict[State, float] = {}

for s in State.all():
    value[s] = 0


# Policy Evaluation
# Value is improved by following a current policy
def policy_evaluation():
    value_new = {}

    for s in State.all():
        # v
        v = 0.0

        # Σ_a
        for a in Action.all():
            # π(a | s)
            p = policy_probability[(s, a)]

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


# Bellman equation for policy improvement
#
#     Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
#     a* = argmax_a Q(s, a)
#     π(a | s) <- 1 if a == a*; 0 otherwise                    // greedy/deterministic
#     π(a | s) <- 0.9 + 1/n if a == a*; 0.1 * 1/n otherwise      // stochastic
#
# Policy is improved by calculating the Q(s,a) for every possible action a and
#   updating the policy table with the action a* (action a with best Q value)
#
def policy_improvement():
    for s in State.all():
        # Place holders until calculated
        best_action = Action.all()[0]  # a*
        best_action_value = -inf  # max_a Q(s, a)

        for a in Action.all():
            # q
            q = 0.0

            # Σ_s'
            for s_next in State.all():
                # q += P(s' | s, a) * [R(s, a, s') + γV(s')]
                q += world_model(s, a, s_next) * (
                    reward(s, a, s_next) + (GAMMA * value[s_next])
                )

            # Q(s, a) = q

            #  a* <- arg_max_a Q(s, a)
            if q > best_action_value:
                best_action = a  # a*
                best_action_value = q  # max_a Q(s, a)

        # π(a | s) <- 1 if a == a*, 0 otherwise
        for a in Action.all():
            if a == best_action:
                policy_probability[(s, a)] = 1.0
            else:
                policy_probability[(s, a)] = 0.0


def policy_iteration():
    # Policy Evaluation
    MAX_ITERATIONS = 3

    for i in range(MAX_ITERATIONS):
        policy_evaluation()

    # Policy Improvement
    policy_improvement()


def train():
    MAX_ITERATIONS = 1000
    DOCK = (0, 3)

    print("Initial value")
    print_v_table(value, DOCK)

    for i in range(MAX_ITERATIONS):
        policy_iteration()

        print("Iteration: ", i + 1)
        print_v_table(value, DOCK)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(e)
