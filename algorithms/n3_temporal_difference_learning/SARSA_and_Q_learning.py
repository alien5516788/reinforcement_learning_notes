# The general TD algorithm doesn't define how the next value is calculated
# It depends on the specific TD implementation
#
# SARSA (On-Policy, Model-Free)
#   - S -> A -> R -> S -> A
#   - SARSA calculate the next value based on the action selected using policy (On-Policy)
#
#       Q_π(s, a) <- Q_π(s, a) + α[R + γQ_π(s', a') - Q_π(s, a)]
#
# Expected-SARSA (On-Policy, Model-Free)
#   - Special type of SARSA
#   - It sums over all possible next actions multiplied by policy
#
#       Q_π(s, a) <- Q_π(s, a) + α[R + γ * Σ_a' π(a' | s') Q_π(s', a') - Q_π(s, a)]
#
# Q-Learning (Off-Policy, Model-Free)
#   - Q-Learning calculates the next value using highest Q-value among all available actions for the next state
#
#       Q(s, a) <- Q(s, a) + α[R + γ max_a' Q(s', a') - Q(s, a)]
#
