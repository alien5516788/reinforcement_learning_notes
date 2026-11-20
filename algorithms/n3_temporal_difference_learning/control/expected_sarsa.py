# Expected-SARSA (On-Policy, Model-Free)
#   - Special type of SARSA
#   - It sums over all possible next actions multiplied by policy
#
#       Q_π(s, a) <- Q_π(s, a) + α[R + γ * Σ_a' π(a' | s') Q_π(s', a') - Q_π(s, a)]
#