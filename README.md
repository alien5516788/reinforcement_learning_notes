> [!Notice]
> **Work in Progress**
>
> The current algorithm implementations are incomplete and contain placeholders. They are intended to demonstrate core concepts and are still under development. Some implementations may not be functional or fully accurate.

# Reinforcement Learning Notes

A hands-on journey into reinforcement learning, implemented in Python and Rust.
This repository documents my process of learning reinforcement learning from the fundamentals.
The goal is to understand how these algorithms work, why they work, and how they differ from one another.

## Topics Covered

### Reinforcement Learning Fundamentals

- Markov Decision Processes (MDPs)
- States, actions, transition probabilities, and rewards
- Discount factor
- Policies and value functions
- State-value functions and action-value functions
- Bellman equations

### 1. Dynamic Programming

- Policy evaluation
- Policy iteration
- Value iteration

### 2. Monte Carlo Learning

- MC prediction
- MC control

### 3. Temporal Difference Learning

- TD prediction
  - TD(0)
  - TD(λ)
- TD control
  - SARSA
  - Expected-SARSA
  - Q-learning

### 4. Policy Gradient Methods

- REINFORCE
- PPO
- Actor-Critic

## Extra Resources

- OpenAI Spinningup: https://spinningup.openai.com/en/latest/
- Python Gymnasium: https://gymnasium.farama.org/
