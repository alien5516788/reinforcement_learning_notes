# Reinforcement Learning Notes

A hands-on journey into reinforcement learning, implemented in Rust.
This repository documents my process of learning reinforcement learning from the fundamentals.
The goal is to understand how these algorithms work, why they work, and how they differ from one another.

## Topics Covered

### 1. Reinforcement Learning Fundamentals

- Markov Decision Processes (MDPs)
- States, actions, transition probabilities, and rewards
- Discount factor
- Policies and value functions
- State-value functions and action-value functions
- Bellman equations

### 2. Dynamic Programming

- Policy evaluation
- Policy iteration
- Value iteration

These algorithms explore how a known environment model can be used to evaluate policies and find optimal policies.

### 3. Monte Carlo Learning

- Monte Carlo prediction
- Monte Carlo control
- Learning from complete episodes
- Estimating values from sampled returns

These algorithms explore learning from experience without requiring a known environment model.

### 4. Temporal-Difference Learning

- Temporal-difference prediction
- SARSA
- Q-learning

These algorithms explore learning through experience and updating. estimates using temporal-difference targets.

## Disclaimer

The code and explanations reflect my current learning progress of the subject.
They may contain mistakes, simplifications, incomplete implementations, or misunderstandings.

## Extra Resources

OpenAI Spinningup: https://spinningup.openai.com/en/latest/
