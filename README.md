# 2048-Reinforcement-Learning
SOC PROJECT REPO

Week 1: Foundations of Reinforcement Learning

This week focused on understanding the theoretical basis of reinforcement learning (RL):

1. Markov Decision Processes (MDPs)

Defined an environment as a tuple  where:

: set of states

: set of actions

: transition probability

: reward function

: discount factor

2. Q-Learning
Implemented the core Q-learning algorithm:

import numpy as np

q_table = np.zeros((state_space_size, action_space_size))

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state]) if np.random.rand() > epsilon else np.random.randint(action_space_size)
        next_state, reward, done, _ = env.step(action)
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state

3. Temporal Difference (TD) Learning

Combined ideas from Monte Carlo and dynamic programming.

Applied TD(0) update rule:

V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])

4. Bellman Equation

Used to iteratively update value functions:

V(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]

Week 2: Deep Q-Learning

1. Exploration vs. Exploitation

Used epsilon-greedy strategy:

if np.random.rand() < epsilon:
    action = np.random.randint(action_space_size)
else:
    action = np.argmax(q_network.predict(state))

2. Deep Q-Networks (DQN)

Replaced Q-table with a neural network:

model = Sequential([
    Dense(24, input_dim=state_size, activation='relu'),
    Dense(24, activation='relu'),
    Dense(action_size, activation='linear')
])

3. Replay Buffer

from collections import deque
replay_buffer = deque(maxlen=2000)
replay_buffer.append((state, action, reward, next_state, done))

4. Training DQN

minibatch = random.sample(replay_buffer, batch_size)
for state, action, reward, next_state, done in minibatch:
    target = reward
    if not done:
        target += gamma * np.amax(model.predict(next_state))
    target_f = model.predict(state)
    target_f[0][action] = target
    model.fit(state, target_f, epochs=1, verbose=0)

Week 3: Understanding the Game of 2048

1. Rules and Mechanics

Game played on 4x4 grid.

Merge tiles with the same number by sliding.

2. Environment Modeling

class Game2048:
    def __init__(self):
        self.board = np.zeros((4,4), dtype=int)
        self.reset()

  **  def reset(self):
        self.board[...]  # Logic to initialize
        return self.board

**    def step(self, action):
        # Implement slide and merge logic
        return next_state, reward, done****

3. Existing Approaches

Expectimax search

Greedy tile-merging policies

RL-based agents (e.g., N-Tuple networks)

Week 4: N-Tuple Networks

1. Concept

An N-Tuple is a fixed selection of board positions.

Each tuple acts as a feature.

Tuple values are used as indices into a large weight table.

2. Related Papers

Rodgers & Levine (2014): Applied to 2048 with promising results.

Szubert & Jaśkowski (2014): Showed superiority of N-Tuple over TD learning in board games.

3. Basic Implementation

def get_tuple_features(board, tuples):
    features = []
    for t in tuples:
        value = tuple(board[i][j] for i, j in t)
        features.append(value)
    return features

4. Learning Weights

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        features = get_tuple_features(state, tuples)
        action = select_action(features, weights)
        next_state, reward, done = env.step(action)
        update_weights(weights, features, reward, alpha)
        state = next_state

Week 5: Policy Gradient Methods

1. Policy Gradient Basics

Directly parameterizes the policy  and optimizes using gradient ascent.

2. REINFORCE Algorithm

for episode in range(num_episodes):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        probs = policy_network.predict(state)
        action = np.random.choice(len(probs), p=probs)
        next_state, reward, done = env.step(action)

**        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
**
    returns = compute_discounted_rewards(rewards)
    update_policy(states, actions, returns)**
**
3. Application Challenges

Sparse rewards

Delayed feedback in 2048

Larger state space requires function approximation

Conclusion and Next Steps

Over the five weeks, I gained a comprehensive understanding of reinforcement learning, from fundamental concepts to advanced implementations. Through practical work on the 2048 game, I explored both classical and deep learning approaches, N-Tuple networks, and policy gradient techniques.

Next Steps:

Integrate actor-critic models

Use convolutional networks for better board representation

Explore curriculum learning for progressive difficulty

Code can be modularized and extended to train agents using PyTorch or TensorFlow for advanced experiments.


