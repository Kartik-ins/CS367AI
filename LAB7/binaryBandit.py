import numpy as np
import matplotlib.pyplot as plt

def binaryBanditA(action):
    p = [0.1, 0.2]
    return 1 if np.random.rand() < p[action] else 0

def binaryBanditB(action):
    p = [0.8, 0.9]
    return 1 if np.random.rand() < p[action] else 0

class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.q_values = np.zeros(2)
        self.action_counts = np.zeros(2)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (1 / self.action_counts[action]) * (reward - self.q_values[action])

def run_binary_simulation(bandit_func, steps=1000):
    agent = EpsilonGreedyAgent(epsilon=0.1)
    rewards = []
    
    q_history = np.zeros((steps, 2))
    
    for t in range(steps):
        action = agent.select_action()
        reward = bandit_func(action)
        agent.update(action, reward)
        rewards.append(reward)
        
        q_history[t] = agent.q_values.copy()
        
    return agent.q_values, rewards, q_history

steps = 1000

final_q_A, rewards_A, history_A = run_binary_simulation(binaryBanditA, steps)
print(f"Bandit A Estimates: {final_q_A} (True: [0.1, 0.2])")

final_q_B, rewards_B, history_B = run_binary_simulation(binaryBanditB, steps)
print(f"Bandit B Estimates: {final_q_B} (True: [0.8, 0.9])")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_A[:, 0], label='Est Q(Action 0) [True=0.1]', color='red', alpha=0.6)
plt.plot(history_A[:, 1], label='Est Q(Action 1) [True=0.2]', color='blue', alpha=0.6)
plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.3)
plt.axhline(y=0.2, color='blue', linestyle='--', alpha=0.3)
plt.title('Bandit A Learning Curve')
plt.xlabel('Steps')
plt.ylabel('Estimated Q-Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_B[:, 0], label='Est Q(Action 0) [True=0.8]', color='orange', alpha=0.6)
plt.plot(history_B[:, 1], label='Est Q(Action 1) [True=0.9]', color='green', alpha=0.6)
plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.3)
plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.3)
plt.title('Bandit B Learning Curve')
plt.xlabel('Steps')
plt.ylabel('Estimated Q-Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()