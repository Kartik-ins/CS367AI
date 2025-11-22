import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.mean_rewards = np.zeros(k)
        self.optimal_action = 0
    
    def step(self, action, drift):
        
        self.optimal_action = np.argmax(self.mean_rewards)
        is_optimal = (action == self.optimal_action)
        reward = np.random.normal(self.mean_rewards[action], 1.0)
        self.mean_rewards += drift
        
        return reward, is_optimal

class StandardAgent:
   
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_values = np.zeros(k)
        self.action_counts = np.zeros(k)
        
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        step_size = 1.0 / self.action_counts[action]
        self.q_values[action] += step_size * (reward - self.q_values[action])

class ModifiedAgent:
   
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha 
        self.q_values = np.zeros(k)
        
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_values)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

def run_simulation(steps=10000, runs=200):
    print(f"Running {runs} independent simulations of {steps} steps...")
    
    avg_optimal_std = np.zeros(steps)
    avg_optimal_mod = np.zeros(steps)
    
    for r in range(runs):
        k = 10
        drifts = np.random.normal(0, 0.01, (steps, k))
        bandit_std = NonStationaryBandit(k)
        agent_std = StandardAgent(k, epsilon=0.1)
        
        for t in range(steps):
            drift = drifts[t]
            act = agent_std.select_action()
            rew, is_opt = bandit_std.step(act, drift)
            agent_std.update(act, rew)
            if is_opt: avg_optimal_std[t] += 1
            
        bandit_mod = NonStationaryBandit(k) 
        agent_mod = ModifiedAgent(k, epsilon=0.1, alpha=0.1)
        
        for t in range(steps):
            drift = drifts[t]
            act = agent_mod.select_action()
            rew, is_opt = bandit_mod.step(act, drift)
            agent_mod.update(act, rew)
            if is_opt: avg_optimal_mod[t] += 1

    pct_std = (avg_optimal_std / runs) * 100
    pct_mod = (avg_optimal_mod / runs) * 100
    
    return pct_std, pct_mod

if __name__ == "__main__":
    steps = 10000
    runs = 200
    
    curve_std, curve_mod = run_simulation(steps, runs)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(curve_std, color='red', label='Standard Agent (Sample Average)', linewidth=1.5)
    plt.plot(curve_mod, color='blue', label='Modified Agent (alpha=0.1)', linewidth=1.5)
    
    plt.title(f'Tracking Performance on Non-Stationary Bandit (Average of {runs} Runs)')
    plt.xlabel('Time Steps')
    plt.ylabel('% Optimal Action Chosen')
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    print("Simulation complete. Displaying plot.")
    plt.show()