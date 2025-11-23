import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# --- Configuration Parameters ---
MAX_BIKES = 20
MAX_MOVE = 5
DISCOUNT_GAMMA = 0.9
RENT_REWARD = 10
MOVE_COST = 2
PARKING_PENALTY = 4
PARKING_LIMIT = 10

# Poisson Expectations
LAMBDA_RENT_A = 3
LAMBDA_RENT_B = 4
LAMBDA_RET_A = 3
LAMBDA_RET_B = 2

# Pre-calculate Poisson probabilities to save computation time
# Cache up to MAX_BIKES for efficiency
poisson_rent_A = [poisson.pmf(i, LAMBDA_RENT_A) for i in range(MAX_BIKES + 1)]
poisson_rent_B = [poisson.pmf(i, LAMBDA_RENT_B) for i in range(MAX_BIKES + 1)]
poisson_ret_A = [poisson.pmf(i, LAMBDA_RET_A) for i in range(MAX_BIKES + 1)]
poisson_ret_B = [poisson.pmf(i, LAMBDA_RET_B) for i in range(MAX_BIKES + 1)]

def get_transition_prob(state_a, state_b, action):
    """
    Calculates the expected return based on the current state and action.
    Returns the total expected reward and a dictionary of next state probabilities.
    """
    # Apply the move (Action is net moves from A to B)
    # If action is positive: Move A -> B
    # If action is negative: Move B -> A
    
    # Morning inventory after overnight moves
    morning_bikes_a = int(min(MAX_BIKES, max(0, state_a - action)))
    morning_bikes_b = int(min(MAX_BIKES, max(0, state_b + action)))
    
    # Calculate Cost immediately
    # Modified Rule: First bike A->B is free
    move_cost = 0
    if action > 0: # Moving A to B
        # First one free, rest cost 2
        billable_moves = max(0, action - 1)
        move_cost = billable_moves * MOVE_COST
    else: # Moving B to A (action is negative)
        move_cost = abs(action) * MOVE_COST
        
    expected_reward = -move_cost

    # Add Parking Penalty (Modified Rule)
    if state_a > PARKING_LIMIT:
        expected_reward -= PARKING_PENALTY
    if state_b > PARKING_LIMIT:
        expected_reward -= PARKING_PENALTY

    transitions = {}
    
    # Iterate through all possible rental/return scenarios
    # Note: We iterate strictly to maintain performance, assuming negligible prob beyond cutoff
    for rent_a, p_rent_a in enumerate(poisson_rent_A):
        for rent_b, p_rent_b in enumerate(poisson_rent_B):
            
            # Actual rentals cannot exceed available bikes
            actual_rent_a = min(morning_bikes_a, rent_a)
            actual_rent_b = min(morning_bikes_b, rent_b)
            
            # Immediate reward from rentals
            prob_rent = p_rent_a * p_rent_b
            current_rent_reward = (actual_rent_a + actual_rent_b) * RENT_REWARD
            
            bikes_on_lot_a = morning_bikes_a - actual_rent_a
            bikes_on_lot_b = morning_bikes_b - actual_rent_b
            
            for ret_a, p_ret_a in enumerate(poisson_ret_A):
                for ret_b, p_ret_b in enumerate(poisson_ret_B):
                    
                    prob_scenario = prob_rent * p_ret_a * p_ret_b
                    
                    # Calculate final state at end of day
                    final_a = min(MAX_BIKES, bikes_on_lot_a + ret_a)
                    final_b = min(MAX_BIKES, bikes_on_lot_b + ret_b)
                    
                    next_s = (final_a, final_b)
                    
                    # Accumulate probability for this next state
                    transitions[next_s] = transitions.get(next_s, 0) + prob_scenario
                    
                    # Add weighted reward for this specific scenario
                    expected_reward += prob_scenario * current_rent_reward

    return expected_reward, transitions

def policy_iteration():
    # Initialize Value Function and Policy
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)
    
    states = [(i, j) for i in range(MAX_BIKES + 1) for j in range(MAX_BIKES + 1)]
    
    is_stable = False
    iteration = 0
    
    while not is_stable:
        print(f"Iteration {iteration} in progress...")
        
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for i, j in states:
                old_v = V[i, j]
                action = policy[i, j]
                
                r_immediate, next_states = get_transition_prob(i, j, action)
                
                new_v = r_immediate
                for (ni, nj), prob in next_states.items():
                    new_v += DISCOCOUNT_GAMMA * prob * V[ni, nj]
                
                V[i, j] = new_v
                delta = max(delta, abs(old_v - V[i, j]))
            
            if delta < 1e-2: # Threshold for value convergence
                break
        
        # --- Policy Improvement ---
        policy_stable = True
        for i, j in states:
            old_action = policy[i, j]
            action_values = []
            
            # Valid actions: limited by MAX_MOVE and cannot result in negative bikes
            min_action = max(-MAX_MOVE, -j) # Can't move more from B than exist
            max_action = min(MAX_MOVE, i)   # Can't move more from A than exist
            
            possible_actions = range(min_action, max_action + 1)
            
            best_act_val = float('-inf')
            best_act = 0
            
            for action in possible_actions:
                r_immediate, next_states = get_transition_prob(i, j, action)
                val = r_immediate
                for (ni, nj), prob in next_states.items():
                    val += DISCOCOUNT_GAMMA * prob * V[ni, nj]
                
                if val > best_act_val:
                    best_act_val = val
                    best_act = action
            
            policy[i, j] = best_act
            if old_action != best_act:
                policy_stable = False
        
        if policy_stable:
            is_stable = True
            print(f"Converged at iteration {iteration}")
        iteration += 1

    return policy, V

def plot_results(policy_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(policy_matrix, cmap='coolwarm', annot=True, fmt='d', 
                cbar_kws={'label': 'Bikes Moved (A -> B)'})
    plt.gca().invert_yaxis() # Standard graph orientation (0,0 at bottom left)
    plt.title("Optimal Policy: Modified Gbike Problem")
    plt.xlabel("Bikes at Location B")
    plt.ylabel("Bikes at Location A")
    plt.savefig("policy_heatmap.png")
    print("Heatmap saved as policy_heatmap.png")

if __name__ == "__main__":
    optimal_policy, optimal_values = policy_iteration()
    plot_results(optimal_policy)
