import numpy as np
import random
import time

# Config
n = 10                   # Number of cities
gamma = 500              # Constraint penalty
iterations = 50_000      # Relaxation steps

# Generate distance matrix
np.random.seed(42)
D = np.random.randint(1, 100, (n, n))
np.fill_diagonal(D, 0)
D = (D + D.T) // 2        # Make symmetric

print(f"Hopfield Tank TSP ({n} cities)")
print("Distance matrix:\n", D)


# Network initialization
m = n * n                # Total neurons
W = np.zeros((m, m))
theta = np.full(m, -gamma)
state = np.random.randint(0, 2, m)   # Random initial state

def idx(city, step):
    return city * n + step


# Build weight matrix
start = time.time()

for c in range(n):
    for s in range(n):
        u = idx(c, s)

        # Row constraint: one city per step
        for c2 in range(n):
            if c2 != c:
                W[u, idx(c2, s)] -= gamma

        # Column constraint: one step per city
        for s2 in range(n):
            if s2 != s:
                W[u, idx(c, s2)] -= gamma

        # Distance term (forward/backward)
        next_step = (s + 1) % n
        prev_step = (s - 1) % n
        for c2 in range(n):
            if c2 != c:
                W[u, idx(c2, next_step)] -= D[c][c2]
                W[u, idx(c2, prev_step)] -= D[c][c2]

# Symmetrize
W = (W + W.T) / 2

# Asynchronous update rule
def update():
    k = random.randint(0, m - 1)
    net = np.dot(W[k], state) - theta[k]
    state[k] = 1 if net > 0 else 0

print(f"\nRunning {iterations} asynchronous updates...")
for _ in range(iterations):
    update()

print(f"Relaxation complete in {time.time() - start:.4f} seconds.")

# Decode final state → tour
tour = []
for step in range(n):
    col = [state[idx(c, step)] for c in range(n)]
    city = np.argmax(col)
    tour.append(city)

# Compute cost
cost = 0
for i in range(n):
    a = tour[i]
    b = tour[(i + 1) % n]
    cost += D[a, b]

# Summary
print("\nTSP Solution")
print("Tour:", [int(x) for x in tour])
print("Total cost:", cost)

# Constraint quality check
V = state.reshape(n, n)
row_errors = np.sum(np.abs(V.sum(axis=1) - 1))
col_errors = np.sum(np.abs(V.sum(axis=0) - 1))

print("\nConstraint violations (ideal = 0):")
print("Row violations:", int(row_errors))
print("Col violations:", int(col_errors))
