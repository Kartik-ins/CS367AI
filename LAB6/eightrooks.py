import numpy as np

# Config 
N = 8
M = N * N
MAX_ITER = 1000
W = -2          # inhibitory weight
B = 1           # bias

def idx(r, c):
    return r * N + c

# Weight matrix 
T = np.zeros((M, M))
for r1 in range(N):
    for c1 in range(N):
        k = idx(r1, c1)
        for r2 in range(N):
            for c2 in range(N):
                l = idx(r2, c2)
                if k == l:
                    continue
                if r1 == r2 or c1 == c2:   # same row/col
                    T[k, l] = W

# Bias vector 
I = np.full(M, B)

# Hopfield update 
def solve(T, I, iters):
    V = np.random.randint(0, 2, M)
    for _ in range(iters):
        k = np.random.randint(M)
        net = np.dot(T[k], V) + I[k]
        V[k] = 1 if net > 0 else 0
    return V.reshape(N, N)

# Run 
board = solve(T, I, MAX_ITER)
print(board)
print("Total rooks:", board.sum())

# Check validity 
valid = True
for r in range(N):
    if board[r].sum() != 1:
        print(f"Row {r} invalid")
        valid = False
for c in range(N):
    if board[:, c].sum() != 1:
        print(f"Column {c} invalid")
        valid = False

print("Is Valid :" , valid and board.sum() == N)
