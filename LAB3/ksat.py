import random

def generate_ksat(num_vars, num_clauses, k, seed=None):
    if seed is not None:
        random.seed(seed)
    clauses = set()
    while len(clauses) < num_clauses:
        vars_sample = random.sample(range(1, num_vars + 1), k)
        clause = tuple(sorted(
            [v if random.random() < 0.5 else -v for v in vars_sample],
            key=lambda x: (abs(x), x)
        ))
        clauses.add(clause)
    return [list(c) for c in clauses]

n = int(input("Enter number of variables: "))
m = int(input("Enter number of clauses: "))
k = int(input("Enter variables per clause (k): "))
seed = input("Enter random seed (press Enter to skip): ")
seed = int(seed) if seed.strip() else None

clauses = generate_ksat(n, m, k, seed)

for i, c in enumerate(clauses, 1):
    print(f"Clause {i}: {c}")
