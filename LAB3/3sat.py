import random
import itertools
from math import inf

def generate_3sat(n, m, seed=None):
    if seed is not None:
        random.seed(seed)
    clauses = set()
    while len(clauses) < m:
        vars_sample = random.sample(range(1, n + 1), 3)
        clause = tuple(sorted(
            [v if random.random() < 0.5 else -v for v in vars_sample],
            key=lambda x: (abs(x), x)
        ))
        clauses.add(clause)
    return [list(c) for c in clauses]

class Evaluator:
    def __init__(self, clauses, heuristic_type="satisfied"):
        self.clauses = clauses
        self.m = len(clauses)
        self.count = 0
        self.heuristic_type = heuristic_type

        if heuristic_type == "weighted":
            self.var_weights = {}
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if var not in self.var_weights:
                        self.var_weights[var] = 0
                    self.var_weights[var] += 1

    def eval(self, assignment):
        self.count += 1

        if self.heuristic_type == "satisfied":
            satisfied = 0
            for clause in self.clauses:
                for lit in clause:
                    val = assignment[abs(lit)]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied += 1
                        break
            return satisfied

        elif self.heuristic_type == "weighted":
            score = 0
            for clause in self.clauses:
                clause_satisfied = False
                for lit in clause:
                    val = assignment[abs(lit)]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        clause_satisfied = True
                        score += self.var_weights[abs(lit)]
                        break
                if clause_satisfied:
                    score += 1
            return score

def hill_climbing(clauses, n, max_steps=1000, maximize=True, heuristic="satisfied"):
    evaluator = Evaluator(clauses, heuristic_type=heuristic)
    current = {i: random.choice([True, False]) for i in range(1, n + 1)}
    score = evaluator.eval(current)
    if not maximize:
        score = evaluator.m - score if heuristic == "satisfied" else -score

    moves = 0
    for _ in range(max_steps):
        best_state, best_score = None, (-inf if maximize else inf)

        for var in range(1, n + 1):
            neighbor = current.copy()
            neighbor[var] = not neighbor[var]
            neighbor_score = evaluator.eval(neighbor)
            if not maximize:
                neighbor_score = evaluator.m - neighbor_score if heuristic == "satisfied" else -neighbor_score

            if (maximize and neighbor_score > best_score) or \
               (not maximize and neighbor_score < best_score):
                best_score = neighbor_score
                best_state = neighbor

        if (maximize and best_score <= score) or \
           (not maximize and best_score >= score):
            break

        current, score = best_state, best_score
        moves += 1

        satisfied_clauses = evaluator.eval(current) if heuristic == "satisfied" else count_satisfied(evaluator.clauses, current)
        if satisfied_clauses == evaluator.m:
            return current, score, moves, evaluator.count, True

    satisfied_clauses = evaluator.eval(current) if heuristic == "satisfied" else count_satisfied(evaluator.clauses, current)
    success = (satisfied_clauses / evaluator.m >= 0.98)
    return current, score, moves, evaluator.count, success

def beam_search(clauses, n, width=3, max_steps=500, maximize=True, heuristic="satisfied"):
    evaluator = Evaluator(clauses, heuristic_type=heuristic)
    beam = [{i: random.choice([True, False]) for i in range(1, n + 1)} for _ in range(width)]
    scores = [evaluator.eval(s) for s in beam]
    if not maximize:
        scores = [evaluator.m - s if heuristic == "satisfied" else -s for s in scores]

    moves = 0
    for _ in range(max_steps):
        for i, state in enumerate(beam):
            satisfied_clauses = evaluator.eval(state) if heuristic == "satisfied" else count_satisfied(evaluator.clauses, state)
            if satisfied_clauses == evaluator.m:
                return state, scores[i], moves, evaluator.count, True

        successors = []
        for state in beam:
            for var in range(1, n + 1):
                neighbor = state.copy()
                neighbor[var] = not neighbor[var]
                neighbor_score = evaluator.eval(neighbor)
                if not maximize:
                    neighbor_score = evaluator.m - neighbor_score if heuristic == "satisfied" else -neighbor_score
                successors.append((neighbor_score, neighbor))

        if not successors:
            break

        successors.sort(key=lambda x: x[0], reverse=maximize)
        beam = [s for _, s in successors[:width]]
        scores = [sc for sc, _ in successors[:width]]
        moves += 1

    best_idx = (max if maximize else min)(range(len(scores)), key=lambda i: scores[i])
    best_state, best_score = beam[best_idx], scores[best_idx]

    satisfied_clauses = evaluator.eval(best_state) if heuristic == "satisfied" else count_satisfied(evaluator.clauses, best_state)
    success = (satisfied_clauses / evaluator.m >= 0.97)
    return best_state, best_score, moves, evaluator.count, success


def vnd(clauses, n, max_loops=100, k_max=3, maximize=True, heuristic="satisfied"):
    evaluator = Evaluator(clauses, heuristic_type=heuristic)
    current = {i: random.choice([True, False]) for i in range(1, n + 1)}
    score = evaluator.eval(current)
    if not maximize:
        score = evaluator.m - score if heuristic == "satisfied" else -score

    moves = 0
    for _ in range(max_loops):
        k = 1
        while k <= k_max:
            satisfied_clauses = evaluator.eval(current) if heuristic == "satisfied" else count_satisfied(evaluator.clauses, current)
            if satisfied_clauses == evaluator.m:
                return current, score, moves, evaluator.count, True

            best_state, best_score = None, score

            if k == 1:
                for var in range(1, n + 1):
                    neighbor = current.copy()
                    neighbor[var] = not neighbor[var]
                    neighbor_score = evaluator.eval(neighbor)
                    if not maximize:
                        neighbor_score = evaluator.m - neighbor_score if heuristic == "satisfied" else -neighbor_score

                    if (maximize and neighbor_score > best_score) or \
                       (not maximize and neighbor_score < best_score):
                        best_state = neighbor
                        best_score = neighbor_score

            elif k == 2:
                clause = random.choice(clauses)
                vars_in_clause = list({abs(lit) for lit in clause})

                neighbor = current.copy()
                for v in vars_in_clause:
                    neighbor[v] = not neighbor[v]
                neighbor_score = evaluator.eval(neighbor)
                if not maximize:
                    neighbor_score = evaluator.m - neighbor_score if heuristic == "satisfied" else -neighbor_score

                if (maximize and neighbor_score > best_score) or \
                   (not maximize and neighbor_score < best_score):
                    best_state = neighbor
                    best_score = neighbor_score

            elif k == 3:
                selected_clauses = random.sample(clauses, min(k, len(clauses)))
                vars_to_flip = list({abs(lit) for clause in selected_clauses for lit in clause})

                neighbor = current.copy()
                for v in vars_to_flip:
                    neighbor[v] = not neighbor[v]
                neighbor_score = evaluator.eval(neighbor)
                if not maximize:
                    neighbor_score = evaluator.m - neighbor_score if heuristic == "satisfied" else -neighbor_score

                if (maximize and neighbor_score > best_score) or \
                   (not maximize and neighbor_score < best_score):
                    best_state = neighbor
                    best_score = neighbor_score

            if best_state and ((maximize and best_score > score) or \
                              (not maximize and best_score < score)):
                current, score = best_state, best_score
                moves += 1
                k = 1
            else:
                k += 1

    satisfied_clauses = evaluator.eval(current) if heuristic == "satisfied" else count_satisfied(evaluator.clauses, current)
    success = (satisfied_clauses / evaluator.m >= 0.97)
    return current, score, moves, evaluator.count, success

def count_satisfied(clauses, assignment):
    satisfied = 0
    for clause in clauses:
        for lit in clause:
            val = assignment[abs(lit)]
            if (lit > 0 and val) or (lit < 0 and not val):
                satisfied += 1
                break
    return satisfied

def simulate(n=20, ratio=3.0, runs=100, seed=None):
    if seed is not None:
        random.seed(seed)
    m = int(round(ratio * n))

    algorithms = [
        ("HC-Standard", lambda c, n: hill_climbing(c, n, heuristic="satisfied")),
        ("HC-Weighted", lambda c, n: hill_climbing(c, n, heuristic="weighted")),
        ("Beam3-Standard", lambda c, n: beam_search(c, n, width=3, heuristic="satisfied")),
        ("Beam3-Weighted", lambda c, n: beam_search(c, n, width=3, heuristic="weighted")),
        ("Beam4-Standard", lambda c, n: beam_search(c, n, width=4, heuristic="satisfied")),
        ("Beam4-Weighted", lambda c, n: beam_search(c, n, width=4, heuristic="weighted")),
        ("VND-Standard", lambda c, n: vnd(c, n, heuristic="satisfied")),
        ("VND-Weighted", lambda c, n: vnd(c, n, heuristic="weighted"))
    ]

    stats = {name: {"wins": 0, "pen_succ": [], "pen_all": []}
             for name, _ in algorithms}

    for run in range(runs):
        clauses = generate_3sat(n, m)
        for name, algo in algorithms:
            _, _, moves, evals, success = algo(clauses, n)
            penetrance = (moves / evals) if evals > 0 else 0.0
            stats[name]["pen_all"].append(penetrance)
            if success:
                stats[name]["wins"] += 1
                stats[name]["pen_succ"].append(penetrance)

    print(f"{'Algorithm':20} | {'Success Rate':12} | {'Pen(succ)':10} | {'Pen(all)':10}")
    print("-" * 60)
    for name in stats:
        wins = stats[name]["wins"]
        win_rate = 100.0 * wins / runs
        avg_succ = sum(stats[name]["pen_succ"]) / len(stats[name]["pen_succ"]) \
                   if stats[name]["pen_succ"] else 0.0
        avg_all = sum(stats[name]["pen_all"]) / len(stats[name]["pen_all"])
        print(f"{name:20} | {win_rate:10.1f}% | {avg_succ:.6f} | {avg_all:.6f}")


runs, n = 50, 25  

print("Easy Problems (m/n = 3.0)")
simulate(n=n, ratio=3.0, runs=runs, seed=0)
print("\nHard Problems (m/n = 4.26)")
simulate(n=n, ratio=4.26, runs=runs, seed=1)
