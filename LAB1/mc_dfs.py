from collections import deque

def is_valid(state):
    m, c, _ = state
    if not (0 <= m <= 3 and 0 <= c <= 3):
        return False
    if (m > 0 and m < c) or ((3 - m) > 0 and (3 - m) < (3 - c)):
        return False
    return True

def get_successors(state):
    m, c, boat = state
    moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
    successors = []
    if boat == 1:
        for dm, dc in moves:
            new_state = (m - dm, c - dc, 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:
        for dm, dc in moves:
            new_state = (m + dm, c + dc, 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

def dfs(start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        state, path = stack.pop()
        if state == goal:
            return path
        if state in visited:
            continue
        visited.add(state)
        for nxt in get_successors(state):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))
    return None

start_state = (3, 3, 1)
goal_state = (0, 0, 0)
solution = dfs(start_state, goal_state)

if solution:
    print("Solution found:")
    for step in solution:
        side = "Left" if step[2] == 1 else "Right"
        print(f"{step[0]}M {step[1]}C Boat:{side}")
else:
    print("No solution found.")
