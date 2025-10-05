def is_valid(state):
    return '.' in state

def get_successors(state):
    successors = []
    i = state.index('.')
    moves = [1, 2]

    # Move 'W' left → '.' right
    for m in moves:
        if i + m < len(state) and state[i + m] == 'W':
            s = list(state)
            s[i], s[i + m] = s[i + m], s[i]
            successors.append(''.join(s))

    for m in moves:
        if i - m >= 0 and state[i - m] == 'E':
            s = list(state)
            s[i], s[i - m] = s[i - m], s[i]
            successors.append(''.join(s))

    return successors

def dfs_recursive(state, goal, path, visited):
    if state in visited:
        return None
    visited.add(state)
    path.append(state)

    if state == goal:
        return path

    for nxt in get_successors(state):
        res = dfs_recursive(nxt, goal, path, visited)
        if res:
            return res

    path.pop()  # Backtrack
    return None

def dfs(start, goal):
    return dfs_recursive(start, goal, [], set())

start_state = "EEE.WWW"
goal_state = "WWW.EEE"

solution = dfs(start_state, goal_state)

print("DFS Solution:")
if solution:
    for step in solution:
        print(step)
else:
    print("No solution found.")
