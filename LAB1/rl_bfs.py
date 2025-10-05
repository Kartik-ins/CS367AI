from collections import deque

def is_valid(state):
    return 0 <= state.index('.') < len(state)

def get_successors(state):
    successors = []
    i = state.index('.')
    moves = [1, 2]

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

def bfs(start, goal):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        for nxt in get_successors(state):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None

start_state = "EEE.WWW"
goal_state = "WWW.EEE"

solution = bfs(start_state, goal_state)

print("BFS Solution:")
if solution:
    for step in solution:
        print(step)
else:
    print("No solution found.")
