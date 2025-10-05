from collections import deque

def is_valid(state):
    m, c, _ = state
    if not (0 <= m <= 3 and 0 <= c <= 3):
        return False
    if (m < c and m > 0) or ((3 - m) < (3 - c) and (3 - m) > 0):
        return False
    return True

def get_successors(state):
    m, c, boat = state
    direction = -1 if boat == 1 else 1
    moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
    successors = []
    for dm, dc in moves:
        new_state = (m + direction * dm, c + direction * dc, 1 - boat)
        if is_valid(new_state):
            successors.append(new_state)
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

start_state = (3, 3, 1)
goal_state = (0, 0, 0)
solution = bfs(start_state, goal_state)

if solution:
    for step in solution:
        side = "Left" if step[2] == 1 else "Right"
        print(f"{step[0]}M {step[1]}C Boat:{side}")
else:
    print("No solution found.")
