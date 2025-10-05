sentences1 = []
with open('input1.txt', 'r') as f:
    for line in f:
        sentences1.append(line.strip())

sentences2 = []
with open('input2.txt', 'r') as f:
    for line in f:
        sentences2.append(line.strip())

words1 = []
letters1 = []
for sentence in sentences1:
    temp = sentence.split(' ')
    words1.append(temp)
    x = 0
    for word in temp:
        x += len(word)
    letters1.append(x)

words2 = []
letters2 = []
for sentence in sentences2:
    temp = sentence.split()
    words2.append(temp)
    x = 0
    for word in temp:
        x += len(word)
    letters2.append(x)

def edit_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],
                                   dp[i][j - 1],
                                   dp[i - 1][j - 1])
    return dp[m][n]

def sentence_alignment_cost(sent1, sent2):
    words1 = sent1.split(' ')
    words2 = sent2.split(' ')
    n = len(words1)
    m = len(words2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                dp[i][j] = 0
            elif i == 0:
                dp[i][j] = dp[i][j - 1] + len(words1[j - 1])
            elif j == 0:
                dp[i][j] = dp[i - 1][j] + len(words2[i - 1])
            else:
                cost_substitute = dp[i - 1][j - 1] + edit_distance(words2[i - 1], words1[j - 1])
                cost_delete_w2 = dp[i - 1][j] + len(words2[i - 1])
                cost_delete_w1 = dp[i][j - 1] + len(words1[j - 1])
                dp[i][j] = min(cost_substitute, cost_delete_w2, cost_delete_w1)
    return dp[m][n]

heuristic_grid = [[0 for _ in range(len(sentences2) + 1)] for _ in range(len(sentences1) + 1)]

for i in range(len(sentences1) - 1, -1, -1):
    for j in range(len(sentences2) - 1, -1, -1):
        align_cost = sentence_alignment_cost(sentences1[i], sentences2[j])
        cost1 = heuristic_grid[i + 1][j + 1] + align_cost
        cost2 = heuristic_grid[i + 1][j] + letters1[i]
        cost3 = heuristic_grid[i][j + 1] + letters2[j]
        heuristic_grid[i][j] = min(cost1, cost2, cost3)

def a_star_search():
    state = [0, 0, 0]

    while state[0] < len(sentences1) and state[1] < len(sentences2):
        i, j = state[0], state[1]
        current_cost = state[2]
        
        moves = []
        
        g_align = current_cost + sentence_alignment_cost(sentences1[i], sentences2[j])
        h_align = heuristic_grid[i + 1][j + 1]
        f_align = g_align + h_align
        moves.append({'f': f_align, 'g': g_align, 'next_state': [i + 1, j + 1, g_align]})
        
        g_skip1 = current_cost + letters1[i]
        h_skip1 = heuristic_grid[i + 1][j]
        f_skip1 = g_skip1 + h_skip1
        moves.append({'f': f_skip1, 'g': g_skip1, 'next_state': [i + 1, j, g_skip1]})

        g_skip2 = current_cost + letters2[j]
        h_skip2 = heuristic_grid[i][j + 1]
        f_skip2 = g_skip2 + h_skip2
        moves.append({'f': f_skip2, 'g': g_skip2, 'next_state': [i, j + 1, g_skip2]})
        
        best_move = min(moves, key=lambda x: x['f'])
        state = best_move['next_state']
        
    final_cost = state[2]
    if state[0] < len(sentences1):
        final_cost += sum(letters1[state[0]:])
    if state[1] < len(sentences2):
        final_cost += sum(letters2[state[1]:])

    return final_cost

total_similarity_cost = a_star_search()

print(f"Total alignment cost: {total_similarity_cost}")

total_chars1 = sum(letters1)
total_chars2 = sum(letters2)
max_chars = max(total_chars1, total_chars2)

if max_chars == 0:
    ratio = 0
else:
    ratio = total_similarity_cost / max_chars

print(f"Similarity Ratio: {ratio:.4f}")

if ratio < 0.3:
    print("Result: Plagiarism detected")
else:
    print("Result: No plagiarism detected")