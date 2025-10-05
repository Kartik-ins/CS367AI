import string
import heapq
import sys

def preprocess_file(filename):
    with open(filename, 'r') as f:
        translator = str.maketrans('', '', string.punctuation)
        sentences = [line.strip().lower().translate(translator) for line in f]
    return sentences

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                cost = 1
                dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def find_optimal_alignment(sentences1, sentences2):
    def heuristic(i, j):
        h_cost = 0
        if i < len(sentences1):
            h_cost += sum(len(s) for s in sentences1[i:])
        if j < len(sentences2):
            h_cost += sum(len(s) for s in sentences2[j:])
        return h_cost

    open_list = [(heuristic(0, 0), 0, [], 0, 0)]
    closed_set = set()

    while open_list:
        f_cost, g_cost, path, i, j = heapq.heappop(open_list)

        if (i, j) in closed_set:
            continue
        closed_set.add((i, j))

        if i == len(sentences1) and j == len(sentences2):
            return path

        if i < len(sentences1) and j < len(sentences2):
            align_cost = levenshtein_distance(sentences1[i], sentences2[j])
            new_g = g_cost + align_cost
            new_path = path + [('ALIGN', i, j)]
            h = heuristic(i + 1, j + 1)
            heapq.heappush(open_list, (new_g + h, new_g, new_path, i + 1, j + 1))

        if i < len(sentences1):
            skip_cost = len(sentences1[i])
            new_g = g_cost + skip_cost
            new_path = path + [('SKIP_S1', i)]
            h = heuristic(i + 1, j)
            heapq.heappush(open_list, (new_g + h, new_g, new_path, i + 1, j))

        if j < len(sentences2):
            skip_cost = len(sentences2[j])
            new_g = g_cost + skip_cost
            new_path = path + [('SKIP_S2', j)]
            h = heuristic(i, j + 1)
            heapq.heappush(open_list, (new_g + h, new_g, new_path, i, j + 1))
            
    return []

doc1_sentences = preprocess_file('input1.txt')
doc2_sentences = preprocess_file('input2.txt')

optimal_path = find_optimal_alignment(doc1_sentences, doc2_sentences)

normalized_distance_threshold = 0.30
plagiarized_pairs = 0

print("--- Full Alignment Report ---")
for move in optimal_path:
    if move[0] == 'ALIGN':
        idx1, idx2 = move[1], move[2]
        s1 = doc1_sentences[idx1]
        s2 = doc2_sentences[idx2]
        
        distance = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            normalized_distance = 0.0
        else:
            normalized_distance = distance / max_len

        print(f"Aligned Pair (S1:{idx1+1}, S2:{idx2+1}) | Edit Distance: {distance}, Normalized: {normalized_distance:.2f}")

        if normalized_distance <= normalized_distance_threshold:
            plagiarized_pairs += 1

print("\n--- Plagiarism Summary ---")
if plagiarized_pairs == 0:
    print(f"No sentences were found below the {normalized_distance_threshold:.2f} normalized distance threshold.")
else:
    print(f"Found {plagiarized_pairs} sentence pairs below the {normalized_distance_threshold:.2f} normalized distance threshold.")

