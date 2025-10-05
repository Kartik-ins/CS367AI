def preprocess_file(filename):
    sentences = [line.strip() for line in open(filename, 'r')]
    words_by_sentence = [s.split(' ') for s in sentences]
    char_counts_by_sentence = [sum(len(word) for word in s) for s in words_by_sentence]
    return sentences, words_by_sentence, char_counts_by_sentence

def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def sentence_alignment_cost(sent1_words, sent2_words):
    m, n = len(sent1_words), len(sent2_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                dp[i][j] = 0
            elif i == 0:
                dp[i][j] = dp[i][j - 1] + len(sent2_words[j - 1])
            elif j == 0:
                dp[i][j] = dp[i - 1][j] + len(sent1_words[i - 1])
            else:
                cost_substitute = dp[i - 1][j - 1] + edit_distance(sent1_words[i - 1], sent2_words[j - 1])
                cost_delete_s1 = dp[i - 1][j] + len(sent1_words[i - 1])
                cost_delete_s2 = dp[i][j - 1] + len(sent2_words[j - 1])
                dp[i][j] = min(cost_substitute, cost_delete_s1, cost_delete_s2)
    return dp[m][n]

def calculate_similarity(doc1_data, doc2_data):
    sentences1, words1, sentence_char_counts1 = doc1_data
    sentences2, words2, sentence_char_counts2 = doc2_data

    heuristic_grid = [[0] * (len(sentences2) + 1) for _ in range(len(sentences1) + 1)]
    for i in range(len(sentences1) - 1, -1, -1):
        for j in range(len(sentences2) - 1, -1, -1):
            align_cost = sentence_alignment_cost(words1[i], words2[j])
            cost1 = heuristic_grid[i + 1][j + 1] + align_cost
            cost2 = heuristic_grid[i + 1][j] + sentence_char_counts1[i]
            cost3 = heuristic_grid[i][j + 1] + sentence_char_counts2[j]
            heuristic_grid[i][j] = min(cost1, cost2, cost3)

    state = [0, 0, 0]
    while state[0] < len(sentences1) and state[1] < len(sentences2):
        i, j = state[0], state[1]
        current_cost = state[2]
        
        g_align = current_cost + sentence_alignment_cost(words1[i], words2[j])
        f_align = g_align + heuristic_grid[i + 1][j + 1]
        
        g_skip1 = current_cost + sentence_char_counts1[i]
        f_skip1 = g_skip1 + heuristic_grid[i + 1][j]
        
        g_skip2 = current_cost + sentence_char_counts2[j]
        f_skip2 = g_skip2 + heuristic_grid[i][j + 1]
        
        min_f = min(f_align, f_skip1, f_skip2)
        
        if min_f == f_align:
            state = [i + 1, j + 1, g_align]
        elif min_f == f_skip1:
            state = [i + 1, j, g_skip1]
        else:
            state = [i, j + 1, g_skip2]

    final_cost = state[2]
    if state[0] < len(sentences1):
        final_cost += sum(sentence_char_counts1[state[0]:])
    if state[1] < len(sentences2):
        final_cost += sum(sentence_char_counts2[state[1]:])
        
    return final_cost

doc1_data = preprocess_file('input1.txt')
doc2_data = preprocess_file('input2.txt')

total_cost = calculate_similarity(doc1_data, doc2_data)

total_chars1 = sum(doc1_data[2])
total_chars2 = sum(doc2_data[2])
max_chars = max(total_chars1, total_chars2) if (total_chars1 + total_chars2) > 0 else 1

dissimilarity_ratio = total_cost / max_chars
similarity_percentage = max(0, 1 - dissimilarity_ratio) * 100

print(f"Total alignment cost: {total_cost}")
print(f"Plagiarism Percentage: {similarity_percentage:.2f}%")

