file_input = open("../test_with_space.full", "r")
right_input = open("../text8val.txt", "r")

file_lines = file_input.readlines()
right_lines = right_input.readlines()

print(len(file_lines), len(right_lines))



import tqdm

def LCS(ar_x, ar_y):
    n = len(ar_x)
    m = len(ar_y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    print(n*m)
    for i in tqdm.tqdm(range(n)):
        for j in range(m):
            if ar_x[i] == ar_y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[n][m]

Fs = []

for i in range(len(right_lines)):
    file_words = file_lines[i*2].split()
    right_words = right_lines[i].split()
    print(file_words[:10], right_words[:10])
    print(len(file_words), len(right_words))
    if len(file_words) == 1 or len(right_words) == 1:
        print(file_words, right_words)
    lcs = LCS(file_words, right_words)
    precision = lcs / len(file_words)
    recall = lcs / len(right_words)
    f1 = 2 * precision * recall / (precision + recall)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1: " + str(f1))
    Fs.append(f1)

print("average f1: " + str(sum(Fs) / len(Fs)))

#(average f1: 0.5269464572662172)