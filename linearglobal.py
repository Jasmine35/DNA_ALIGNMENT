import numpy as np

# Define the 5×5 similarity matrix for DNA
similarity_matrix = {
    ('A', 'A'): 1, ('A', 'C'): 0, ('A', 'G'): 0, ('A', 'T'): 0, ('A', '-'): 0,
    ('C', 'A'): 0, ('C', 'C'): 1, ('C', 'G'): 0, ('C', 'T'): 0, ('C', '-'): 0,
    ('G', 'A'): 0, ('G', 'C'): 0, ('G', 'G'): 1, ('G', 'T'): 0, ('G', '-'): 0,
    ('T', 'A'): 0, ('T', 'C'): 0, ('T', 'G'): 0, ('T', 'T'): 1, ('T', '-'): 0,
    ('-', 'A'): 0, ('-', 'C'): 0, ('-', 'G'): 0, ('-', 'T'): 0, ('-', '-'): 0
}

# Score function
def score(a, b):
    return similarity_matrix.get((a, b), 0)

# Compute the last row of the DP table
def compute_last_row(seq1, seq2):
    n = len(seq1)
    m = len(seq2)
    prev = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        curr = np.zeros(m + 1, dtype=int)
        for j in range(1, m + 1):
            match = prev[j - 1] + score(seq1[i - 1], seq2[j - 1])
            delete = prev[j] + score(seq1[i - 1], '-')
            insert = curr[j - 1] + score('-', seq2[j - 1])
            curr[j] = max(match, delete, insert)
        prev = curr

    return prev

# Hirschberg's algorithm for global alignment
def hirschberg(seq1, seq2):
    n = len(seq1)
    m = len(seq2)

    if n == 0:
        return '-' * m, seq2
    if m == 0:
        return seq1, '-' * n
    if n == 1 or m == 1:
        # Fallback to standard alignment for small problems
        return needleman_wunsch(seq1, seq2)

    mid = n // 2
    left_score = compute_last_row(seq1[:mid], seq2)
    right_score = compute_last_row(seq1[mid:][::-1], seq2[::-1])[::-1]
    split = np.argmax(left_score + right_score)

    left_alignment = hirschberg(seq1[:mid], seq2[:split])
    right_alignment = hirschberg(seq1[mid:], seq2[split:])

    return (left_alignment[0] + right_alignment[0], left_alignment[1] + right_alignment[1])

# Standard Needleman-Wunsch for small alignments
def needleman_wunsch(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + score(seq1[i - 1], '-')
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + score('-', seq2[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + score(seq1[i - 1], seq2[j - 1])
            delete = dp[i - 1][j] + score(seq1[i - 1], '-')
            insert = dp[i][j - 1] + score('-', seq2[j - 1])
            dp[i][j] = max(match, delete, insert)

    aligned_seq1, aligned_seq2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + score(seq1[i - 1], seq2[j - 1]):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + score(seq1[i - 1], '-'):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append('-')
            i -= 1
        else:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    return ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2))

# Read sequences
with open("APOL3.fasta", "r") as f:
    seq1 = f.read().strip()

with open("APOL4.fasta", "r") as f:
    seq2 = f.read().strip()

# Perform alignment
aligned_seq1, aligned_seq2 = hirschberg(seq1, seq2)

# Output results
print("Aligned Sequences:")
print(aligned_seq1)
print(aligned_seq2)
print(f"Alignment Length: {len(aligned_seq1)}")
