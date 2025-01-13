import numpy as np
from Bio.SubsMat import MatrixInfo

# Function to read sequences from a file
def read_sequences(path):
    with open(path, 'r') as f:
        sequences = f.read().splitlines()
    return sequences

# Function to compute the gap penalty (linear penalty) `d`
def compute_gap_penalty(subs_matrix):
    scores = []
    for (a1, a2), score in subs_matrix.items():
        if a1 != a2:  # Only consider mismatches
            scores.append(score)
    return -int(round(sum(scores) / len(scores)))  # Average mismatch score as penalty

# Similarity score calculation
def similarity_score(x, y, subs_matrix, d):
    if x == '-' or y == '-':  # Gap
        return -d
    elif (x, y) in subs_matrix:
        return subs_matrix[(x, y)]
    elif (y, x) in subs_matrix:  # Substitution matrices are SYMMETRIC
        return subs_matrix[(y, x)]
    else:
        raise ValueError(f"Pair ({x}, {y}) not found in the substitution matrix.")

# Local alignment using dynamic programming
def local_alignment(sequence1, sequence2, subs_matrix, d):
    m, n = len(sequence1), len(sequence2)

    # Initialize DP and traceback tables
    dp = np.zeros((m + 1, n + 1), dtype=int)
    traceback = np.zeros((m + 1, n + 1), dtype=str)

    max_score = 0
    max_pos = (0, 0)

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = dp[i - 1][j - 1] + similarity_score(sequence1[i - 1], sequence2[j - 1], subs_matrix, d)
            gap_x = dp[i - 1][j] - d
            gap_y = dp[i][j - 1] - d
            dp[i][j] = max(0, match_score, gap_x, gap_y)

            # Update traceback table
            if dp[i][j] == 0:
                traceback[i][j] = '0'
            elif dp[i][j] == match_score:
                traceback[i][j] = 'D'  # Diagonal
            elif dp[i][j] == gap_x:
                traceback[i][j] = 'U'  # Up
            elif dp[i][j] == gap_y:
                traceback[i][j] = 'L'  # Left

            # Track the maximum score and its position
            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_pos = (i, j)

    # Traceback from the max_pos to the first 0
    aligned_seq1, aligned_seq2 = [], []
    i, j = max_pos
    while i > 0 and j > 0 and dp[i][j] != 0:
        if traceback[i][j] == 'D':
            aligned_seq1.append(sequence1[i - 1])
            aligned_seq2.append(sequence2[j - 1])
            i -= 1
            j -= 1
        elif traceback[i][j] == 'U':
            aligned_seq1.append(sequence1[i - 1])
            aligned_seq2.append('-')
            i -= 1
        elif traceback[i][j] == 'L':
            aligned_seq1.append('-')
            aligned_seq2.append(sequence2[j - 1])
            j -= 1

    # Reverse the alignments
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))

    return aligned_seq1, aligned_seq2, max_score


# Load substitution matrix
substitution_matrix = MatrixInfo.pam120  # Use PAM120 matrix from BioPython  # Use PAM120 matrix from BioPython

# Compute gap penalty
d = compute_gap_penalty(substitution_matrix)
print(f"Computed Gap Penalty (d): {d}")

# Read sequences
sequences_file = "sequences.txt"
sequences = read_sequences(sequences_file)

# Perform local alignment
aligned_seq1, aligned_seq2, score = local_alignment(sequences[0], sequences[1], substitution_matrix, d)

# Print the results
print("Aligned Sequences:")
print(aligned_seq1)
print(aligned_seq2)
print(f"Local Alignment Score: {score}")
