from Bio.SubsMat import MatrixInfo
import numpy as np

# function to read the sequences of a given text file (each line = 1 sequence)
def read_sequences(path):
    with open(path, 'r') as f:
        sequences = f.read().splitlines()

    return sequences

# Calculate the linear penalty (gap penalty) d based on the substitution matrix
def compute_gap_penalty(subs_matrix):
    scores = []
    for (a1, a2), score in subs_matrix.items():
        if a1 != a2:  # Only consider mismatches
            scores.append(score)
    return -int(round(sum(scores) / len(scores)))  # Use the average mismatch score as the penalty

# this function calculates the similarity score from the substitution matrix or gap penalty s(x[i], y[j])
def similarity_score(x, y, subs_matrix, d):
    if x == '-' or y == '-': # gap case
        return -d
    
    elif (x,y) in subs_matrix:
        return subs_matrix[(x, y)]
    
    elif (y, x) in subs_matrix:  # Substitution matrices are symmetric
        return subs_matrix[(y, x)]
    
    else:
        return ValueError(f"Pair ({x}, {y}) not found in the substitution matrix.")
    

# global alignment algorithm that uses DP to solve
def global_alignment(sequence1, sequence2, subs_matrix, d):
    m, n = len(sequence1), len(sequence2)

    #create the dp table
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # create a traceback table
    traceback = np.zeros((m + 1, n + 1), dtype=str)
    
    # Initialize DP table
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] - d
        traceback[i][0] = 'U'  # 'Up'

    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] - d
        traceback[0][j] = 'L'  # 'Left'
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = dp[i - 1][j - 1] + similarity_score(sequence1[i - 1], sequence2[j - 1], subs_matrix, d)
            gap_x = dp[i - 1][j] - d
            gap_y = dp[i][j - 1] - d
            
            #get the maximum score
            dp[i][j] = max(match_score, gap_x, gap_y)
            
            if dp[i][j] == match_score:
                traceback[i][j] = 'D'  # Diagonal
            elif dp[i][j] == gap_x:
                traceback[i][j] = 'U'
            else:
                traceback[i][j] = 'L'
    
    # Traceback to build alignment
    aligned_seq1, aligned_seq2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
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

    return ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2)), dp[m][n]


substitution_matrix = MatrixInfo.pam120  # Use PAM120 matrix from BioPython  # Use PAM120 matrix from BioPython

d = compute_gap_penalty(substitution_matrix)
print(f"Computed Gap Penalty (d): {d}")

sequences_file = "sequences.txt"

sequences = read_sequences(sequences_file)

aligned_seq1, aligned_seq2, score = global_alignment(sequences[0], sequences[1], substitution_matrix, d)

# print the results: 
print(aligned_seq1)

print(aligned_seq2)

print(f"alignment score: {score}")