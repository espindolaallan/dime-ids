"""
Clustered Representative Sampling (CRS)
---------------------------------------
This script enumerates all 4-of-16 class combinations, represents each as a binary vector, and computes
a chosen distance metric (Jaccard or Hamming) to form a distance matrix. It then uses hierarchical 
clustering to group similar subsets and selects 30 "medoid" representatives. Finally, it creates a 
coverage table (Train vs. Test) showing how often each class is included or excluded in these 30 
representative subsets.

Usage:
    python crs.py
"""

from itertools import combinations
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from EspPipeML import esp_utilities

distance = 'jaccard'  # 'jaccard' or 'hamming'

# Step 1: Generate all combinations of 4 out of 16 classes
n_classes = 16 # Number of classes
k = 4 # Number of classes to select
all_combos = list(combinations(range(n_classes), k))
num_combos = len(all_combos)
print(f"Number of combinations: {num_combos}")

# Step 2: Represent each combination as a 16-dimensional binary vector
binary_vectors = np.zeros((num_combos, n_classes), dtype=int)
for i, combo in enumerate(all_combos):
    for c in combo:
        binary_vectors[i, c] = 1

# Step 3: distance_matrix
distance_matrix = np.zeros((num_combos, num_combos))
# Compute the Jaccard distance matrix
if distance == 'jaccard':
    for i in range(num_combos):
        intersection_counts = (binary_vectors[i] & binary_vectors).sum(axis=1)
        # |A| = 4, |B| = 4, so |A ∪ B| = 8 - |A ∩ B|
        union_counts = 8 - intersection_counts
        jaccard_similarity = intersection_counts / union_counts
        jaccard_distance = 1 - jaccard_similarity
        distance_matrix[i, :] = jaccard_distance
# Compute the Hamming distance matrix
elif distance == 'hamming':
    for i in range(num_combos):
        hamming_distance = np.sum(binary_vectors[i] != binary_vectors, axis=1)
        distance_matrix[i, :] = hamming_distance

np.fill_diagonal(distance_matrix, 0.0)

# Step 4: Perform hierarchical clustering using precomputed distance
n_clusters = 30 
clustering = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='average',
    metric='precomputed'
)

labels = clustering.fit_predict(distance_matrix)

# Step 5: Select a representative subset (medoid) from each cluster
representatives = []
for cluster_id in range(n_clusters):
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 1:
        representatives.append(cluster_indices[0])
    else:
        sub_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_index = np.argmin(np.sum(sub_matrix, axis=1))
        representatives.append(cluster_indices[medoid_index])

print("Representative subsets (by index):", representatives)
print("Representative subsets (as class sets):")
chosen_combos = [all_combos[idx] for idx in representatives] # List of chosen combinations
for combo in chosen_combos:
    print(combo)

# Save the chosen_combos to pickle
esp_utilities.save_to_pickle(chosen_combos, '../results/crs/crs_chosen_combos.pkl')

# Step 6: Create a table of information
# For each class (0-15), count how many chosen subsets selected it (Test)
# and how many did not (Train)
test_counts = np.zeros(n_classes, dtype=int)
for combo in chosen_combos:
    for c in combo:
        test_counts[c] += 1

# Train count = 30 - Test count
train_counts = n_clusters - test_counts

# Coverage table (Class, Train, Test)
data = {
    'Class': np.arange(n_classes),
    'Train': train_counts,
    'Test': test_counts
}

df = pd.DataFrame(data)
df.to_csv('../results/crs/crs_coverage.csv', index=False)
# print("\nCoverage DataFrame:")
# print(df)