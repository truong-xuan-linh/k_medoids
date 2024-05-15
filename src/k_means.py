import numpy as np
from typing import List, Tuple

def k_means_clustering(
    data: np.ndarray, k: int, max_iter: int = 100, random_seed=0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data: input array
    k: number of clusters
    max_iter: upper bound of iteration
    """
    np.random.seed(random_seed)
    N = data.shape[0]
    centroids = data[np.random.choice(N, k, replace=False), :]
    cluster_center_history = [centroids.copy()]

    # labels = np.empty(N)
    distances = np.linalg.norm(data[:, None, :] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    labels_history = [labels.copy()]
    
    old_labels = np.empty(N)
    for i in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        for j in range(k):
            centroids[j] = np.mean(data[labels == j], axis=0)
        cluster_center_history.append(centroids.copy())
        labels_history.append(labels.copy())
        # terminate if cluster assingment never changed from the step before
        if i > 0 and np.all(labels == old_labels):
            break

        old_labels = labels
    print(f"* converged after {i + 1} iterations")
    cluster_center_history = np.array(cluster_center_history)
    labels_history = np.array(labels_history)
    
    return labels, centroids, cluster_center_history, labels_history