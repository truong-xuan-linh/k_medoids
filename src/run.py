import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_experiment(
    data: np.ndarray,
    num_classes: int,
    clustering_algorithm,
    random_seed=42,
    max_iter=100,
):
    colors = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    centroid_colors = [
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
    ]
    labels, centroids, cluster_center_history = clustering_algorithm(
        data, k=num_classes, random_seed=random_seed, max_iter=max_iter
    )
    pca = PCA(n_components=2, random_state=42).fit(data)
    X_reduced = pca.transform(data)

    fig, ax = plt.subplots()

    for k in range(num_classes):
        idxs = np.where(labels == k)[0]
        ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], color=colors[k], alpha=0.5)
        centroids_ = pca.transform(cluster_center_history[:, k, :])
        ax.plot(
            centroids_[:, 0],
            centroids_[:, 1],
            marker="*",
            color=centroid_colors[k],
            markersize=8,
        )

    # plot trajectory of cluster centroid
    centroids_ = pca.transform(centroids)
    for k in range(num_classes):
        ax.plot(
            centroids_[k, 0],
            centroids_[k, 1],
            marker="*",
            color=centroid_colors[k],
            markersize=20,
        )
    ax.set(
        title=f"{clustering_algorithm.__name__}: Trajectory of Cluster Centroid",
        xlabel="pca(0)",
        ylabel="pca(1)",
    )
    plt.show()