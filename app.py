import time
import numpy as np
import streamlit as st 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.k_means import k_means_clustering
from src.k_medoids import k_medoids_clustering

st.set_page_config(page_title="Clustering", layout="wide", page_icon = "https://software.llnl.gov/muster/kmedoids.png")

hide_menu_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

breast_cancer = datasets.load_breast_cancer()

num_classes=len(breast_cancer["target_names"])
random_seed=0
data = breast_cancer["data"]
max_iter=100

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

left, right = st.columns(2)

with left:

    labels, centroids, cluster_center_history, labels_history = k_means_clustering(
        data, k=num_classes, random_seed=random_seed, max_iter=max_iter
    )
    pca = PCA(n_components=2, random_state=42).fit(data)
    X_reduced = pca.transform(data)

    fig, ax = plt.subplots()
    ax.set(
            title=f"K-Means Method",
        )
    holder = st.empty()

    for i in range(cluster_center_history.shape[0]):
        labels = labels_history[i]
        accuracy = sum(labels == breast_cancer["target"])/len(breast_cancer["target"])
        # ax.set(
        #     xlabel=f"Accuracy: {accuracy: 4f}",
        # )
        for k in range(num_classes):
            
            idxs = np.where(labels == k)[0]
            ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], color=colors[k], alpha=0.5)
        
            idxs = np.where(labels == k)[0]
            ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], color=colors[k], alpha=0.5)
            centroids_ = pca.transform(cluster_center_history[:i+1, k, :])
            ax.plot(
                centroids_[:, 0],
                centroids_[:, 1],
                marker="*",
                color=centroid_colors[k],
                markersize=8,
            )
            holder.pyplot(fig=fig)
            time.sleep(1)
            
    centroids_ = pca.transform(centroids)
    for k in range(num_classes):
        ax.plot(
            centroids_[k, 0],
            centroids_[k, 1],
            marker="*",
            color=centroid_colors[k],
            markersize=20,
        )
    holder.pyplot(fig=fig)


with right:

    labels, centroids, cluster_center_history, labels_history = k_medoids_clustering(
        data, k=num_classes, random_seed=random_seed, max_iter=max_iter
    )
    pca = PCA(n_components=2, random_state=42).fit(data)
    X_reduced = pca.transform(data)

    fig, ax = plt.subplots()
    # ax.set(
    #         title=f"K-Medoids Method",
    #     )
    holder = st.empty()

    for i in range(cluster_center_history.shape[0]):
        labels = labels_history[i]
        for k in range(num_classes):
            accuracy = sum(labels == breast_cancer["target"])/len(breast_cancer["target"])
            ax.set(
                xlabel=f"Accuracy: {accuracy: 4f}",
            )
            
            idxs = np.where(labels == k)[0]
            ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], color=colors[k], alpha=0.5)

            idxs = np.where(labels == k)[0]
            ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], color=colors[k], alpha=0.5)
            centroids_ = pca.transform(cluster_center_history[:i+1, k, :])
            ax.plot(
                centroids_[:, 0],
                centroids_[:, 1],
                marker="*",
                color=centroid_colors[k],
                markersize=8,
            )
            holder.pyplot(fig=fig)
            time.sleep(1)
            
    centroids_ = pca.transform(centroids)
    for k in range(num_classes):
        ax.plot(
            centroids_[k, 0],
            centroids_[k, 1],
            marker="*",
            color=centroid_colors[k],
            markersize=20,
        )
    holder.pyplot(fig=fig)