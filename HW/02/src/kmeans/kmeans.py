from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def kmeans():
    X = np.array([
        [1,2], [1, 2.1], [1, 1.9], [1.1, 2],
        [1,4], [1.2, 4], [0.8, 4.3], [1, 3.8], 
        [10, 2], [10, 2.4], [10, 1.6], [11,2.1]
    ])

    NUM_CLUSTERS = 3
    kmeans_model = KMeans(
        n_clusters=NUM_CLUSTERS, 
        random_state=0
    )

    kmeans_model.fit(X)

    labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    # Plot each cluster's points
    for i in range(NUM_CLUSTERS):
        cluster_pts = X[labels == i]
        plt.scatter(
            x=cluster_pts[:, 0], 
            y=cluster_pts[:, 1],
            label=f"Cluster {i}"
        )

    # Plot the centroids
    plt.scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        c="red",
        marker="x",
        label="Centroids"
    )

    plt.legend() 
    plt.savefig("kmeans.png")
