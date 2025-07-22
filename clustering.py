import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
  return np.linalg.norm(a - b)

def init_centroid(X, k):
  indices = np.random.choice(len(X), k, replace=False)
  return X[indices]

def init_centroid_kmeanspp(X, k):
  centroids = []
  centroids.append(X[np.random.choice(len(X))])
  for _ in range(1, k):
    dist_sq = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
    probs = dist_sq / dist_sq.sum()
    cumulative_probs = probs.cumsum()
    r = np.random.rand()
    for i, p in enumerate(cumulative_probs):
      if r < p:
        centroids.append(X[i])
        break
  return np.array(centroids)

def update_centroids(X, clusters, k):
  new_centroids = []
  for i in range(k):
    points = X[clusters == i]
    if len(points) == 0:
      new_centroids.append(X[np.random.choice(len(X))])
    else:
      new_centroids.append(points.mean(axis=0))
  return np.array(new_centroids)

def assign_clusters(X, centroids):
  clusters = []
  for x in X:
    distances = [euclidean_distance(x, c) for c in centroids]
    clusters.append(np.argmin(distances))
  return np.array(clusters)

def kmeans(X, k, kmeans_pp=False, max_iter=100, tol=1e-4, elbow=False):
  if kmeans_pp:
    centroids = init_centroid_kmeanspp(X, k)
  else:
    centroids = init_centroid(X, k)
  clusters = []
  for _ in range(max_iter):
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, clusters, k)
    if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
      break
    centroids = new_centroids
  wcss = 0
  if elbow:
    for i in range(k):
      points = X[clusters == i]
      wcss += np.sum((points - centroids[i]) ** 2)
  return clusters, centroids, wcss

def elbow_method(X, max_k=10):
  wcss_values = []
  for k in range(1, max_k + 1):
    _, _, wcss = kmeans(X, k, elbow=True)
    wcss_values.append(wcss)
  plt.figure(figsize=(8, 5))
  plt.plot(range(1, max_k + 1), wcss_values, marker='o')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('WCSS (Inertia)')
  plt.title('Elbow Method for Optimal k')
  plt.xticks(range(1, max_k + 1))
  plt.grid(True)
  plt.savefig("./plots/elbow.png", dpi=500)
  return wcss_values