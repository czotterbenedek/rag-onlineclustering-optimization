import numpy as np
import pandas as pd


class OnlineKMeans:
    """
    Incremental (online) K-Means clustering with optional:
      - Dynamic cluster creation (based on distance threshold)
      - Cluster merging (based on proximity)
      - Euclidean or cosine distance metric
    """

    def __init__(
        self,
        n_clusters=4,
        max_clusters=20,
        metric="euclidean",
        new_cluster_threshold=None,
        merge_threshold=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.metric = metric
        self.new_cluster_threshold = new_cluster_threshold
        self.merge_threshold = merge_threshold
        self.rng = np.random.RandomState(random_state)

        self.centroids = None
        self.counts = None
        self.sums = None
        self.vars = None
        self.total_seen = 0

    # ---------------------- Helper methods ---------------------- #

    def _normalize(self, X):
        """Normalize vectors for cosine distance."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def _pairwise_dist(self, X, C):
        """Compute pairwise distance between X and centroids C."""
        if self.metric == "euclidean":
            d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
            return np.sqrt(d2 + 1e-12)
        elif self.metric == "cosine":
            Xn = self._normalize(X)
            Cn = self._normalize(C)
            return 1.0 - (Xn @ Cn.T)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _kmeans_pp_init(self, X, k):
        """Initialize centroids using the k-means++ method."""
        n_samples = X.shape[0]
        centers = [self.rng.randint(0, n_samples)]
        d2 = np.full(n_samples, np.inf)

        for _ in range(1, k):
            current_center = X[centers[-1]][None, :]
            dist = np.sum((X - current_center) ** 2, axis=1)
            d2 = np.minimum(d2, dist)
            probs = d2 / (d2.sum() + 1e-12)
            next_idx = self.rng.choice(n_samples, p=probs)
            centers.append(int(next_idx))

        return np.array(centers, dtype=int)

    # ---------------------- Initialization ---------------------- #

    def initialize_centroids(self, X_init):
        """Initialize centroids on the first batch."""
        k = self.n_clusters
        n_samples = len(X_init)

        if n_samples < k:
            # Fallback: random selection with replacement
            indices = self.rng.choice(n_samples, k, replace=True)
        else:
            indices = self._kmeans_pp_init(X_init, k)

        centers = X_init[indices]
        if self.metric == "cosine":
            centers = self._normalize(centers)

        self.centroids = centers.copy()
        self.counts = np.zeros(k, dtype=float)
        self.sums = np.zeros_like(self.centroids)
        self.vars = np.zeros(k, dtype=float)
        self.total_seen = 0

    # ---------------------- Cluster Merging ---------------------- #

    def _merge_close_clusters(self):
        """Merge clusters that are closer than merge_threshold."""
        C = self.centroids
        D = self._pairwise_dist(C, C)
        np.fill_diagonal(D, np.inf)

        merge_pairs = np.argwhere(D < self.merge_threshold)
        if not len(merge_pairs):
            return

        to_remove = set()
        for i, j in merge_pairs:
            if i in to_remove or j in to_remove:
                continue

            n_i, n_j = self.counts[i], self.counts[j]
            total = n_i + n_j if (n_i + n_j) > 0 else 1.0

            # Weighted merge
            self.centroids[i] = (n_i * C[i] + n_j * C[j]) / total
            self.counts[i] = total
            self.sums[i] += self.sums[j]
            self.vars[i] = (self.vars[i] + self.vars[j]) / 2.0
            to_remove.add(j)

        # Keep non-merged clusters
        keep = [idx for idx in range(len(C)) if idx not in to_remove]
        self.centroids = self.centroids[keep]
        self.counts = self.counts[keep]
        self.sums = self.sums[keep]
        self.vars = self.vars[keep]

    # ---------------------- Cluster Splitting ---------------------- #

    def _split_cluster(self, X_batch, labels, threshold=0.2, sigma_factor=3.0):





        return

    # ---------------------- Online Update ---------------------- #

    def partial_fit(self, X_batch):
        """
        Incrementally update centroids with a new batch of points.
        Supports:
          - Dynamic creation of new clusters (if threshold set)
          - Optional merging of nearby clusters
        """
        X = np.asarray(X_batch, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.metric == "cosine":
            X = self._normalize(X)

        # Initialize on first call
        if self.centroids is None:
            self.initialize_centroids(X)

        # Assign points to nearest centroid
        D = self._pairwise_dist(X, self.centroids)
        min_dist = D.min(axis=1)
        labels = D.argmin(axis=1)

        # --- Dynamic cluster creation ---
        if self.new_cluster_threshold is not None and len(self.centroids) < self.max_clusters:
            far_points = np.where(min_dist > self.new_cluster_threshold)[0]
            to_create = min(len(far_points), self.max_clusters - len(self.centroids))

            for idx in far_points[:to_create]:
                new_center = X[idx].copy()
                if self.metric == "cosine":
                    new_center = self._normalize(new_center.reshape(1, -1))[0]

                self.centroids = np.vstack([self.centroids, new_center])
                self.counts = np.append(self.counts, 0.0)
                self.sums = np.vstack([self.sums, np.zeros_like(new_center)])
                self.vars = np.append(self.vars, 0.0)
                labels[idx] = len(self.centroids) - 1

            # Recompute distances if clusters changed
            D = self._pairwise_dist(X, self.centroids)
            labels = D.argmin(axis=1)

        # --- Update centroids incrementally ---
        for k in range(len(self.centroids)):
            mask = labels == k
            if not np.any(mask):
                continue

            cluster_points = X[mask]
            batch_sum = cluster_points.sum(axis=0)
            batch_mean = batch_sum / len(cluster_points)

            n_old = self.counts[k]
            if n_old == 0:
                # Initialize new cluster stats
                self.centroids[k] = batch_mean
                self.counts[k] = len(cluster_points)
                self.sums[k] = batch_sum
                diffs = cluster_points - batch_mean
                self.vars[k] = np.mean(np.sum(diffs ** 2, axis=1))
                continue

            # Incremental update
            n_new = n_old + len(cluster_points)
            mu_old = self.centroids[k]
            mu_batch = batch_mean
            new_centroid = (n_old * mu_old + batch_sum) / n_new

            # Variance update
            diffs = cluster_points - mu_batch
            var_batch = np.mean(np.sum(diffs ** 2, axis=1))
            delta_old = mu_old - new_centroid
            delta_batch = mu_batch - new_centroid
            var_new = (n_old * (self.vars[k] + np.sum(delta_old ** 2)) + len(cluster_points) * (var_batch + np.sum(delta_batch ** 2))) / n_new

            # Commit updates
            self.centroids[k] = new_centroid
            self.counts[k] = n_new
            self.sums[k] += batch_sum
            self.vars[k] = var_new

        self.total_seen += len(X)

        # Normalize centroids for cosine metric
        if self.metric == "cosine":
            self.centroids = self._normalize(self.centroids)

        # Merge close clusters if needed
        if self.merge_threshold is not None and len(self.centroids) > 1:
            self._merge_close_clusters()

    # ---------------------- Utility ---------------------- #

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.metric == "cosine":
            X = self._normalize(X)

        D = self._pairwise_dist(X, self.centroids)
        
        return D.argmin(axis=1)

    def get_state(self):
        """Return current clustering state."""
        return {
            "centroids": self.centroids.copy(),
            "counts": self.counts.copy(),
            "vars": self.vars.copy(),
            "total_seen": self.total_seen,
        }
