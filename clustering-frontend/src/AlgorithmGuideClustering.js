import React from 'react';

const AlgorithmGuideClustering = () => {
  return (
    <div style={{ padding: '20px', maxHeight: '60vh', overflowY: 'auto' }}>
      <h2>Clustering Algorithms Guide</h2>

      <h3>MiniBatchKMeans</h3>
      <p>A faster version of KMeans that uses small random batches of data to reduce computation. Suitable for large datasets.</p>
      <ul>
        <li><strong>n_clusters</strong>: Number of clusters to form.</li>
        <li><strong>random_state</strong>: Random seed for reproducibility.</li>
        <li><strong>init</strong>: Initialization method (e.g., "k-means++").</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>n_clusters</code>, <code>init</code></p>

      <h3>KMeans</h3>
      <p>Partitions data into k clusters by minimizing variance within each cluster.</p>
      <ul>
        <li><strong>n_clusters</strong>: Number of clusters to form.</li>
        <li><strong>random_state</strong>: Random seed for reproducibility.</li>
        <li><strong>init</strong>: Initialization strategy (e.g., "k-means++").</li>
        <li><strong>algorithm</strong>: Algorithm to use: "lloyd" (standard) or "elkan" (faster on dense data).</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>n_clusters</code>, <code>algorithm</code></p>

      <h3>AffinityPropagation</h3>
      <p>Clustering by message passing. Finds exemplars automatically based on similarity.</p>
      <ul>
        <li><strong>damping</strong>: Damping factor for convergence (between 0.5 and 1).</li>
        <li><strong>preference</strong>: Controls the number of exemplars.</li>
        <li><strong>affinity</strong>: Similarity measure (e.g., "euclidean").</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>preference</code>, <code>damping</code></p>

      <h3>MeanShift</h3>
      <p>Shifts data points toward the mode in feature space. Non-parametric.</p>
      <ul>
        <li><strong>bandwidth</strong>: Window size to define density region.</li>
        <li><strong>cluster_all</strong>: Whether to assign all points to clusters.</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>bandwidth</code></p>

      <h3>SpectralClustering</h3>
      <p>Clustering via graph Laplacian. Good for non-convex clusters.</p>
      <ul>
        <li><strong>n_clusters</strong>: Number of clusters.</li>
        <li><strong>affinity</strong>: How to construct the similarity graph (e.g., "nearest_neighbors").</li>
        <li><strong>n_neighbors</strong>: Used when affinity is "nearest_neighbors".</li>
        <li><strong>assign_labels</strong>: Label assignment method ("kmeans" or "discretize").</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>affinity</code>, <code>n_neighbors</code></p>

      <h3>AgglomerativeClustering</h3>
      <p>Hierarchical clustering using a bottom-up approach.</p>
      <ul>
        <li><strong>n_clusters</strong>: Number of clusters to find.</li>
        <li><strong>linkage</strong>: Linkage criteria (e.g., "ward", "average").</li>
        <li><strong>metric</strong>: Distance metric.</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>linkage</code>, <code>n_clusters</code></p>

      <h3>DBSCAN</h3>
      <p>Density-Based Spatial Clustering. Finds core samples of high density and expands clusters from them.</p>
      <ul>
        <li><strong>eps</strong>: Maximum distance for neighborhood.</li>
        <li><strong>min_samples</strong>: Minimum samples to form a core point.</li>
        <li><strong>metric</strong>: Distance metric (e.g., "euclidean").</li>
        <li><strong>algorithm</strong>: Search algorithm (e.g., "auto").</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>eps</code>, <code>min_samples</code></p>

      <h3>HDBSCAN</h3>
      <p>Hierarchical DBSCAN. Handles clusters with varying densities. More flexible than DBSCAN.</p>
      <ul>
        <li><strong>min_cluster_size</strong>: Minimum number of points in a cluster.</li>
        <li><strong>min_samples</strong>: Controls how conservative clustering is.</li>
        <li><strong>cluster_selection_epsilon</strong>: Tolerance for merging.</li>
        <li><strong>max_cluster_size</strong>: Optional limit on cluster size.</li>
        <li><strong>metric</strong>: Distance metric.</li>
        <li><strong>cluster_selection_method</strong>: Method used to extract clusters ("eom" or "leaf").</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>min_cluster_size</code>, <code>min_samples</code></p>

      <h3>OPTICS</h3>
      <p>Orders points to identify density-based clustering structure. Good for varied density data.</p>
      <ul>
        <li><strong>min_samples</strong>: Minimum samples in a neighborhood.</li>
        <li><strong>xi</strong>: Determines minimum steepness on reachability plot.</li>
        <li><strong>min_cluster_size</strong>: Minimum size of clusters as a fraction or int.</li>
        <li><strong>metric</strong>: Distance metric.</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>xi</code>, <code>min_samples</code></p>

      <h3>BIRCH</h3>
      <p>Builds a tree structure for efficient clustering of large datasets.</p>
      <ul>
        <li><strong>n_clusters</strong>: Number of clusters.</li>
        <li><strong>threshold</strong>: Distance threshold for subcluster merging.</li>
        <li><strong>branching_factor</strong>: Max number of CF subclusters in a node.</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>threshold</code>, <code>n_clusters</code></p>

      <h3>GaussianMixture</h3>
      <p>Model-based clustering using Gaussian mixture models.</p>
      <ul>
        <li><strong>n_components</strong>: Number of mixture components.</li>
        <li><strong>covariance_type</strong>: Type of covariance parameters (e.g., "full").</li>
        <li><strong>init_params</strong>: Method to initialize (e.g., "kmeans").</li>
      </ul>
      <p><strong>Recommended to tune:</strong> <code>n_components</code>, <code>covariance_type</code></p>
    </div>
  );
};

export default AlgorithmGuideClustering;
