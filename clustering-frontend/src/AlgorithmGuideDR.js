// src/AlgorithmGuideDR.js
import React from 'react';

const AlgorithmGuideDR = () => (
  <div style={{ padding: '20px' }}>
    <h2>Dimensionality Reduction Methods</h2>

    <h3>UMAP</h3>
    <p>
      Uniform Manifold Approximation and Projection (UMAP) is a non-linear dimensionality
      reduction technique that captures both local and global structure. It is well-suited
      for visualizing clusters in high-dimensional data.
    </p>
    <ul>
      <li><strong>n_components:</strong> Number of output dimensions (default: 2).</li>
      <li><strong>n_neighbors:</strong> Controls local vs. global structure (lower = local detail).</li>
      <li><strong>min_dist:</strong> Minimum distance between points in the low-dimensional space (smaller = tighter clusters).</li>
      <li><strong>metric:</strong> Distance metric used in the input space (e.g., euclidean, cosine).</li>
      <li><strong>random_state:</strong> Seed for reproducibility.</li>
    </ul>
    <p><strong>Recommended to tune:</strong>  <code>n_neighbors</code>, <code>min_dist</code></p>

    <h3>t-SNE</h3>
    <p>
      t-distributed Stochastic Neighbor Embedding (t-SNE) is a probabilistic method
      that preserves local neighborhood structures. It excels at visualizing
      small to medium datasets with tight local clusters.
    </p>
    <ul>
      <li><strong>n_components:</strong> Target dimensionality (default: 2).</li>
      <li><strong>perplexity:</strong> Balances attention between local/global aspects (usually between 5 and 50).</li>
      <li><strong>learning_rate:</strong> Optimization step size (too small or too large harms convergence).</li>
      <li><strong>n_iter:</strong> Number of optimization iterations (default: 1000).</li>
      <li><strong>metric:</strong> Distance metric in input space.</li>
      <li><strong>random_state:</strong> Seed for reproducibility.</li>
    </ul>
    <p><strong>Recommended to tune:</strong>  <code>perplexity</code>, <code>learning_rate</code></p>

    <h3>PCA</h3>
    <p>
      Principal Component Analysis (PCA) is a linear dimensionality reduction
      technique that transforms the data to maximize variance. It is very fast
      and ideal for capturing global patterns in the data.
    </p>
    <ul>
      <li><strong>n_components:</strong> Number of principal components to keep (e.g., 2).</li>
      <li><strong>whiten:</strong> Whether to normalize components to unit variance.</li>
      <li><strong>svd_solver:</strong> Algorithm for SVD ('auto', 'full', 'randomized').</li>
      <li><strong>random_state:</strong> Only used when <code>svd_solver='randomized'</code>.</li>
    </ul>
    <p><strong>Recommended to tune:</strong>  <code>n_components</code>, <code>whiten</code></p>

    <h3>None</h3>
    <p>
      No dimensionality reduction is applied. The clustering operates directly
      in the original high-dimensional feature space.
    </p>
  </div>
);

export default AlgorithmGuideDR;