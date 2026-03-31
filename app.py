import os
import time
import warnings
import umap
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, mixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
import matplotlib.cm as cm 
import matplotlib.colors as mcolors 
from flask import Flask, request, jsonify
from flask_cors import CORS 


app = Flask(__name__)
CORS(app) # Enable CORS for all routes


DEFAULT_PARAMS = {
    "MiniBatchKMeans": {"n_clusters": 8, "random_state": 42, "init": "k-means++"},
    "KMeans": {"n_clusters": 8, "random_state": 42, "init": "k-means++", "algorithm": "lloyd"},
    "AffinityPropagation": {"damping": 0.5, "preference": -50, "affinity": "euclidean"},
    "MeanShift": {"bandwidth": 1.0, "cluster_all": False},
    "SpectralClustering": {"n_clusters": 8, "affinity": "nearest_neighbors", "n_neighbors": 10, "assign_labels": "kmeans", "random_state": 42},
    "AgglomerativeClustering": {"n_clusters": 8, "linkage": "ward", "metric": "euclidean"}, # Ward needs euclidean
    "DBSCAN": {"eps": 0.5, "min_samples": 5, "metric": "euclidean", "leaf_size": 30, "algorithm": "auto"},
    "HDBSCAN": {"min_cluster_size": 5, "min_samples": 5, "cluster_selection_epsilon": 0.0, "max_cluster_size": 40, "metric": "euclidean", "leaf_size": 20, "cluster_selection_method": "eom"},
    "OPTICS": {"min_samples": 5, "xi": 0.05, "min_cluster_size": 0.05, "metric": "minkowski", "predecessor_correction": True, "algorithm": "auto"},
    "BIRCH": {"n_clusters": 8, "threshold": 0.5, "branching_factor": 50, "compute_labels": True},
    "GaussianMixture": {"n_components": 8, "covariance_type": "full", "random_state": 42, "init_params": "kmeans"},
}


ALGORITHM_PARAMS = {
    "MiniBatchKMeans": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "init", "type": "select", "default": "k-means++", "options": ["k-means++", "random"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "KMeans": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "init", "type": "select", "default": "k-means++", "options": ["k-means++", "random"]},
        {"name": "algorithm", "type": "select", "default": "lloyd", "options": ["lloyd", "elkan"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "AffinityPropagation": [
        {"name": "damping", "type": "float", "default": 0.5, "min": 0.5, "max": 1.0, "step": 0.01},
        {"name": "preference", "type": "float", "default": -50, "min": -1000, "max": 0, "step": 1},
        {"name": "affinity", "type": "select", "default": "euclidean", "options": ["euclidean"]}, # 'precomputed' requires special input, keep simple for now
    ],
    "MeanShift": [
        {"name": "bandwidth", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "nullable": True},
        {"name": "cluster_all", "type": "boolean", "default": True},
    ],
    "SpectralClustering": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "affinity", "type": "select", "default": "nearest_neighbors", "options": ["nearest_neighbors", "rbf"]}, # rbf requires gamma
        {"name": "n_neighbors", "type": "int", "default": 10, "min": 1, "max": 50, "step": 1}, # Only for nearest_neighbors affinity
        {"name": "assign_labels", "type": "select", "default": "kmeans", "options": ["kmeans", "discretize"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "AgglomerativeClustering": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "linkage", "type": "select", "default": "ward", "options": ["ward", "complete", "average", "single"]},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "l1", "l2", "manhattan", "cosine"]}, # Ward only works with euclidean
    ],
    "DBSCAN": [
        {"name": "eps", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1},
        {"name": "min_samples", "type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "chebyshev", "minkowski"]},
        {"name": "leaf_size", "type": "int", "default": 30, "min": 10, "max": 100, "step": 10},
        {"name": "algorithm", "type": "select", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"]},
    ],
    "HDBSCAN": [
        {"name": "min_cluster_size", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples", "type": "int", "default": 15, "min": 1, "max": 20, "step": 1, "nullable": True},
        {"name": "cluster_selection_epsilon", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "max_cluster_size", "type": "int", "default": 20, "min": 10, "max": 100, "step": 10, "nullable": True},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "l1", "l2", "chebyshev"]},
        {"name": "leaf_size", "type": "int", "default": 20, "min": 10, "max": 100, "step": 10},
        {"name": "cluster_selection_method", "type": "select", "default": "eom", "options": ["eom", "leaf"]},
    ],
    "OPTICS": [
        {"name": "min_samples", "type": "int", "default": 5, "min": 2, "max": 50, "step": 1},
        {"name": "xi", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "min_cluster_size", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}, # Can also be int
        {"name": "metric", "type": "select", "default": "minkowski", "options": ["minkowski", "euclidean", "manhattan", "chebyshev"]},
        {"name": "predecessor_correction", "type": "boolean", "default": True},
        {"name": "algorithm", "type": "select", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"]},
    ],
    "BIRCH": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1, "nullable": True}, # Can be None
        {"name": "threshold", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "branching_factor", "type": "int", "default": 50, "min": 20, "max": 200, "step": 10},
        {"name": "compute_labels", "type": "boolean", "default": True},
    ],
    "GaussianMixture": [
        {"name": "n_components", "type": "int", "default": 8, "min": 1, "max": 50, "step": 1},
        {"name": "covariance_type", "type": "select", "default": "full", "options": ["full", "tied", "diag", "spherical"]},
        {"name": "init_params", "type": "select", "default": "kmeans", "options": ["kmeans", "random", "k-means++", "random_from_data"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
}

REDUCTION_PARAMS = {
    "UMAP": [
        {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 10, "step": 1},
        {"name": "n_neighbors", "type": "int", "default": 15, "min": 2, "max": 200, "step": 1},
        {"name": "min_dist", "type": "float", "default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "cosine"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "TSNE": [
        {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 3, "step": 1},
        {"name": "perplexity", "type": "float", "default": 30.0, "min": 5.0, "max": 100.0, "step": 1.0},
        {"name": "learning_rate", "type": "float", "default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0},
        {"name": "n_iter", "type": "int", "default": 1000, "min": 250, "max": 5000, "step": 50},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "cosine"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "PCA": [
        {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 10, "step": 1},
        {"name": "svd_solver", "type": "select", "default": "auto", "options": ["auto", "full", "arpack", "randomized"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
}



def load_data(file_path, file_content, file_type, name_column, intensity_start_index):
    """
    Loads data from various file types (CSV, TSV, XLSX) and preprocesses it.
    """
    df = None
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_content)
        elif file_type == 'tsv':
            df = pd.read_csv(file_content, sep='\t')
        elif file_type == 'xlsx':
            df = pd.read_excel(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


    for col in df.columns[intensity_start_index:]:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    
    intensity_cols = df.columns[intensity_start_index:]
    data = df[intensity_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

    data[np.isnan(data)] = 0.0
    data = np.log2(data + 1.0)

    nonzero_mask = ~(np.all(data == 0.0, axis=1))
    data = data[nonzero_mask]

    if name_column in df.columns:
        names = df.loc[nonzero_mask, name_column].astype(str).tolist()
    else:
        warnings.warn(f"Name column '{name_column}' not found. Using default names.")
        names = [f"Node_{i+1}" for i in range(data.shape[0])]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.T).T 
    data_log10_transformed = np.log10(data + 1)

    return data_scaled, names, data_log10_transformed


def apply_dimensionality_reduction(X, method, params):
    """Applies selected dimensionality reduction."""
    if method == "UMAP":
        reducer = umap.UMAP(**params)
        X_reduced = reducer.fit_transform(X)
    elif method == "TSNE":
        reducer = TSNE(**params)
        X_reduced = reducer.fit_transform(X)
    elif method == "PCA":
        reducer = PCA(**params)
        X_reduced = reducer.fit_transform(X)
    else: 
        X_reduced = X

    if X_reduced.shape[1] < 2:
        raise ValueError(f"Dimensionality reduction resulted in {X_reduced.shape[1]} dimensions. At least 2 are needed for 2D visualization.")
    return X_reduced[:, :2] 


def run_clustering(X, algo_name, config, connectivity=None):
    """Runs the selected clustering algorithm."""
    labels = None
    runtime = 0.0
    error_msg = None

    try:
        # Special handling for AgglomerativeClustering's connectivity
        if algo_name == "AgglomerativeClustering" and connectivity is not None:
            # Ensure 'ward' linkage uses 'euclidean' metric
            if config.get("linkage") == "ward" and config.get("metric") != "euclidean":
                warnings.warn("Ward linkage requires 'euclidean' metric. Overriding metric to 'euclidean'.")
                config["metric"] = "euclidean"
            algo = cluster.AgglomerativeClustering(connectivity=connectivity, **config)
        elif algo_name == "MiniBatchKMeans":
            algo = cluster.MiniBatchKMeans(**config)
        elif algo_name == "KMeans":
            algo = cluster.KMeans(**config)
        elif algo_name == "AffinityPropagation":

            if config.get("affinity") == "precomputed":
                warnings.warn("AffinityPropagation with 'precomputed' affinity requires a distance matrix input. Proceeding with 'euclidean'.")
                config["affinity"] = "euclidean" # Fallback if precomputed is not given
            algo = cluster.AffinityPropagation(**{k: v for k, v in config.items() if k not in ['random_state']})
        elif algo_name == "MeanShift":
            # MeanShift does not take 'n_jobs' directly, relies on underlying KDTree if used
            algo = cluster.MeanShift(**{k: v for k, v in config.items() if v is not None and k != 'n_jobs'}) # Remove n_jobs if it's there
        elif algo_name == "SpectralClustering":
            # SpectralClustering can use n_jobs with 'nearest_neighbors' affinity
            algo = cluster.SpectralClustering(**config)
        elif algo_name == "DBSCAN":
            algo = cluster.DBSCAN(**config)
        elif algo_name == "HDBSCAN":
            # HDBSCAN has n_jobs
            algo = HDBSCAN(**config)
        elif algo_name == "OPTICS":
            algo = cluster.OPTICS(**config)
        elif algo_name == "BIRCH":
            algo = cluster.Birch(**config)
        elif algo_name == "GaussianMixture":
            # Correct init_params smart quotes for GaussianMixture
            if config.get("init_params") == "‘k-means++":
                config["init_params"] = "k-means++"
            elif config.get("init_params") == "‘random_from_data’":
                config["init_params"] = "random_from_data"
            algo = mixture.GaussianMixture(**config)
        else:
            error_msg = f"Unknown algorithm: {algo_name}"
            return None, None, error_msg

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(algo, 'fit_predict'):
                labels = algo.fit_predict(X)
            else:
                algo.fit(X)
                labels = getattr(algo, 'labels_', algo.predict(X))
        t1 = time.time()
        runtime = t1 - t0

        if labels is None:
            raise ValueError("Clustering did not produce labels.")

    except Exception as e:
        error_msg = str(e)

    return labels, runtime, error_msg


def evaluate(labels, X):
    """Calculates clustering evaluation metrics."""
    if labels is None or len(set(labels)) <= 1:
        return -1, 0, -1, -1
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        non_noise_mask = (labels != -1)
        if np.sum(non_noise_mask) < 2:
            return -1, 0, -1, -1 
        
        filtered_labels = labels[non_noise_mask]
        filtered_X = X[non_noise_mask]
        
        if len(set(filtered_labels)) <= 1: 
            return -1, 0, -1, -1

        sil_score = silhouette_score(filtered_X, filtered_labels)
        ch_score = calinski_harabasz_score(filtered_X, filtered_labels)
        db_score = davies_bouldin_score(filtered_X, filtered_labels)
        n_clusters = len(set(filtered_labels))
    else:
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        n_clusters = len(unique_labels)
        
    return sil_score, n_clusters, ch_score, db_score


@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Returns the list of available clustering algorithms and their parameters."""
    return jsonify({
        "clustering_algorithms": list(ALGORITHM_PARAMS.keys()),
        "dimensionality_reduction_methods": list(REDUCTION_PARAMS.keys()),
        "algorithm_params": ALGORITHM_PARAMS,
        "reduction_params": REDUCTION_PARAMS,
        "default_params": DEFAULT_PARAMS 
    })

@app.route('/api/cluster', methods=['POST'])
def cluster_data():
    """Receives data and parameters, performs clustering, and returns results."""
    #import pdb; pdb.set_trace()
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        name_column = request.form.get('nameColumn')
        intensity_start_index = int(request.form.get('intensityStartIndex'))
        
        reduction_method = request.form.get('reductionMethod', 'None')
        reduction_params_str = request.form.get('reductionParams', '{}')
        
        clustering_algo = request.form.get('clusteringAlgorithm')
        clustering_params_str = request.form.get('clusteringParams', '{}')

        reduction_params = {}
        if reduction_params_str:
            try:
                reduction_params = eval(reduction_params_str) # Using eval for simplicity, but json.loads is safer if strict JSON
                for k, v in reduction_params.items():
                    if isinstance(v, str) and v.lower() == 'none':
                        reduction_params[k] = None
            except Exception as e:
                return jsonify({"error": f"Invalid reduction parameters JSON: {e}"}), 400
        
        clustering_params = {}
        if clustering_params_str:
            try:
                #clustering_params = eval(clustering_params_str) # Using eval for simplicity
                clustering_params = json.loads(clustering_params_str)
                for k, v in clustering_params.items():
                    if isinstance(v, str) and v.lower() == 'none':
                        clustering_params[k] = None
            except Exception as e:
                return jsonify({"error": f"Invalid clustering parameters JSON: {e}"}), 400

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input parameter type: {e}"}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    file_type = None
    if file_extension == '.csv':
        file_type = 'csv'
    elif file_extension == '.tsv':
        file_type = 'tsv'
    elif file_extension in ['.xls', '.xlsx']:
        file_type = 'xlsx'
    else:
        return jsonify({"error": "Unsupported file format. Please upload TSV."}), 400

    try:

        original_data, names, data_log10_transformed = load_data(file.filename, file.stream, file_type, name_column, intensity_start_index)
        
        X_processed = apply_dimensionality_reduction(original_data, reduction_method, reduction_params)
        
        connectivity = None
        if clustering_algo == "AgglomerativeClustering":
            try:
                connectivity = kneighbors_graph(X_processed, n_neighbors=min(X_processed.shape[0]-1, 10), include_self=False)
                connectivity = 0.5 * (connectivity + connectivity.T) # Symmetrize
            except Exception as e:
                warnings.warn(f"Could not compute connectivity graph for Agglomerative: {e}. Running without connectivity.")
                connectivity = None # Revert to None if error

        labels, runtime, error = run_clustering(X_processed, clustering_algo, clustering_params, connectivity)

        if error:
            return jsonify({"error": f"Clustering failed: {error}"}), 500

        sil_score, n_clusters, ch_score, db_score = evaluate(labels, X_processed)

        unique_labels = sorted(set(labels))

        cmap = cm.get_cmap("tab20")
        colors_map = {}
        color_idx = 0
        for label in unique_labels:
            python_label = int(label)
            if python_label == -1: 
                colors_map[python_label] = "#808080"
            else:
                colors_map[python_label] = mcolors.rgb2hex(cmap(color_idx % 20))
                color_idx += 1

        cluster_results = []
        for i in range(len(names)):
            cluster_results.append({
                "name": names[i],
                "x": float(X_processed[i, 0]),
                "y": float(X_processed[i, 1]),
                "cluster": int(labels[i]),
                "color": colors_map.get(int(labels[i]), "#000000"),
                "intensities": data_log10_transformed[i].tolist()
            })

        response_data = {
            "status": "success",
            "metrics": {
                "silhouette_score": float(sil_score),
                "n_clusters": int(n_clusters),
                "calinski_harabasz_score": float(ch_score),
                "davies_bouldin_score": float(db_score),
                "runtime_seconds": float(runtime)
            },
            "cluster_data": cluster_results,
            "cluster_colors": colors_map
        }
        return jsonify(response_data), 200

    except ValueError as e:
        return jsonify({"error": f"Data processing error: {e}"}), 400
    except Exception as e:
        app.logger.error(f"An unhandled error occurred: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)