import React, { useState, useEffect, useRef } from 'react';
import Modal from 'react-modal';
// import Plot from 'react-plotly.js-basic-dist';
import Plot from 'react-plotly.js';
import { readString } from 'react-papaparse';
import * as XLSX from 'xlsx';
import './App.css';
import AlgorithmGuideDR from './AlgorithmGuideDR';
import AlgorithmGuideClustering from './AlgorithmGuideClustering';


const methodDescriptions = {
  clustering: {
    MiniBatchKMeans: "Fast, memory-efficient K-Means variant for large datasets.",
    KMeans: "Partitions data into k clusters minimizing within-cluster variance.",
    AffinityPropagation: "Message-passing clustering that identifies exemplars.",
    MeanShift: "Clusters based on data density without predefining cluster number.",
    SpectralClustering: "Graph-based clustering using Laplacian eigenmaps.",
    AgglomerativeClustering: "Hierarchical clustering using bottom-up merging.",
    DBSCAN: "Density-based clustering that labels noise and finds dense areas.",
    HDBSCAN: "Improved DBSCAN for varying density with hierarchical clustering.",
    OPTICS: "Orders data to identify clusters at varying density thresholds.",
    BIRCH: "Scalable clustering using compact feature trees.",
    GaussianMixture: "Probabilistic clustering assuming Gaussian-distributed data."
  },
  reduction: {
    PCA: "Linear projection preserving maximal variance in fewer dimensions.",
    UMAP: "Non-linear projection preserving local and global data structure.",
    TSNE: "Projects high-dimensional data by preserving local similarities.",
    None: "Without applying dimensionality reduction."
  }
};

const paramDescriptions = {
  clustering: {
    n_clusters: "Number of clusters to form",
    init: "Method for initialization",
    random_state: "Seed for reproducibility",
    algorithm: "K-Means computation algorithm",
    damping: "Responsiveness to updates in AffinityPropagation",
    preference: "Preferences for exemplars",
    affinity: "Similarity metric used",
    bandwidth: "Kernel bandwidth for MeanShift",
    cluster_all: "Whether to assign all points to a cluster",
    n_neighbors: "Number of neighbors for graph construction",
    assign_labels: "Method to assign labels in SpectralClustering",
    linkage: "Linkage criteria for merging clusters",
    metric: "Distance metric used",
    eps: "Maximum distance between two samples for DBSCAN",
    min_samples: "Minimum samples to form a dense region",
    leaf_size: "Tree leaf size for neighbor queries",
    algorithm: "Algorithm used for nearest neighbors",
    min_cluster_size: "Minimum size for a cluster in HDBSCAN",
    cluster_selection_epsilon: "Distance tolerance for cluster separation",
    max_cluster_size: "Maximum allowed cluster size",
    cluster_selection_method: "Strategy to extract clusters",
    xi: "Minimum steepness for OPTICS clustering structure",
    threshold: "Radius threshold for subclusters (BIRCH)",
    branching_factor: "Max children per non-leaf node (BIRCH)",
    compute_labels: "Whether to compute final labels",
    n_components: "Number of mixture components",
    covariance_type: "Covariance matrix type",
    init_params: "Initialization method for parameters"
  },
  reduction: {
    n_components: "Output dimensionality",
    n_neighbors: "Size of local neighborhood (UMAP/tSNE)",
    min_dist: "Minimum distance between embedded points (UMAP)",
    metric: "Distance metric",
    perplexity: "Effective number of neighbors (tSNE)",
    learning_rate: "Learning rate for tSNE",
    random_state: "Seed for reproducibility"
  }
};

function App() {
  const [file, setFile] = useState(null);
  const [methodDetails, setMethodDetails] = useState('');
  const [networkPlotData, setNetworkPlotData] = useState([]);
  const [networkPlotLayout, setNetworkPlotLayout] = useState({});
  const [subnetworkData, setSubnetworkData] = useState([]);
  const [topNNeighbors, setTopNNeighbors] = useState(5);
  const [subnetworkLayout, setSubnetworkLayout] = useState({
    autosize: true,
    hovermode: 'closest',
    dragmode: 'lasso', // Default drag mode
    title: 'Select a Cluster to View Sub-network',
    xaxis: { title: 'Reduced Dimension 1' },
    yaxis: { title: 'Reduced Dimension 2' },
    margin: { l: 40, r: 40, b: 80, t: 80 },
    plot_bgcolor: '#fcfcfc',
    paper_bgcolor: '#fcfcfc',
    height: 600,
    showlegend: true
});
  const [selectedSubnetwork, setSelectedSubnetwork] = useState(null);
  const [showReductionModal, setShowReductionModal] = useState(false);
  const [showClusteringModal, setShowClusteringModal] = useState(false);
  const [selectedProfilesScatterData, setSelectedProfilesScatterData] = useState([]);

  const closeModals = () => {
    setShowReductionModal(false);
    setShowClusteringModal(false);
  };
  const [activeTab, setActiveTab] = useState("scatter");
  const [fileName, setFileName] = useState('');
  const [fileType, setFileType] = useState('');
  const [nameColumn, setNameColumn] = useState('Name');
  const [intensityStartIndex, setIntensityStartIndex] = useState(1); 
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [clusterData, setClusterData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [clusterColors, setClusterColors] = useState({});

  const [availableAlgos, setAvailableAlgos] = useState([]);
  const [availableReductions, setAvailableReductions] = useState([]);
  const [algoParams, setAlgoParams] = useState({}); 
  const [reductionParamsDef, setReductionParamsDef] = useState({}); 

  const [selectedReduction, setSelectedReduction] = useState('None');
  const [customReductionParams, setCustomReductionParams] = useState({});
  const [useDefaultReductionParams, setUseDefaultReductionParams] = useState(true);

  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [customAlgorithmParams, setCustomAlgorithmParams] = useState({});
  const [useDefaultAlgorithmParams, setUseDefaultAlgorithmParams] = useState(true);

  const [selectedProfilesData, setSelectedProfilesData] = useState([]);
  const [selectedProfilesLayout, setSelectedProfilesLayout] = useState({});
  const [selectedProfileNames, setSelectedProfileNames] = useState([]);

  const fileInputRef = useRef(null); 

  useEffect(() => {
    const fetchAlgorithms = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/algorithms');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAvailableAlgos(data.clustering_algorithms);
        setAvailableReductions(['None', ...data.dimensionality_reduction_methods]);
        setAlgoParams(data.algorithm_params);
        setReductionParamsDef(data.reduction_params);
        
        // Set initial defaults if available
        if (data.clustering_algorithms.length > 0) {
          setSelectedAlgorithm(data.clustering_algorithms[0]);
        }
      } catch (e) {
        console.error("Failed to fetch algorithms:", e);
        setError("Failed to load clustering algorithms. Is the backend running?");
      }
    };
    fetchAlgorithms();
  }, []);

  useEffect(() => {
    if (
      activeTab === "subnetwork" &&
      selectedSubnetwork !== null &&
      subnetworkData.length > 0
    ) {
      handleSubnetworkClick(selectedSubnetwork, clusterColors);
    }
  }, [topNNeighbors]);

  const handleDownloadTSV = () => {
    if (!clusterData || clusterData.length === 0) {
      alert("No cluster data to download."); 
      return;
    }

    const headers = ["Name", "Cluster", "X", "Y", "Color"];
    const tsvRows = clusterData.map(d => [
      d.name,
      d.cluster === -1 ? 'Noise (-1)' : d.cluster, 
      d.x.toFixed(6), 
      d.y.toFixed(6),
      d.color
    ]);

    const allRows = [headers.join('\t'), ...tsvRows.map(row => row.join('\t'))];
    const tsvContent = allRows.join('\n');

    const blob = new Blob([tsvContent], { type: 'text/tab-separated-values;charset=utf-8;' });

    const link = document.createElement('a');
    if (link.download !== undefined) { 
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', 'cluster_details.tsv');
      link.style.visibility = 'hidden'; 
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url); 
    } else {
      alert("Your browser does not support automatic downloads. Please copy the data manually.");
      console.error("Browser does not support download attribute.");
    }
  };
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      const ext = selectedFile.name.split('.').pop().toLowerCase();
      if (['csv', 'tsv', 'xlsx', 'xls'].includes(ext)) {
        setFileType(ext === 'xls' ? 'xlsx' : ext); 
      } else {
        setError('Unsupported file type. Please select a TSV file.');
        setFile(null);
        setFileName('');
        setFileType('');
      }
    } else {
      setFile(null);
      setFileName('');
      setFileType('');
    }
  };

  const handleReductionParamChange = (name, value, type) => {
    setCustomReductionParams(prev => ({
      ...prev,
      [name]: type === 'int' ? parseInt(value) : (type === 'float' ? parseFloat(value) : value)
    }));
    setUseDefaultReductionParams(false);
  };

  const handleAlgorithmParamChange = (name, value, type) => {
    setCustomAlgorithmParams(prev => ({
      ...prev,
      [name]: type === 'int' ? parseInt(value) : (type === 'float' ? parseFloat(value) : value)
    }));
    setUseDefaultAlgorithmParams(false);
  };

  const renderParams = (paramsDef, currentParams, setter, useDefault, selectedAlgoName) => {
    if (!paramsDef || !paramsDef[selectedAlgoName]) return null;

    const definitions = paramsDef[selectedAlgoName];
    const defaultVals = {};
    definitions.forEach(p => {
      defaultVals[p.name] = p.default;
    });

    return (
      <div className="param-grid">
        {definitions.map(param => (
          <div className="param-item" key={param.name}>
            <label htmlFor={`${selectedAlgoName}-${param.name}`}>{param.name}:</label>
            {param.type === 'int' && (
              <input
                type="number"
                id={`${selectedAlgoName}-${param.name}`}
                value={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.value, 'int')}
                min={param.min}
                max={param.max}
                step={param.step}
              />
            )}
            {param.type === 'float' && (
              <input
                type="number"
                id={`${selectedAlgoName}-${param.name}`}
                value={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.value, 'float')}
                min={param.min}
                max={param.max}
                step={param.step}
              />
            )}
            {param.type === 'boolean' && (
              <input
                type="checkbox"
                id={`${selectedAlgoName}-${param.name}`}
                checked={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.checked, 'boolean')}
              />
            )}
            {param.type === 'select' && (
              <select
                id={`${selectedAlgoName}-${param.name}`}
                value={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.value, 'string')}
              >
                {param.options.map(option => (
                  <option key={option} value={option}>{String(option)}</option>
                ))}
              </select>
            )}
            {param.nullable && (
                <div className="nullable-info">
                  (Leave blank for default `None`)
                </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setClusterData(null);
    setMetrics(null);
    setClusterColors({});

    let details = `Clustering Algorithm: ${selectedAlgorithm}\n`;
    details += `  → ${methodDescriptions.clustering[selectedAlgorithm] || 'No description.'}\n`;

    const clusterDef = algoParams[selectedAlgorithm] || [];
    const clusterLines = clusterDef.map(p => {
      const val = useDefaultAlgorithmParams
        ? p.default
        : (customAlgorithmParams[p.name] ?? p.default);
      const desc = p.description ? ` (${p.description})` : '';
      return `- ${p.name}: ${val}${desc}`;
    }).join('\n');
    details += `${clusterLines}\n\n`;

    if (selectedReduction === 'None') {
      details += `Dimensionality Reduction: None\n`;
      details += `  → ${methodDescriptions.reduction.None}\n`;
    } else {
      details += `Dimensionality Reduction: ${selectedReduction}\n`;
      details += `  → ${methodDescriptions.reduction[selectedReduction] || 'No description.'}\n`;

      const redDef = reductionParamsDef[selectedReduction] || [];
      const redLines = redDef.map(p => {
        const val = useDefaultReductionParams
          ? p.default
          : (customReductionParams[p.name] ?? p.default);
        const desc = p.description ? ` (${p.description})` : '';
        return `- ${p.name}: ${val}${desc}`;
      }).join('\n');
      details += `${redLines}`;
    }

    setMethodDetails(details);

    if (!file) {
      setError('Please select a file first.');
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('nameColumn', nameColumn);
    formData.append('intensityStartIndex', intensityStartIndex);
    formData.append('reductionMethod', selectedReduction);
    
    // Determine which parameters to send for reduction
    let finalReductionParams = {};
    if (!useDefaultReductionParams && selectedReduction !== 'None') {
        const defaultDef = reductionParamsDef[selectedReduction] || [];
        defaultDef.forEach(paramDef => {
            const val = customReductionParams[paramDef.name];
            // Handle nullable 'None' values
            if (paramDef.nullable && (val === '' || val === undefined || val === null)) {
                finalReductionParams[paramDef.name] = 'None';
            } else if (val !== undefined) {
                finalReductionParams[paramDef.name] = val;
            } else {
                finalReductionParams[paramDef.name] = paramDef.default;
            }
        });
    } else if (selectedReduction !== 'None') {
        const defaultDef = reductionParamsDef[selectedReduction] || [];
        defaultDef.forEach(paramDef => {
            finalReductionParams[paramDef.name] = paramDef.default;
        });
    }
    formData.append('reductionParams', JSON.stringify(finalReductionParams));

    let finalAlgorithmParams = {};
    if (!useDefaultAlgorithmParams && selectedAlgorithm) {
        const defaultDef = algoParams[selectedAlgorithm] || [];
        defaultDef.forEach(paramDef => {
            const val = customAlgorithmParams[paramDef.name];
            if (paramDef.nullable && (val === '' || val === undefined || val === null)) {
                finalAlgorithmParams[paramDef.name] = 'None';
            } else if (val !== undefined) {
                finalAlgorithmParams[paramDef.name] = val;
            } else {
                finalAlgorithmParams[paramDef.name] = paramDef.default;
            }
        });
    } else if (selectedAlgorithm) {
        const defaultDef = algoParams[selectedAlgorithm] || [];
        defaultDef.forEach(paramDef => {
            finalAlgorithmParams[paramDef.name] = paramDef.default;
        });
    }
    formData.append('clusteringAlgorithm', selectedAlgorithm);
    formData.append('clusteringParams', JSON.stringify(finalAlgorithmParams));
    
    console.log("Sending reduction params:", finalReductionParams);
    console.log("Sending clustering params:", finalAlgorithmParams);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/cluster', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setClusterData(data.cluster_data);
      setMetrics(data.metrics);
      setClusterColors(data.cluster_colors);
      const algoParamsUsed = (algoParams[selectedAlgorithm] || []).map(p => {
        const val = useDefaultAlgorithmParams
          ? p.default
          : (customAlgorithmParams[p.name] ?? p.default);
        const desc = paramDescriptions.clustering[p.name] || "";
        return `- ${p.name}: ${val}${desc ? ` (${desc})` : ''}`;
      }).join('\n');
      
      const redParamsUsed = selectedReduction === 'None'
      ? 'Dimensionality Reduction: Not applied'
      : `Dimensionality Reduction: ${selectedReduction}\n` + 
        (reductionParamsDef[selectedReduction] || []).map(p => {
          const val = useDefaultReductionParams
            ? p.default
            : (customReductionParams[p.name] ?? p.default);
          const desc = paramDescriptions.reduction[p.name] || "";
          return `- ${p.name}: ${val}${desc ? ` (${desc})` : ''}`;
        }).join('\n');
      
      const methodText = `Clustering Algorithm: ${selectedAlgorithm}\n${algoParamsUsed}\n\n${redParamsUsed}`;
      const methodFull = 
        `Clustering Algorithm: ${selectedAlgorithm}\n` +
        `  → ${methodDescriptions.clustering[selectedAlgorithm] || 'No description.'}\n` +
        algoParamsUsed + '\n\n' +
        redParamsUsed +
        (selectedReduction !== 'None' ? `\n  → ${methodDescriptions.reduction[selectedReduction] || 'No description.'}` : '');

      // setMethodDetails(methodFull);
      const { data: netData, layout: netLayout } = computeCentroidsAndEdges(data.cluster_data, data.cluster_colors);
      setNetworkPlotData(netData);
      setNetworkPlotLayout(netLayout);
    } catch (e) {
      console.error("Clustering failed:", e);
      setError(`Error during clustering: ${e.message}`);
    } finally {
      setLoading(false);
    }
  
  };
  
  const computeCentroidsAndEdges = (data, clusterColors) => {
    const clusters = {};
    data.forEach(d => {
      if (!clusters[d.cluster]) clusters[d.cluster] = [];
      clusters[d.cluster].push([d.x, d.y]);
    });
  
    const centroids = Object.entries(clusters).map(([cluster, points]) => {
      const meanX = points.reduce((sum, p) => sum + p[0], 0) / points.length;
      const meanY = points.reduce((sum, p) => sum + p[1], 0) / points.length;
      return { cluster: parseInt(cluster), x: meanX, y: meanY, size: points.length };
    });
  
    const maxClusterSize = Math.max(...centroids.map(c => c.size));
    const minCentroidSize = 10; // Minimum size for a centroid
    const maxCentroidSize = 30; // Maximum size for a centroid
    const getCentroidSize = (clusterSize) => {
      if (maxClusterSize === 0) return minCentroidSize;
      return minCentroidSize + (maxCentroidSize - minCentroidSize) * (clusterSize / maxClusterSize);
    };

    const edges = [];
    for (let i = 0; i < centroids.length; i++) {
      for (let j = i + 1; j < centroids.length; j++) {
        const c1 = centroids[i];
        const c2 = centroids[j];
        const dist = Math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2);
        edges.push({ source: c1, target: c2, distance: dist });
      }
    }
  
    const maxNetworkDistance = Math.max(...edges.map(e => e.distance));

    const edgeTraces = edges.map((edge, idx) => {
      const safeDistance = Math.max(edge.distance, 0.01);
      const lineWidth = Math.min(3, 1 + 2 / safeDistance); // Existing width logic
      
      const normalizedDistance = edge.distance / maxNetworkDistance;
      const opacity = Math.max(0.2, 1 - normalizedDistance * 0.8);
      const colorValue = Math.floor(255 * normalizedDistance);
      const edgeColor = `rgb(${colorValue}, ${colorValue}, ${colorValue}, ${opacity})`; // Greyscale variant, or pick another color

      return {
        type: 'scatter',
        mode: 'lines',
        x: [edge.source.x, edge.target.x],
        y: [edge.source.y, edge.target.y],
        line: {
          width: lineWidth,
          color: edgeColor, 
        },
        hoverinfo: 'text',
        text: `Cluster ${edge.source.cluster} ↔ Cluster ${edge.target.cluster}<br>Distance: ${edge.distance.toFixed(3)}`,
        showlegend: false
      };
    });
  
    const nodeTrace = {
      type: 'scatter',
      mode: 'markers+text', 
      x: centroids.map(c => c.x),
      y: centroids.map(c => c.y),
      marker: {
        size: centroids.map(c => getCentroidSize(c.size)), 
        color: centroids.map(c => clusterColors[c.cluster] || '#000'), 
        line: { width: 1, color: '#000' }
      },
      text: centroids.map(c => `Cluster ${c.cluster}`), 
      textposition: 'top center',
      hoverinfo: 'text',

      hovertext: centroids.map(c => `Cluster ${c.cluster}<br>Candidates: ${c.size}`),
      name: 'Centroids'
    };
  
    return {
      data: [...edgeTraces, nodeTrace],
      layout: {
        title: 'Network of Cluster Centroids',
        xaxis: { title: 'X', showgrid: false, zeroline: false, showticklabels: true },
        yaxis: { title: 'Y', showgrid: false, zeroline: false, showticklabels: true },
        hovermode: 'closest',
        height: 700,
        width: 1000,
        plot_bgcolor: '#fcfcfc',
        paper_bgcolor: '#fcfcfc',
      }
    };
  };

  const handleSubnetworkClick = (clusterId, clusterColors) => {
    if (!clusterData) return;
  
    const nodes = clusterData.filter(d => d.cluster === clusterId);
    const edges = [];
  
    nodes.forEach((n1, i) => {
      const distances = nodes
        .map((n2, j) => {
          if (i === j) return null;
          const dist = Math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2);
          return { source: n1, target: n2, distance: dist };
        })
        .filter(Boolean)
        .sort((a, b) => a.distance - b.distance)
        .slice(0, topNNeighbors); // <== Use user-selected value

    });

    const nodeDegrees = {};
    nodes.forEach(node => (nodeDegrees[node.name] = 0)); // Initialize all node degrees to 0

    nodes.forEach((n1, i) => {
      const distances = nodes
        .map((n2, j) => {
          if (i === j) return null;
          const dist = Math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2);
          return { source: n1, target: n2, distance: dist };
        })
        .filter(Boolean)
        .sort((a, b) => a.distance - b.distance)
        .slice(0, topNNeighbors);

      // Update degrees for connected nodes
      distances.forEach(edge => {
          edges.push(edge); // Keep this line as it's part of edge creation
          nodeDegrees[edge.source.name]++;
          nodeDegrees[edge.target.name]++;
      });
    });
    
    const maxDegree = Math.max(...Object.values(nodeDegrees));
    const minNodeSize = 8; // Minimum size for a node
    const maxNodeSize = 25; // Maximum size for a node
    const getNodeSize = (degree) => {
      if (maxDegree === 0) return minNodeSize; // Avoid division by zero if all degrees are 0
      return minNodeSize + (maxNodeSize - minNodeSize) * (degree / maxDegree);
    };

    const maxDistanceInSubnetwork = () => {
      if (edges.length === 0) return 1; 
      return Math.max(...edges.map(e => e.distance));
    };
    const dynamicMaxDist = maxDistanceInSubnetwork() * 1.2; // Add a buffer
  
    const edgeTraces = edges.map(edge => {
      const safeDistance = Math.max(edge.distance, 0.01);
      const lineWidth = Math.min(3, 1 + 2 / safeDistance);
  
      const opacity = Math.max(0.2, 1 - (edge.distance / dynamicMaxDist));
      const edgeColor = `rgba(150, 150, 150, ${opacity})`;
  
      return {
        type: 'scatter',
        mode: 'lines',
        x: [edge.source.x, edge.target.x],
        y: [edge.source.y, edge.target.y],
        line: {
          width: lineWidth,
          color: edgeColor, // Use the calculated edgeColor
        },
        hoverinfo: 'text',
        text: `${edge.source.name} ↔ ${edge.target.name}<br>Distance: ${edge.distance.toFixed(3)}`,
        showlegend: false
      };
    });


    const nodeTrace = {
      type: 'scatter',
      mode: 'markers',
      x: nodes.map(n => n.x),
      y: nodes.map(n => n.y),
      marker: {
        size: nodes.map(n => getNodeSize(nodeDegrees[n.name])), // Variable node size based on degree
        color: clusterColors[clusterId] || '#2b8cbe',// Keep uniform for now, or introduce sub-cluster colors from 'clusterColors' if needed
        line: { width: 1, color: '#ffffff' },
        opacity: 0.9
      },
      text: '',
      textposition: 'top center',
      textfont: {
        size: 10,
        color: '#111'
      },
      hovertext: nodes.map(n => `Name: ${n.name}<br>Cluster: ${n.cluster}<br>Connections: ${nodeDegrees[n.name]}`),
      hoverinfo: 'text',
      name: `Cluster ${clusterId}`
    };
  
    setSubnetworkData([...edgeTraces, nodeTrace]);
    setSubnetworkLayout({
      title: `Sub-network of Cluster ${clusterId} (Top ${topNNeighbors} Connections)`,
      height: 700,
      width: 1000,
      xaxis: { visible: false },
      yaxis: { visible: false },
      hovermode: 'closest',
      plot_bgcolor: '#fcfcfc',
      paper_bgcolor: '#fcfcfc',
      margin: { l: 20, r: 20, t: 40, b: 20 },
      dragmode: 'lasso', // or select
    });
    setSelectedSubnetwork(clusterId);
  };

  const handlePointsSelected = (points, clusterColors) => {
    if (!points || !points.points || points.points.length === 0) {
        setSelectedProfilesData([]);
        setSelectedProfilesLayout({});
        setSelectedProfileNames([]);
        return;
    }


    const selectedNames = points.points.map(p => p.hovertext.split('<br>')[0].replace('Name: ', '')); // Extract name from hovertext
    const selectedNodes = clusterData.filter(d => selectedNames.includes(d.name) && d.cluster === selectedSubnetwork);

    const linePlotTraces = [];
    const namesForCaption = [];

    selectedNodes.forEach(node => {
        if (node.intensities && node.intensities.length > 0) {
            const numDimensions = node.intensities.length;
            linePlotTraces.push({
                x: Array.from({ length: numDimensions }, (_, i) => `Dim ${i + 1}`),
                y: node.intensities,
                mode: 'lines',
                name: node.name,
                line: { color: clusterColors[node.cluster], width: 1.5 }, // Use cluster color or a consistent color
                hoverinfo: 'name+y' // Show name and intensity value on hover
            });
            namesForCaption.push(node.name);
        }
    });

    setSelectedProfilesData(linePlotTraces);
    setSelectedProfileNames(namesForCaption);
    setSelectedProfilesLayout({
        title: `Intensity Profiles of Selected Candidates from Cluster ${selectedSubnetwork}`,
        xaxis: { title: 'Dimension' },
        yaxis: { title: 'Intensity/Value' },
        hovermode: 'closest',
        height: 500,
        width: 800,
        plot_bgcolor: '#fcfcfc',
        paper_bgcolor: '#fcfcfc',
        showlegend: true // Show legend to identify individual lines
    });
};
  const plotData = clusterData ? [{
    x: clusterData.map(d => d.x),
    y: clusterData.map(d => d.y),
    mode: 'markers',
    type: 'scatter',
    marker: {
      color: clusterData.map(d => d.color),
      size: 7,
      line: {
        color: 'rgba(0, 0, 0, 0.5)',
        width: 0.5
      }
    },
    text: clusterData.map(d => `Name: ${d.name}<br>Cluster: ${d.cluster}<br>X: ${d.x.toFixed(3)}<br>Y: ${d.y.toFixed(3)}`),
    hoverinfo: 'text'
  }] : [];

  const plotLayout = {
    title: `Clustering Results (${selectedAlgorithm} with ${selectedReduction})`,
    xaxis: { title: 'Dimension 1' },
    yaxis: { title: 'Dimension 2' },
    hovermode: 'closest',
    height: 700,
    width: 1000,
    margin: { l: 50, r: 50, b: 50, t: 50 },
    plot_bgcolor: '#f5f5f5',
    paper_bgcolor: '#f5f5f5',
    shapes: [],
  };

  const downloadMethodDetails = () => {
    if (!methodDetails.trim()) {
      alert("No method details available to download.");
      return;
    }
  
    const blob = new Blob([methodDetails], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `method_details_${selectedAlgorithm}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // --- Helper to clear file input ---
  const clearFileInput = () => {
    setFile(null);
    setFileName('');
    setFileType('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
        <a
          href="https://www.rappsilberlab.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img
            src={require('./RapLabTextLogo.png')}
            alt="RapLab Logo"
            style={{ height: '90px', marginRight: '30px' }}
          />
        </a>
        <img
          src={require('./logo_rapcluster.svg').default}
          alt="RapCluster Logo"
          style={{ height: '90px', marginRight: '40px' }}
        />
        </div>
          <h1 style={{ flexGrow: 1, textAlign: 'center', margin: 0 }}>Explore, Tune, and Compare Clustering Methods at Scale</h1>
          <div style={{ width: '70px' }} /> {/* filler to balance the logo space */}
        </div>
      </header>
      <main className="App-main">
        <form onSubmit={handleSubmit} className="clustering-form">
          <section className="input-section">
            <h2>1. Upload Data</h2>
            <div className="form-group">
              <label htmlFor="file-upload">Select File (TSV):</label>
              <input 
                type="file" 
                id="file-upload" 
                accept=".csv,.tsv,.xlsx,.xls" 
                onChange={handleFileChange} 
                ref={fileInputRef}
              />
              {fileName && <span className="file-name">{fileName} <button type="button" onClick={clearFileInput}>x</button></span>}
            </div>

            <div className="form-group">
              <label htmlFor="name-column">Node Name Column:</label>
              <input
                type="text"
                id="name-column"
                value={nameColumn}
                onChange={(e) => setNameColumn(e.target.value)}
                placeholder="e.g., GeneName"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="intensity-start-index">Data Start Column Index (0-based):</label>
              <input
                type="number"
                id="intensity-start-index"
                value={intensityStartIndex}
                onChange={(e) => setIntensityStartIndex(parseInt(e.target.value))}
                min="0"
                required
              />
            </div>
          </section>

          <section className="reduction-section">
            <h2>2. Dimensionality Reduction</h2>
             <p style={{ marginTop: '-10px' }}>
                <button
                  type="button"
                  onClick={() => setShowReductionModal(true)}
                  style={{
                    fontSize: '0.9em',
                    background: 'none',
                    color: '#007bff',
                    border: 'none',
                    cursor: 'pointer',
                    paddingLeft: '0'
                  }}
                >
                  Learn More
                </button>
              </p>
            <div className="form-group">
              <label htmlFor="reduction-method">Method:</label>
              <select
                id="reduction-method"
                value={selectedReduction}
                onChange={(e) => {
                  setSelectedReduction(e.target.value);
                  setUseDefaultReductionParams(true); // Reset to default when method changes
                  setCustomReductionParams({});
                }}
              >
                {availableReductions.map(method => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </div>

            {selectedReduction !== 'None' && (
              <div className="param-control">
                <label>
                  <input
                    type="checkbox"
                    checked={useDefaultReductionParams}
                    onChange={(e) => {
                        setUseDefaultReductionParams(e.target.checked);
                        if(e.target.checked) setCustomReductionParams({}); // Clear custom if defaulting
                    }}
                  />
                  Use Default Parameters
                </label>
                {!useDefaultReductionParams && (
                  <div className="param-custom-settings">
                    <h3>Custom {selectedReduction} Parameters</h3>
                    {renderParams(reductionParamsDef, customReductionParams, handleReductionParamChange, false, selectedReduction)}
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="clustering-section">
            <h2>3. Clustering Algorithm</h2>
            <p style={{ marginTop: '-10px' }}>
              <button
                type="button"
                onClick={() => setShowClusteringModal(true)}
                style={{
                  fontSize: '0.9em',
                  background: 'none',
                  color: '#007bff',
                  border: 'none',
                  cursor: 'pointer',
                  paddingLeft: '0'
                }}
              >
                Learn More
              </button>
            </p>
            <div className="form-group">
              <label htmlFor="clustering-algorithm">Algorithm:</label>
              <select
                id="clustering-algorithm"
                value={selectedAlgorithm}
                onChange={(e) => {
                  setSelectedAlgorithm(e.target.value);
                  setUseDefaultAlgorithmParams(true); // Reset to default when algorithm changes
                  setCustomAlgorithmParams({});
                }}
                required
              >
                {availableAlgos.map(algo => (
                  <option key={algo} value={algo}>{algo}</option>
                ))}
              </select>
            </div>

            {selectedAlgorithm && (
              <div className="param-control">
                <label>
                  <input
                    type="checkbox"
                    checked={useDefaultAlgorithmParams}
                    onChange={(e) => {
                        setUseDefaultAlgorithmParams(e.target.checked);
                        if(e.target.checked) setCustomAlgorithmParams({}); // Clear custom if defaulting
                    }}
                  />
                  Use Default Parameters
                </label>
                {!useDefaultAlgorithmParams && (
                  <div className="param-custom-settings">
                    <h3>Custom {selectedAlgorithm} Parameters</h3>
                    {renderParams(algoParams, customAlgorithmParams, handleAlgorithmParamChange, false, selectedAlgorithm)}
                  </div>
                )}
              </div>
            )}
          </section>

          <button type="submit" className="submit-button" disabled={loading || !file}>
            {loading ? 'Processing...' : 'Run Clustering'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {metrics && clusterData && (
          <section className="results-section">
            <h2>Clustering Results</h2>
            <div className="metrics-summary">
              <h3>Metrics:</h3>
              <p><strong>Clusters Found:</strong> {metrics.n_clusters}</p>
              <p><strong>Silhouette Score:</strong> {metrics.silhouette_score.toFixed(3)} (Higher is better)</p>
              <p><strong>Calinski-Harabasz Score:</strong> {metrics.calinski_harabasz_score.toFixed(3)} (Higher is better)</p>
              <p><strong>Davies-Bouldin Score:</strong> {metrics.davies_bouldin_score.toFixed(3)} (Lower is better)</p>
              <p><strong>Runtime:</strong> {metrics.runtime_seconds.toFixed(2)} seconds</p>
            </div>

            <div className="visualization">
              <h3>Visualization:</h3>

              <div className="tab-buttons">
                <button
                  onClick={() => setActiveTab("scatter")}
                  className={activeTab === "scatter" ? "active" : ""}
                >
                  Scatter Plot
                </button>
                <button
                  onClick={() => setActiveTab("network")}
                  className={activeTab === "network" ? "active" : ""}
                >
                  Clusters Network
                </button>
                <button
                  onClick={() => setActiveTab("subnetwork")}
                  className={activeTab === "subnetwork" ? "active" : ""}
                  disabled={!subnetworkData.length}
                >
                  Sub-network View
                </button>
              </div>

              {activeTab === "scatter" && (
                <>
                  <Plot
                    data={plotData}
                    layout={plotLayout}
                    config={{
                      responsive: true,
                      displayModeBar: true,
                      toImageButtonOptions: {
                        format: "svg",
                        filename: "clustering_plot",
                        scale: 1,
                      },
                    }}
                  />
                  <div className="plot-caption">
                    <h4>Caption Template:</h4>
                    <p>
                      Clustering results using <strong>{selectedAlgorithm}</strong>{" "}
                      {useDefaultAlgorithmParams
                        ? "(default parameters)"
                        : `with custom parameters: ${JSON.stringify(customAlgorithmParams)}`}.{" "}
                      Dimensionality reduction: <strong>{selectedReduction}</strong>{" "}
                      {selectedReduction !== "None"
                        ? useDefaultReductionParams
                          ? "(default parameters)"
                          : `with custom parameters: ${JSON.stringify(
                              customReductionParams
                            )}`
                        : ""}. Evaluation scores: Silhouette:{" "}
                      <strong>{metrics.silhouette_score.toFixed(3)}</strong>, CH:{" "}
                      <strong>{metrics.calinski_harabasz_score.toFixed(3)}</strong>, Davies-Bouldin:{" "}
                      <strong>{metrics.davies_bouldin_score.toFixed(3)}</strong>.
                    </p>
                  </div>
                </>
              )}

              {activeTab === "network" && (
                <>
                  <Plot
                    data={networkPlotData}
                    layout={networkPlotLayout}
                    config={{
                      responsive: true,
                      displayModeBar: true,
                      toImageButtonOptions: {
                        format: "svg",
                        filename: "network_plot",
                        scale: 1,
                      },
                    }}
                    onClick={(e) => {
                      const clicked = e.points[0];
                      if (clicked?.text?.startsWith("Cluster")) {
                        const clusterId = parseInt(clicked.text.replace("Cluster ", ""));
                        handleSubnetworkClick(clusterId, clusterColors);
                        setActiveTab("subnetwork");
                      }
                    }}
                  />

                  <div className="plot-caption">
                    <h4>Caption Template:</h4>
                    <p>
                      Network of cluster centroids generated from{" "}
                      <strong>{selectedAlgorithm}</strong>. Each edge represents inverse
                      Euclidean distance in 2D space derived via{" "}
                      <strong>{selectedReduction}</strong>. Total clusters:{" "}
                      <strong>{metrics.n_clusters}</strong>.
                    </p>
                  </div>
                </>
              )}
              {activeTab === "subnetwork" && (
                <section>
                    <h2>Sub-network View</h2>
                    <div style={{ marginBottom: '10px' }}>
                        <button
                            onClick={() => setSubnetworkLayout(prev => ({ ...prev, dragmode: 'lasso' }))}
                            style={{
                                marginRight: '10px',
                                padding: '8px 15px',
                                cursor: 'pointer',
                                backgroundColor: subnetworkLayout.dragmode === 'lasso' ? '#28a745' : '#e8e8e8',
                                color: subnetworkLayout.dragmode === 'lasso' ? 'white' : '#333',
                                border: '1px solid #ccc',
                                borderRadius: '5px',
                                transition: 'background-color 0.15s ease-in-out'
                            }}
                        >
                            Select (Lasso)
                        </button>
                        <button
                            onClick={() => setSubnetworkLayout(prev => ({ ...prev, dragmode: 'pan' }))}
                            style={{
                                padding: '8px 15px',
                                cursor: 'pointer',
                                backgroundColor: subnetworkLayout.dragmode === 'pan' ? '#28a745' : '#e8e8e8',
                                color: subnetworkLayout.dragmode === 'pan' ? 'white' : '#333',
                                border: '1px solid #ccc',
                                borderRadius: '5px',
                                transition: 'background-color 0.15s ease-in-out'
                            }}
                        >
                            Pan/Zoom
                        </button>
                    </div>

                    {}
                    <>
                        <div style={{ marginBottom: '10px', textAlign: 'center' }}>
                            <label>
                                Show top&nbsp;
                                <select
                                    value={topNNeighbors}
                                    onChange={(e) => setTopNNeighbors(parseInt(e.target.value))}
                                >
                                    {[3, 5, 10, 15, 20].map(n => (
                                        <option key={n} value={n}>{n}</option>
                                    ))}
                                </select>
                                &nbsp;nearest connections
                            </label>
                        </div>
                        <Plot
                            data={subnetworkData}
                            layout={subnetworkLayout}
                            onSelected={(points) => handlePointsSelected(points, clusterColors)}
                            config={{
                                responsive: true,
                                displayModeBar: true,
                                toImageButtonOptions: {
                                    format: "svg",
                                    filename: `subnetwork_cluster_${selectedSubnetwork}`,
                                    scale: 1,
                                },
                            }}
                        />
                        <div className="plot-caption">
                            <h4>Caption Template:</h4>
                            <p>
                                Intra-cluster network for <strong>Cluster {selectedSubnetwork}</strong>{" "}
                                from <strong>{selectedAlgorithm}</strong>. Nodes are individual <strong>{nameColumn}</strong> values. Edges are drawn between a node and its top N nearest neighbors (default = 5 or user-selected)
                                and represent the applied proximity in 2D projection (<strong>{selectedReduction}</strong>).
                            </p>
                        </div>

                        {selectedProfilesData.length > 0 && (
                            <div style={{ marginTop: '30px' }}>
                                <Plot
                                    data={selectedProfilesData}
                                    layout={selectedProfilesLayout}
                                    config={{
                                        responsive: true,
                                        displayModeBar: true,
                                        toImageButtonOptions: {
                                            format: "svg",
                                            filename: `selected_profiles_${selectedSubnetwork}`,
                                            scale: 1,
                                        },
                                    }}
                                />
                                <div className="plot-caption">
                                    <h4>Selected Profiles:</h4>
                                    <p>
                                        Log10 of the raw profiles for: <strong>{selectedProfileNames.join(', ')}</strong>.
                                    </p>
                                </div>
                            </div>
                        )}
                        <div style={{ textAlign: 'center', marginTop: '10px' }}>
                            <button onClick={() => setSubnetworkData([])}>Clear Subnetwork</button>
                        </div>
                    </>
                </section>
            )}
            
            </div>

            <div style={{
                backgroundColor: '#f0f8ff',
                padding: '20px',
                borderRadius: '10px',
                marginTop: '20px',
                marginBottom: '20px'
              }}>
                <h3 style={{ color: '#6a1b9a', marginBottom: '10px' }}>Method Details:</h3>
                <pre style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>{methodDetails}</pre>
                <button onClick={downloadMethodDetails} style={{
                  marginTop: '10px',
                  padding: '8px 12px',
                  backgroundColor: '#555',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}>
                  Download Method Details
                </button>
              </div>

            <div className="cluster-details">
                <h3>Cluster Details:</h3>
                <button
                  onClick={handleDownloadTSV}
                  className="download-button" // Add a class for styling
                  disabled={!clusterData || clusterData.length === 0}
                >
                  Download TSV
                </button>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Cluster</th>
                            <th>X</th>
                            <th>Y</th>
                            <th style={{width: '20px'}}>Color</th>
                        </tr>
                    </thead>
                    <tbody>
                        {clusterData.map((d, index) => (
                            <tr key={index}>
                                <td>{d.name}</td>
                                <td>{d.cluster === -1 ? 'Noise (-1)' : d.cluster}</td>
                                <td>{d.x.toFixed(3)}</td>
                                <td>{d.y.toFixed(3)}</td>
                                <td style={{ backgroundColor: d.color, border: '1px solid #ccc' }}></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
          </section>
        )}
      </main>
      <Modal
          isOpen={showReductionModal}
          onRequestClose={() => setShowReductionModal(false)}
          contentLabel="Dimensionality Reduction Info"
          className="guide-modal"
          style={{ content: { maxWidth: '700px', margin: 'auto' } }}
        >
          <AlgorithmGuideDR />
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <button
              onClick={() => setShowReductionModal(false)}
              style={{
                padding: '8px 16px',
                backgroundColor: '#6a1b9a',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer'
              }}
            >
              Close
            </button>
          </div>
        </Modal>

        <Modal
          isOpen={showClusteringModal}
          onRequestClose={() => setShowClusteringModal(false)}
          contentLabel="Clustering Algorithms Info"
          className="guide-modal"
          style={{ content: { maxWidth: '700px', margin: 'auto' } }}
        >
          <AlgorithmGuideClustering />
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <button
              onClick={() => setShowClusteringModal(false)}
              style={{
                padding: '8px 16px',
                backgroundColor: '#6a1b9a',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer'
              }}
            >
              Close
            </button>
          </div>
        </Modal>
    </div>
  );
}

export default App;