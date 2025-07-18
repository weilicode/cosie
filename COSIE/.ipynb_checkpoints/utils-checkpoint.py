import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import anndata as ad
import math
import scipy
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import random
import torch
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from scipy.spatial import distance_matrix

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 


from annoy import AnnoyIndex
import hnswlib





def nn_approx(ds1, ds2, knn=10, metric='euclidean', n_trees=10, include_distances=False):

    """
    Efficiently find approximate K-nearest neighbors using the Annoy library.

    Parameters
    ----------
    ds1 : np.ndarray
        Query data of shape (n_query, dim), where neighbors will be searched for.
    
    ds2 : np.ndarray
        Reference data of shape (n_ref, dim), in which the neighbors are searched.
    
    knn : int, optional
        Number of neighbors to retrieve per query. Default is 10.
    
    metric : str, optional
        Distance metric used in Annoy. Must be one of: {'euclidean', 'manhattan', 'angular', 'hamming', 'dot'}. Default is 'euclidean'.
    
    n_trees : int, optional
        Number of trees used to build the Annoy index. Higher values increase accuracy at the cost of indexing time. Default is 10.
    
    include_distances : bool, optional
        Whether to also return distances to the nearest neighbors. Default is False.

    Returns
    -------
    ind : np.ndarray
        If `include_distances` is False, returns an array of shape (n_query, knn) with indices of nearest neighbors.

    tuple of (ind, dist) : (np.ndarray, np.ndarray)
        If `include_distances` is True, returns a tuple:
        
        - `ind` : array of shape (n_query, knn) with indices of nearest neighbors.
        - `dist` : array of shape (n_query, knn) with corresponding distances.
    """


    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind, dist = [], []
    for i in range(ds1.shape[0]):
        i_ind, i_dist = a.get_nns_by_vector(ds1[i, :], knn, search_k=-1, include_distances=True)
        ind.append(i_ind)
        dist.append(i_dist)
    ind = np.array(ind)
    
    if include_distances:
        return ind, np.array(dist)
    else:
        # return ind.flatten()
        return ind


def setup_seed(seed=8, mode='fast'):

    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Parameters
    ----------
    seed : int, optional
        The random seed to be set for all relevant libraries. Default is 8.
    mode : str, optional
        Controls how strictly reproducibility is enforced. Must be one of {'fast', 'strict'}.
        
        - 'fast': Ensures reproducibility in most cases, with minimal performance impact.  
        - 'strict': Enforces full determinism across all operations (including CUDA),
          but may significantly slow down certain models.

        It is recommended to use 'strict' only when exact reproducibility across runs is required.

    Returns
    -------
    None
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # control cuDNN 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # strict mode
    if mode == 'strict':
        ### This will slow down the learning process
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'





def compute_knn_graph(input_data, n_neighbors):

    """
    Construct a k-nearest neighbors (k-NN) graph based on an input feature matrix.

    Parameters
    ----------
    input_data : np.ndarray
        An array of shape (n_cells, n_features) representing input features of cells,
        which can be spatial coordinates or feature vectors.
    
    n_neighbors : int
        Number of nearest neighbors to connect for each node.

    Returns
    -------
    edge_index : torch.LongTensor
        A tensor of shape (2, num_edges) representing the edges of the graph. Each column (i, j) represents an edge from node i to its j-th nearest neighbor.
    """

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(input_data)  
    _ , indices = nbrs.kneighbors(input_data)
    x = indices[:, 0].repeat(n_neighbors+1)
    y = indices[:, 0:].flatten() 
    edge_index = np.vstack((x, y))    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index






def construct_knn_graph_hnsw(data, k=20, space='l2'):

    """
    Efficiently compute approximate k-nearest neighbor (k-NN) graph using the hnswlib library.
    This method is suitable for large-scale datasets.

    Parameters
    ----------
    data : torch.Tensor
        A tensor of shape (n_cells, n_features) representing the input feature matrix.
    
    k : int, optional
        Number of nearest neighbors to retrieve for each sample. Default is 20.
    
    space : str, optional
        Distance metric used to build the index. Must be one of {'l2', 'ip', 'cosine'}.
        
        - 'l2': Euclidean distance  
        - 'ip': Inner product  
        - 'cosine': Cosine similarity  

        Default is 'l2'.

    Returns
    -------
    edge_index : torch.LongTensor
        A tensor of shape (2, n_edges) representing the edges of the graph.
        Each column represents an edge from source node to target node.
    """

    random.seed(42)
    np.random.seed(42)
    data = data.cpu().numpy().astype(np.float32)
    num_samples, dim = data.shape

    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=200, M=16)
    p.add_items(data)

    p.set_ef(50)

    indices, distances = p.knn_query(data, k=k)
    
    row_indices = np.repeat(np.arange(num_samples), k)
    col_indices = indices.flatten()
    edge_index = torch.tensor(np.vstack((row_indices, col_indices)), dtype=torch.long)
    
    return edge_index



def compute_neighborhood_embedding(edge_index_spatial: torch.Tensor, embedding_matrix: torch.Tensor, device) -> torch.Tensor:

    """
    Compute the neighborhood embedding of each cell based on its spatial k-nearest neighbor graph.

    Parameters
    ----------
    edge_index_spatial : torch.Tensor
        A tensor of shape (2, num_edges), where each column represents a directed edge from source node to target node in the spatial graph.
    
    embedding_matrix : torch.Tensor
        A tensor of shape (n_cells, dim) representing the embedding of each cell.
    
    device : torch.device
        The computation device, e.g., `torch.device('cuda')` or `torch.device('cpu')`.

    Returns
    -------
    neighborhood_embedding_matrix : torch.Tensor
        A tensor of shape (n_cells, dim) representing the average of each cell’s neighbors' embeddings.
    """

    
    # Get the number of nodes and embedding dimension
    num_nodes = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    
    # Initialize the neighborhood embedding matrix and count tensor
    neighborhood_embedding_matrix = torch.zeros(num_nodes, embedding_dim, device=device)
    neighbor_count = torch.zeros(num_nodes, dtype=torch.float32, device=device)

    # Use edge_index_spatial to index into the embedding matrix
    source_nodes = edge_index_spatial[0]  # Get source nodes
    target_nodes = edge_index_spatial[1]  # Get target nodes

    # Accumulate the embeddings from target nodes to the corresponding source nodes
    neighborhood_embedding_matrix.index_add_(0, source_nodes, embedding_matrix[target_nodes])
    
    # Count neighbors for each source node
    neighbor_count.index_add_(0, source_nodes, torch.ones_like(target_nodes, dtype=torch.float32, device=device))

    # Avoid division by zero by replacing zero counts with one
    neighbor_count[neighbor_count == 0] = 1

    # Compute the average by dividing by the count of neighbors
    neighborhood_embedding_matrix /= neighbor_count.view(-1, 1)
    

    return neighborhood_embedding_matrix

    





##