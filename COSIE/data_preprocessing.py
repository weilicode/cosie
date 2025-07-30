import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import anndata as ad
import json
import torch
from collections import defaultdict



def preprocess_adata(adata_raw, modality, hvg_num=3000, n_comps=50, target_sum=None):
    """
    Preprocess an AnnData object based on the specified modality. The pipeline includes highly variable feature selection, normalization, log-transformation, scaling, and PCA.

    This function supports preprocessing of epigenomic, RNA, protein, metabolite, and histology embedding (HE) modalities.

    Parameters
    ----------
    adata_raw : AnnData
        The raw AnnData object to be processed.
    
    modality : str
        The modality type. Must be one of:
        
        - `'RNA'`, `'RNA_panel2'`: RNA count matrix, supports different panels within the same RNA modality.
        - `'H3K27me3'`, `'H3K27ac'`, `'ATAC'`, `'H3K4me3'`: Epigenomic signals.  
          We recommend first converting raw epigenomics data to gene scores before using this function.
          Gene score generation scripts are available at spatial-Mux-seq Repository.  
        - `'Protein'`: Protein abundance matrix; CLR normalization will be applied. For COMET protein data, we recommend using arcsinh normalization.
        - `'Metabolite'`: Metabolite expression matrix.  
        - `'HE'`: Histology image embeddings; PCA will be applied directly without normalization.
    
    hvg_num : int, optional
        Number of highly variable features to select. If None, HVG selection is skipped. Default is 3000.
    
    n_comps : int, optional
        Number of PCA components to compute. Default is 50.
    
    target_sum : float or None, optional
        Target sum for total-count normalization (used in `normalize_total`). If None, the Scanpy default is used. Default is None.

    Returns
    -------
    adata : AnnData
        A preprocessed AnnData object with normalized, log-transformed, scaled, and PCA-reduced `.X`. For protein modality, CLR normalization is used. HVG selection is only applied if `hvg_num` is provided and the number of input features exceeds this threshold.
    """

    adata = adata_raw.copy()
    adata.var_names_make_unique()
    # print(f'Processing data... Modality: {modality}')

    if modality == 'HE':
        sc.tl.pca(adata, n_comps=n_comps)
    else:
        if hvg_num and len(adata.var_names)>hvg_num:
            if modality in {"RNA", "RNA_panel2","H3K27me3", "H3K27ac", "ATAC", "H3K4me3", "Metabolite"}:
                use_batch = 'batch' in adata.obs
                # print(f'Selecting HVG for {modality} {"with batch key" if use_batch else "without batch key"}')
                if modality in ["RNA", "RNA_panel2"]:
                    sc.pp.highly_variable_genes(adata, n_top_genes=hvg_num, flavor="seurat_v3", batch_key='batch' if use_batch else None)
                else:
                    sc.pp.highly_variable_genes(adata, n_top_genes=hvg_num, batch_key='batch' if use_batch else None)
                adata = adata[:, adata.var['highly_variable']]

        if modality == "Protein":
            adata = clr_normalize_each_cell(adata)  
            sc.pp.scale(adata)
            # adata.X = np.arcsinh(adata.X / 5)   ### For COMET data
            # print('using arcsinh')
            n_proteins = adata.shape[1]

            if n_proteins >= n_comps:
                sc.tl.pca(adata, n_comps=n_comps)
            elif n_proteins >= 20:
                sc.tl.pca(adata, n_comps=20)
            else:
                sc.tl.pca(adata, n_comps=15)
        else:
            if target_sum:
                sc.pp.normalize_total(adata, target_sum=target_sum)
            else:
                sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.scale(adata)
            sc.tl.pca(adata, n_comps=n_comps)

    return adata



def load_data(data_dict, n_comps=50, hvg_num=3000, target_sum=None, use_harmony=True, metacell = False):

    """
    Process input spatial multi-modal data, returning processed feature matrices and spatial coordinates.
    
    Shared modalities that appear in multiple sections are concatenated and jointly processed.
    Unique modalities (only present in one section) are processed independently. Each section's feature matrix is stored
    as a PyTorch tensor for downstream modeling. Spatial coordinates are checked for consistency across modalities; if
    inconsistent within a section, an error is raised.

    Parameters
    ----------
    data_dict : dict
        A dictionary mapping each modality name (e.g., `'RNA'`, `'Protein'`) to a list of AnnData objects, one per tissue section. Each AnnData should contain `.X`, `.obs`, `.var`, and `.obsm['spatial']`. If a modality is missing from a section, use `None` as a placeholder in the list.
    
    n_comps : int, optional
        Number of PCA components to compute. Default is 50.
    
    hvg_num : int, optional
        Number of highly variable features to select. If the feature dimension is smaller than `hvg_num`, HVG selection is skipped. Default is 3000.
    
    target_sum : float or None, optional
        Target sum for total-count normalization (used in `scanpy.pp.normalize_total`). If None, Scanpy default is used. Default is None.
    
    use_harmony : bool, optional
        Whether to perform Harmony integration across sections for shared modalities. If False, only joint PCA is applied. Default is True.
    
    metacell : bool, optional
        Whether to merge each 2×2 spatial grid of cells into a "metacell" for reducing memory usage and improving speed. Applies to all modalities. Default is False.

    Returns
    -------
    feature_dict : dict
        A dictionary mapping each section name (e.g., `'s1'`, `'s2'`) to a sub-dictionary of processed feature tensors for each modality. Each feature is a `torch.FloatTensor` of shape (n_cells, n_comps).
    
    spatial_loc_dict : dict
        A dictionary mapping each section name to a 2D NumPy array of spatial coordinates, extracted from `.obsm['spatial']`. Shape is (n_cells, 2).
    
    data_dict : dict
        The updated input dictionary. Each AnnData object is modified to include reduced features (e.g., PCA or Harmony output) in `.obsm`.
    """

    if metacell:
        # This will combine every 2x2 adjacent cells into a meta-cell across all modalities.
        print('Combine adjacent 4 cells into metacell to save memory and speed up computation')
        data_dict = construct_metacell_data_dict(data_dict)
        

    feature_dict = {}   
    spatial_loc_dict = {}  
    num_sections = max(len(sections) for sections in data_dict.values()) 

    # Detect shared modality
    shared_modalities = {modality: [adata for adata in sections if adata is not None] 
                         for modality, sections in data_dict.items() 
                         if sum(x is not None for x in sections) > 1}
    
    # print('Shared modalities:', shared_modalities)
    shared_modality_sections = {modality: [idx for idx, adata in enumerate(data_dict[modality]) if adata is not None]
                                for modality in shared_modalities}
    
    # Process shared modality
    for modality, adata_list in shared_modalities.items():
        print(f'-------- Processing shared modality {modality} across sections --------')

        if modality == 'HE':
            adata_sub_list = []
            for i, adata in enumerate(adata_list):
                adata_sub = adata.copy()
                adata_sub.obs_names = adata_sub.obs_names + f"_{shared_modality_sections[modality][i]}"
                adata_sub_list.append(adata_sub)

        else:
            common_var_names = adata_list[0].var_names
            for adata in adata_list[1:]:
                common_var_names = common_var_names.intersection(adata.var_names)
            
            adata_sub_list = []
            for i, adata in enumerate(adata_list):
                adata_sub = adata[:, common_var_names].copy()
                adata_sub.obs_names = adata_sub.obs_names + f"_{shared_modality_sections[modality][i]}"
                adata_sub_list.append(adata_sub)


        adata_combined = ad.concat(adata_sub_list)
        adata_combined.obs['batch'] = [f'batch_{shared_modality_sections[modality][i]}' 
                                       for i, adata in enumerate(adata_list) for _ in range(adata.shape[0])]

        
        adata_combined = preprocess_adata(adata_combined, modality, hvg_num=hvg_num, n_comps=n_comps)
        if use_harmony:
            print(f"Running Harmony for {modality}")
            sc.external.pp.harmony_integrate(adata_combined, key='batch')
            pca_data_combined = adata_combined.obsm['X_pca_harmony']  
        else:
            pca_data_combined = adata_combined.obsm['X_pca']
        

        # split back to each section
        split_indices = np.cumsum([adata.shape[0] for adata in adata_list])[:-1]
        combined_data_splits = np.split(pca_data_combined, split_indices)


        for i, section in enumerate(shared_modality_sections[modality]):
            key_name = f'{modality}_harmony' if use_harmony else f'{modality}_pca'
            data_dict[modality][section].obsm[key_name] = combined_data_splits[i]
            if section not in feature_dict:
                feature_dict[section] = {}

            shared_data = combined_data_splits[i].copy()
            feature_dict[section][modality] = torch.from_numpy(shared_data).float()
            del shared_data


    # Process unique modality
    for modality, sections in data_dict.items():
        if modality in shared_modalities:
            continue  

        for section, adata in enumerate(sections):
            if adata is not None:
                print(f'-------- Processing unique modality {modality} for section {section+1} --------')
                if section not in feature_dict:
                    feature_dict[section] = {}
                adata_processed = preprocess_adata(adata, modality, hvg_num=hvg_num, n_comps=n_comps, target_sum=target_sum)
                pca_data = adata_processed.obsm['X_pca'].copy()
                data_dict[modality][section].obsm[f'{modality}_pca'] = pca_data
                feature_dict[section][modality] = torch.from_numpy(pca_data).float()
                del pca_data
    feature_dict = {f's{int(k) + 1}': v for k, v in feature_dict.items()}

    # Process spatial location
    for section_idx in range(num_sections): 
        print(f'Extracting spatial location for section {section_idx+1}')
        spatial_list = []
        for modality, sections in data_dict.items():
            if section_idx < len(sections) and sections[section_idx] is not None and 'spatial' in sections[section_idx].obsm:
                spatial_list.append(sections[section_idx].obsm['spatial'])

        if len(spatial_list) == 1:
            spatial_loc_dict[f's{section_idx+1}'] = spatial_list[0]
        elif len(spatial_list) > 1:
            if all(np.array_equal(spatial_list[0], spatial) for spatial in spatial_list[1:]):
                spatial_loc_dict[f's{section_idx+1}'] = spatial_list[0]
            else:
                raise ValueError(f"Section {section_idx+1} contains inconsistent spatial information across different modalities!")


    return feature_dict, spatial_loc_dict, data_dict




### CLR normalization for protein borrowed from SpatialGLUE
def clr_normalize_each_cell(adata, inplace=True):
    """
    Normalize each cell's protein counts using Centered Log-Ratio (CLR) normalization,
    following the approach used in Seurat and SpatialGLUE.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object where `.X` stores raw count data (e.g., protein abundance).
    
    inplace : bool, optional
        Whether to modify the input `adata` in place. If True, the normalization will overwrite `adata.X`. If False, a normalized copy of `adata` is returned. Default is True.

    Returns
    -------
    adata : AnnData
        The AnnData object with CLR-normalized `.X`. If `inplace=True`, returns the modified input object; if `inplace=False`, returns a new normalized copy.
    """
    
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata  



def metacell_construction_optimized(adata):
    """
    Construct metacells by aggregating every 2×2 spatially adjacent grid of cells into one,
    using integer grid grouping for O(n) performance.

    Parameters
    ----------
    adata : AnnData
        An AnnData object with .obsm['spatial'] coords and .X expr. matrix.

    Returns
    -------
    adata_metacell : AnnData
        New AnnData of metacells with averaged expression and coords, plus mapping.
    """
    import scipy.sparse as sp
    # Extract coords and expression
    spatial = adata.obsm['spatial']
    expr = adata.X
    # handle sparse
    if sp.issparse(expr):
        expr = expr.toarray()

    # Compute grid origin and step sizes
    y = spatial[:, 0]; x = spatial[:, 1]
    y0, x0 = y.min(), x.min()
    uniq_y = np.unique(y); uniq_x = np.unique(x)
    dy = np.diff(uniq_y)
    dx = np.diff(uniq_x)
    # choose smallest non-zero step
    step_y = np.min(dy[dy > 0])
    step_x = np.min(dx[dx > 0])

    # Map to integer grid
    grid_y = np.round((y - y0) / step_y).astype(int)
    grid_x = np.round((x - x0) / step_x).astype(int)

    # Block coordinates for 2x2 grouping
    block_y = grid_y // 2
    block_x = grid_x // 2

    # Group indices by block
    from collections import defaultdict
    blocks = defaultdict(list)
    for idx, (by, bx) in enumerate(zip(block_y, block_x)):
        blocks[(by, bx)].append(idx)

    # Aggregate each block
    meta_expr = []
    meta_coords = []
    meta_to_original = []
    for (by, bx), indices in blocks.items():
        meta_to_original.append(indices)
        # mean expression and coords
        meta_expr.append(expr[indices].mean(axis=0))
        meta_coords.append(spatial[indices].mean(axis=0))

    # Build new AnnData
    meta_X = np.vstack(meta_expr)
    adata_meta = sc.AnnData(X=meta_X)
    adata_meta.var_names = adata.var_names.copy()
    adata_meta.obsm['spatial'] = np.vstack(meta_coords)
    adata_meta.uns['meta_to_original'] = meta_to_original
    adata_meta.uns['original_cell_num'] = adata.n_obs

    return adata_meta




# def metacell_construction(adata):
#     """
#     Construct metacells by aggregating every 2×2 spatially adjacent grid of cells into one,
#     to reduce memory usage and speed up computation.

#     Parameters
#     ----------
#     adata : AnnData
#         An AnnData object. The spatial coordinates should be stored in `adata.obsm['spatial']`. The expression matrix `adata.X` can be either dense or sparse.

#     Returns
#     -------
#     adata_metacell : AnnData
#         A new AnnData object where each observation (cell) corresponds to a metacell,
#         formed by averaging a group of up to 4 adjacent spatial cells (in a 2×2 pattern).

#         The returned object includes:
        
#         - `.X`: Averaged expression matrix across grouped cells.  
#         - `.obsm['spatial']`: Spatial coordinates (mean of each group).  
#         - `.uns['meta_to_original']`: A list mapping each metacell to the indices of original cells it includes.  
#         - `.uns['original_cell_num']`: Total number of original cells before metacell construction.
#     """
#     spatial_coords = adata.obsm['spatial']  
#     expression = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

#     # sort (y, x)
#     sorted_indices = np.lexsort((spatial_coords[:, 0], spatial_coords[:, 1]))
#     sorted_coords = spatial_coords[sorted_indices]
#     sorted_expression = expression[sorted_indices]

#     meta_cells = []
#     meta_coords = []
#     meta_to_original = []  
#     visited = set()

#     # visit each cell
#     for i in range(len(sorted_coords)):
#         if i in visited:
#             continue

#         # current cell
#         group_indices = [i]
#         visited.add(i)

#         # neighbors
#         right = None
#         down = None
#         diag = None

#         for j in range(i + 1, len(sorted_coords)):
#             if j in visited:
#                 continue
#             dx = sorted_coords[j][0] - sorted_coords[i][0]
#             dy = sorted_coords[j][1] - sorted_coords[i][1]

#             if right is None and dx > 0 and dy == 0: 
#                 right = j
#                 visited.add(j)
#             elif down is None and dx == 0 and dy > 0: 
#                 down = j
#                 visited.add(j)
#             elif diag is None and dx > 0 and dy > 0:  
#                 diag = j
#                 visited.add(j)


#             if right is not None and down is not None and diag is not None:
#                 break

#         if right is not None and down is not None and diag is not None:
#             group_indices.extend([right, down, diag])

#         original_indices = sorted_indices[group_indices]
#         meta_to_original.append(original_indices.tolist())

#         merged_expression = sorted_expression[group_indices].mean(axis=0)
#         merged_coords = sorted_coords[group_indices].mean(axis=0)
        

#         meta_cells.append(merged_expression)
#         meta_coords.append(merged_coords)


#     # build new adata
#     meta_X = np.vstack(meta_cells)
#     meta_coords = np.array(meta_coords)
#     adata_metacell = sc.AnnData(X=meta_X)
#     adata_metacell.var_names = adata.var_names.copy()
#     adata_metacell.obsm['spatial'] = meta_coords
#     # adata_metacell.uns['meta_to_original'] = [json.dumps(indices) for indices in meta_to_original]
#     adata_metacell.uns['meta_to_original'] = meta_to_original
#     adata_metacell.uns['original_cell_num'] = adata.n_obs
#     adata_metacell.X = np.array(adata_metacell.X)

#     return adata_metacell



def construct_metacell_data_dict(data_dict):
    """
    Apply metacell construction to all available AnnData objects in a multimodal dataset dictionary.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key is a modality name (e.g., `'RNA'`, `'Protein'`) and each value
        is a list of AnnData objects, one per tissue section. Each AnnData should contain:
        
        - `.X`: Expression or feature matrix (dense or sparse)  
        - `.obs`, `.var`: Standard metadata  
        - `.obsm['spatial']`: 2D spatial coordinates

        If a modality is missing in a section, use `None` to indicate it.

    Returns
    -------
    metacell_dict : dict
        A new dictionary with the same structure as `data_dict`, where each AnnData object has been replaced by its metacell-aggregated version, created using `metacell_construction_optimized()`. The modality and section alignment are preserved.
    """

    metacell_dict = {}

    for modality, adata_list in data_dict.items():
        new_list = []
        for adata in adata_list:
            if adata is None:
                new_list.append(None)
            else:
                # new_adata = metacell_construction(adata)
                new_adata = metacell_construction_optimized(adata)
                new_list.append(new_adata)
        metacell_dict[modality] = new_list

    return metacell_dict





def reconstruct_metacell_to_original(adata_metacell, metacell_embedding):
    """
    Expand metacell-level embeddings back to individual cells based on the
    metacell-to-original cell mapping.

    Parameters
    ----------
    adata_metacell : AnnData
        An AnnData object generated by `metacell_construction()`. It must contain:
        
        - `.uns['meta_to_original']`: A list of lists, where each sublist contains the indices of original cells belonging to a given metacell.
        - `.uns['original_cell_num']`: Total number of original cells before metacell construction.
    
    metacell_embedding : np.ndarray
        An array of shape (n_metacells, d), representing the learned embedding for each metacell.

    Returns
    -------
    original_embedding : np.ndarray
        An array of shape (n_original_cells, d), where each original cell inherits the embedding of its corresponding metacell.
    """
    meta_to_original = adata_metacell.uns['meta_to_original']
    original_cell_num = adata_metacell.uns['original_cell_num']

    # initialize full-size embedding matrix
    original_embedding = np.zeros((original_cell_num, metacell_embedding.shape[1]))

    for meta_idx, original_indices in enumerate(meta_to_original):
        original_embedding[original_indices] = metacell_embedding[meta_idx]

    return original_embedding






















###
