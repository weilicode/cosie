import os
import numpy as np
import anndata as ad
import math
import scipy
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
from pathlib import Path


from .utils import nn_approx

def perform_strong_linkage_knn(data1, data2):
    """
    Construct triplet pairs between two sections of the same modality using symmetric KNN-based linkage.

    This function is designed to identify strong linkages between shared-modality data across two tissue sections. For each anchor cell, a positive sample is selected from the other section, and a negative sample is randomly selected from the same section as the anchor.

    Parameters
    ----------
    data1 : np.ndarray
        Feature matrix of dataset 1, of shape (n1, d).
    data2 : np.ndarray
        Feature matrix of dataset 2, of shape (n2, d).

    Returns
    -------
    triplets : np.ndarray
        An array of shape (n1 + n2, 3), where each row contains:
        
        - anchor index  
        - positive index (from the other section via KNN)  
        - negative index (randomly chosen from the same section as anchor)

        Indices are in concatenated form:
        
        - `[0, ..., n1 - 1]` for data1  
        - `[n1, ..., n1 + n2 - 1]` for data2
    """

    n1 = data1.shape[0]
    n2 = data2.shape[0]

    # data1 → data2 
    anchor1 = np.arange(n1)
    pos1 = nn_approx(data1, data2, knn=1, include_distances=False).flatten()
    pos1 += n1  
    pair1 = np.column_stack((anchor1, pos1))

    # data2 → data1
    anchor2 = np.arange(n2) + n1
    pos2 = nn_approx(data2, data1, knn=1, include_distances=False).flatten()
    pair2 = np.column_stack((anchor2, pos2))

    # combine
    pair = np.concatenate((pair1, pair2), axis=0)

    # generate negative partner 
    anchors = pair[:, 0]
    negatives = np.zeros_like(anchors)

    mask1 = anchors < n1 
    negatives[mask1] = np.random.choice(np.arange(n1), size=np.sum(mask1), replace=True)
    mask2 = ~mask1  
    negatives[mask2] = np.random.choice(np.arange(n2) + n1, size=np.sum(mask2), replace=True)

    triplets = np.column_stack((pair, negatives))

    return triplets



def load_protein_gene_mapping():
    """
    Load relationships between proteins and their associated genes. 
    This function reads a CSV file (Protein_gene_relationship.csv) containing curated protein-to-gene mappings and returns a dictionary used for weak linkage construction. 
    Users may also customize the CSV file to suit their own dataset.

    Parameters
    ----------
    None

    Returns
    -------
    protein_to_gene : dict
        A dictionary mapping each protein name (lowercased `str`) to a `set` of associated gene names (also lowercased `str`). 
    """

    csv_path = Path(__file__).parent / 'Protein_gene_relationship.csv'  
    mapping_df = pd.read_csv(csv_path)
    protein_to_gene = {}
    for _, row in mapping_df.iterrows():
        curr_protein_name = row["Protein name"].strip().casefold() 
        genes = set(row["RNA name"].strip().split("/")) if pd.notna(row["RNA name"]) else set()
        
        if curr_protein_name in protein_to_gene:
            protein_to_gene[curr_protein_name].update(gene.casefold() for gene in genes)
        else:
            protein_to_gene[curr_protein_name] = {gene.casefold() for gene in genes}
    
    return protein_to_gene



def perform_weak_linkage_knn(adata1, adata2, modality1, modality2, num_hvg=3000):
    """
    Construct weak linkage triplet pairs between two datasets of different but biologically related modalities.

    This function performs symmetric nearest neighbor search using a shared feature space
    derived from overlapping features (e.g., shared genes, or mapped protein-gene pairs).
    For each anchor cell, a positive cell is selected from the other dataset (opposite modality),
    and a negative cell is randomly selected from the same modality.

    - For RNA-protein matching, a curated mapping is loaded via `load_protein_gene_mapping()`.
    - For other epigenomic or transcriptomic modalities, shared features are determined by
      intersecting `.var_names`.

    Parameters
    ----------
    adata1 : AnnData
        AnnData object for dataset 1.
    
    adata2 : AnnData
        AnnData object for dataset 2.
    
    modality1 : str
        Modality type of `adata1`. Must be one of:
        {'RNA', 'RNA_panel2', 'Protein', 'H3K27me3', 'H3K27ac', 'ATAC', 'H3K4me3'}.
    
    modality2 : str
        Modality type of `adata2`. Must be one of:
        {'RNA', 'RNA_panel2', 'Protein', 'H3K27me3', 'H3K27ac', 'ATAC', 'H3K4me3'}.
    
    num_hvg : int, optional
        Number of highly variable features to retain for feature matching.
        Default is 3000.

    Returns
    -------
    triplets : np.ndarray
        An array of shape (n1 + n2, 3), where each row represents:
        
        - anchor index  
        - positive index (from the other section via KNN)  
        - negative index (randomly chosen from the same section as anchor)


        Indices are in concatenated form:
        
        - `[0, ..., n1 - 1]` for data1  
        - `[n1, ..., n1 + n2 - 1]` for data2
    """

    adata1_tmp = adata1.copy()
    adata2_tmp = adata2.copy()
    adata1_tmp.var_names_make_unique()
    adata2_tmp.var_names_make_unique()

    def select_hvg(adata, modality):
        if adata.shape[1] > num_hvg:
            print(f'Selecting {num_hvg} HVGs for {modality} with {adata.shape[1]} genes for linkage construction')
            if modality in ['RNA', 'RNA_panel2']:
                sc.pp.highly_variable_genes(adata, n_top_genes=num_hvg, flavor="seurat_v3")
            else:
                sc.pp.highly_variable_genes(adata, n_top_genes=num_hvg)
            return adata[:, adata.var['highly_variable']].copy()
        return adata  

    adata1_tmp = select_hvg(adata1_tmp, modality1)
    adata2_tmp = select_hvg(adata2_tmp, modality2)

    if (modality1 in ['RNA', 'RNA_panel2'] and modality2 == "Protein") or (modality1 == "Protein" and modality2 in ['RNA', 'RNA_panel2']):

        # Load the pre-defined gene-protein relationship
        protein_to_gene = load_protein_gene_mapping()

        if modality1 in ['RNA', 'RNA_panel2']:
            adata_rna, adata_protein = adata1_tmp, adata2_tmp
        else:
            adata_rna, adata_protein = adata2_tmp, adata1_tmp

        rna_var_names_casefold = {gene.casefold(): gene for gene in adata_rna.var_names}
        rna_protein_correspondence = []
        for protein_name in adata_protein.var_names:
            normalized_protein_name = protein_name.strip().casefold()
            if normalized_protein_name not in protein_to_gene:
                continue  

            curr_rna_names = protein_to_gene[normalized_protein_name]  

            for rna_name in curr_rna_names:
                normalized_rna_name = rna_name.casefold() 

                if normalized_rna_name in rna_var_names_casefold:
                    matched_rna_name = rna_var_names_casefold[normalized_rna_name] 
                    rna_protein_correspondence.append([matched_rna_name, protein_name])

        if len(rna_protein_correspondence) == 0:
            raise ValueError("No RNA-Protein matches found. Check if RNA and protein names exist in the dataset and mapping file.")
        rna_protein_correspondence = np.array(rna_protein_correspondence)
        print(f'Number of overlapping features: {rna_protein_correspondence.shape[0]}')

        adata_rna = adata_rna[:, rna_protein_correspondence[:, 0]].copy()
        adata_protein = adata_protein[:, rna_protein_correspondence[:, 1]].copy()

        if modality1 in ['RNA', 'RNA_panel2']:
            adata1_tmp, adata2_tmp = adata_rna, adata_protein
        else:
            adata1_tmp, adata2_tmp = adata_protein, adata_rna

    
    else:
        overlap_features = adata1_tmp.var_names.intersection(adata2_tmp.var_names)
        print(f'Number of overlapping features: {len(overlap_features)}')
    
        if len(overlap_features) == 0:
            raise ValueError(f"No overlapping features found between {modality1} and {modality2} datasets.")
    
        adata1_tmp = adata1_tmp[:, overlap_features].copy()
        adata2_tmp = adata2_tmp[:, overlap_features].copy()

    if isinstance(adata1_tmp.X, np.ndarray) is False:
        print("Converting adata1_tmp.X from sparse to dense...")
        adata1_tmp.X = adata1_tmp.X.toarray()

    if isinstance(adata2_tmp.X, np.ndarray) is False:
        print("Converting adata2_tmp.X from sparse to dense...")
        adata2_tmp.X = adata2_tmp.X.toarray()

    target_sum = (np.median(adata1_tmp.X.sum(axis=1)) + np.median(adata2_tmp.X.sum(axis=1))) / 2
    sc.pp.normalize_total(adata1_tmp, target_sum=target_sum)
    sc.pp.log1p(adata1_tmp)
    sc.pp.scale(adata1_tmp)

    sc.pp.normalize_total(adata2_tmp, target_sum=target_sum)
    sc.pp.log1p(adata2_tmp)
    sc.pp.scale(adata2_tmp)

    num1 = adata1_tmp.shape[0]
    num2 = adata2_tmp.shape[0]

    anchor = np.arange(num1)
    positive = nn_approx(adata1_tmp.X, adata2_tmp.X, knn=1, include_distances=False).flatten() + num1
    pair1 = np.column_stack((anchor, positive))

    anchor2 = np.arange(num2) + num1
    positive2 = nn_approx(adata2_tmp.X, adata1_tmp.X, knn=1, include_distances=False).flatten()
    pair2 = np.column_stack((anchor2, positive2))

    pair = np.concatenate((pair1, pair2), axis=0)

    # negavite partners generate 
    anchors = pair[:, 0]
    negative_samples = np.zeros_like(anchors)

    mask1 = anchors < num1
    negative_samples[mask1] = np.random.choice(np.arange(num1), size=np.sum(mask1), replace=True)
    mask2 = ~mask1
    negative_samples[mask2] = np.random.choice(np.arange(num2) + num1, size=np.sum(mask2), replace=True)

    pair_with_negatives = np.column_stack((pair, negative_samples))


    return pair_with_negatives



def compute_linkages(data_dict, linkage_indicator, num_hvg=3000):
    """
    Construct pairwise linkage triplets across specified tissue sections and modalities
    using symmetric nearest neighbor matching.

    This function supports both strong linkages (between sections sharing the same modality)
    and weak linkages (between biologically related but distinct modalities such as RNA-Protein).
    This function allows flexible linkage construction based on a user-defined `linkage_indicator` dictionary.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key is a modality (e.g., `'RNA'`, `'Protein'`) and each value
        is a list of AnnData objects (one per tissue section). Each AnnData must include:
        
        - `.X`: Feature or expression matrix  
        - `.obs`, `.var`: Metadata  
        - `.obsm['spatial']`: Spatial coordinates  
        
        Use `None` if a modality is missing in a section.

    linkage_indicator : dict
        A dictionary specifying which tissue section pairs and modality pairs should be linked.
        Format:
        {("s1", "s2"): [("RNA", "RNA"), ("RNA", "Protein")],("s2", "s3"): [("ATAC", "RNA")]}
        means: constructing linkage between section s1 and s2 using both RNA-RNA strong linkage and RNA-Protein weak linkage; constructing linkage between section s2 and s3 using ATAC-RNA linkage.

    num_hvg : int, optional
        Number of highly variable features to retain during feature matching.
        Default is 3000.

    Returns
    -------
    linkage_results : dict
        A dictionary mapping each section pair (e.g., `'s1_s2'`) to a NumPy array of triplets.
        Each triplet has the form:
        
        - `(anchor_index, positive_index, negative_index)`,  

        Example format:
        {"s1_s2": np.ndarray of shape (N1, 3), "s2_s3": np.ndarray of shape (N2, 3)}
    """
    linkage_results = {}

    # visit all specified section pairs
    for (sec1, sec2), modality_pairs in linkage_indicator.items():
        sec1_idx = int(sec1[1:]) - 1 
        sec2_idx = int(sec2[1:]) - 1  

        all_knn_pairs = []  
        shared_neg_samples = None  

        for i, (modality1, modality2) in enumerate(modality_pairs):
            sections1 = data_dict.get(modality1)
            sections2 = data_dict.get(modality2)

            if sections1 is None or sections2 is None:
                raise ValueError(f"Modality {modality1} or {modality2} not found in data_dict")

            try:
                adata1 = sections1[sec1_idx]
                adata2 = sections2[sec2_idx]
            except IndexError:
                raise ValueError(f"Section index {sec1_idx} or {sec2_idx} out of range for modalities {modality1}, {modality2}")

            if adata1 is None or adata2 is None:
                raise ValueError(f"Missing data: {modality1} in {sec1} or {modality2} in {sec2} is None")

            print(f'Computing linkage between {modality1} ({sec1}) and {modality2} ({sec2})')

            # for shared modality, use joint harmony/pca representation
            if modality1 == modality2:
                # Try to use Harmony embedding if available
                if f'{modality1}_harmony' in adata1.obsm and f'{modality1}_harmony' in adata2.obsm:
                    embedding_key = f'{modality1}_harmony'
                    # print(f"Using Harmony embedding for shared modality {modality1} between {sec1} and {sec2}")
                elif f'{modality1}_pca' in adata1.obsm and f'{modality1}_pca' in adata2.obsm:
                    embedding_key = f'{modality1}_pca'
                    # print(f"Using PCA embedding for shared modality {modality1} between {sec1} and {sec2}")
                else:
                    raise ValueError(f"Neither Harmony nor PCA embedding found for shared modality {modality1} in sections {sec1}, {sec2}")
                
                knn_pair = perform_strong_linkage_knn(adata1.obsm[embedding_key], adata2.obsm[embedding_key])


            else:
                knn_pair = perform_weak_linkage_knn(adata1, adata2, modality1, modality2, num_hvg=num_hvg)

            # If multiple linkages exist between two sections, share the same negative partner 
            if shared_neg_samples is None:
                shared_neg_samples = knn_pair[:, 2].copy()  
            else:
                knn_pair[:, 2] = shared_neg_samples  

            all_knn_pairs.append(knn_pair) 

        linkage_results[f"{sec1}_{sec2}"] = np.vstack(all_knn_pairs)

    return linkage_results




def split_raw_data(data_dict, spatial_loc_dict, n_x, n_y):

    """
    Divide spatial omics data from each section into spatially partitioned subgraphs.

    Each tissue section is split into a grid of `n_x × n_y` rectangular regions of equal size.
    This function is primarily used for scaling to large spatial datasets by enabling subgraph-level analysis.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key is a modality (e.g., `'RNA'`, `'Protein'`) and each value
        is a list of AnnData objects (one per tissue section). Each AnnData must include:
        
        - `.X`: Feature matrix  
        - `.obs`, `.var`: Metadata  
        - `.obsm['spatial']`: 2D spatial coordinates  
        
        Use `None` if a modality is missing in a section.
    
    spatial_loc_dict : dict
        A dictionary mapping each section name (e.g., `'s1'`, `'s2'`, ...) to a 2D NumPy array of shape `(n_cells, 2)`, representing the spatial coordinates of each cell in the section.
    
    n_x : int
        Number of spatial partitions along the x-axis.
    
    n_y : int
        Number of spatial partitions along the y-axis.

    Returns
    -------
    sub_data_dict : dict
        A nested dictionary organizing modality-specific AnnData subgraphs by section and region.

        Format:
        {
            's1': {
                0: {'RNA': AnnData, 'Protein': AnnData, ...},  # First spatial region of section s1
                
                1: {...},
                ...
            },
            
            's2': {
                ...
            }
        }

    """


    sub_data_dict = {}

    for sec_idx, section in enumerate(spatial_loc_dict.keys()):  
        spatial_coords = spatial_loc_dict[section]

        x_min, x_max = np.min(spatial_coords[:, 0]), np.max(spatial_coords[:, 0])
        y_min, y_max = np.min(spatial_coords[:, 1]), np.max(spatial_coords[:, 1])

        window_size_x = (x_max - x_min) / n_x
        window_size_y = (y_max - y_min) / n_y

        if section not in sub_data_dict:
            sub_data_dict[section] = {}

        subgraph_masks = {}  

        for i in range(n_x):
            for j in range(n_y):
                x_start, x_end = x_min + i * window_size_x, x_min + (i + 1) * window_size_x
                y_start, y_end = y_min + j * window_size_y, y_min + (j + 1) * window_size_y

                mask = (
                    (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] <= x_end) &
                    (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] <= y_end)
                )

                if np.any(mask):
                    subgraph_masks[len(subgraph_masks)] = mask  

        for modality, adata_list in data_dict.items():
            adata_section = adata_list[sec_idx]  

            if adata_section is None:
                continue

            for sub_idx, mask in subgraph_masks.items():
                if sub_idx not in sub_data_dict[section]:
                    sub_data_dict[section][sub_idx] = {}

                sub_data_dict[section][sub_idx][modality] = adata_section[mask].copy()

    return sub_data_dict


def split_into_subgraphs(data_tensor, spatial_coords, n_x, n_y):

    """
    Split a feature tensor and its corresponding spatial coordinates into `n_x × n_y`
    rectangular subregions.

    This function is commonly used in subgraph-based training or analysis pipelines for
    large-scale spatial omics datasets. It partitions both the data and the coordinate
    space into equal-sized grid regions, producing corresponding subsets.

    Parameters
    ----------
    data_tensor : torch.Tensor
        A tensor of shape `(n_cells, dim)`, representing the input features.
    
    spatial_coords : np.ndarray
        A NumPy array of shape `(n_cells, 2)` containing 2D spatial coordinates.
    
    n_x : int
        Number of partitions along the x-axis.
    
    n_y : int
        Number of partitions along the y-axis.

    Returns
    -------
    sub_feature_dict : List[torch.Tensor]
        A list of tensors, each containing the features corresponding to one spatial subgraph.
    
    sub_spatial_list : List[np.ndarray]
        A list of NumPy arrays of shape `(n_sub_cells, 2)`, where each array contains the spatial coordinates of cells within the corresponding subgraph. 
    """

    sub_feature_dict, sub_spatial_list = [], []

    x_min, x_max = np.min(spatial_coords[:, 0]), np.max(spatial_coords[:, 0])
    y_min, y_max = np.min(spatial_coords[:, 1]), np.max(spatial_coords[:, 1])

    window_size_x = (x_max - x_min) / n_x
    window_size_y = (y_max - y_min) / n_y

    for i in range(n_x):
        for j in range(n_y):
            x_start, x_end = x_min + i * window_size_x, x_min + (i + 1) * window_size_x
            y_start, y_end = y_min + j * window_size_y, y_min + (j + 1) * window_size_y

            mask = (
                (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] <= x_end) &
                (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] <= y_end)
            )

            sub_indices = np.where(mask)[0]

            if len(sub_indices) > 0:
                sub_feature_dict.append(data_tensor[sub_indices].clone())  
                sub_spatial_list.append(spatial_coords[sub_indices].copy())  

    return sub_feature_dict, sub_spatial_list



def preprocess_data_for_subgraphs(data_dict, feature_dict, spatial_loc_dict, linkage_indicator, n_x, n_y, num_hvg=3000):
    """
    Split each section into spatial subgraphs and compute cross-section triplet linkages
    at the subgraph level.

    This function partitions the full spatial omics data into `n_x × n_y` subregions per section,
    assigns feature and spatial data to each subgraph, and constructs triplet linkages across
    specified pairs of sections and modalities using the provided `linkage_indicator`.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key is a modality (e.g., `'RNA'`, `'Protein'`) and each value
        is a list of AnnData objects (one per tissue section). Use `None` if a modality is
        missing from a section.
    
    feature_dict : dict
        A dictionary mapping each section name (e.g., `'s1'`, `'s2'`, ...) to a sub-dictionary
        containing processed feature tensors for each modality as `torch.FloatTensor`.
        Format:
        {
            's1': {'RNA': torch.Tensor, 'Protein': torch.Tensor, ...},
            's2': {...}
        }
    
    spatial_loc_dict : dict
        A dictionary mapping each section name to a 2D NumPy array of spatial coordinates.
    
    linkage_indicator : dict
        A dictionary specifying which tissue section pairs and modality pairs should be linked.
        Format:
        {
            ("s1", "s2"): [("RNA", "RNA"), ("RNA", "Protein")],
            ("s2", "s3"): [("ATAC", "RNA")]
        }
        means: constructing linkage between section s1 and s2 using both RNA-RNA strong linkage and RNA-Protein weak linkage; constructing linkage between section s2 and s3 using ATAC-RNA linkage

    n_x : int
        Number of spatial divisions along the x-axis per section.
    
    n_y : int
        Number of spatial divisions along the y-axis per section.
    
    num_hvg : int, optional
        Number of highly variable features to retain for linkage matching.
        Default is 3000.

    Returns
    -------
    new_feature_dict : dict
        Nested dictionary of subgraph-level feature tensors.
        Format: `{section -> subregion index -> modality -> feature tensor}`.
    
    new_spatial_loc_dict : dict
        Nested dictionary of subgraph spatial coordinates.
        Format: `{section -> subregion index -> spatial coordinate array}`.
    
    new_linkage_results : dict
        Dictionary storing cross-section linkage triplets at the subgraph level.
        Keys are 4-tuples `(sec1, sub1_idx, sec2, sub2_idx)`, and values are NumPy arrays
        of triplets `(anchor, positive, negative)` in concatenated space.
    """

    new_feature_dict = {}
    new_spatial_loc_dict = {}
    for section, modalities in feature_dict.items():
        print(f"Splitting section [{section}] into {n_x} x {n_y} subgraphs")

        new_feature_dict[section] = {}
        new_spatial_loc_dict[section] = {}

        spatial_coords = spatial_loc_dict[section]  # `numpy.ndarray`

        for modality, data_tensor in modalities.items():
            print(f"Splitting {modality} in section {section}...")

            sub_feature_list, sub_spatial_list = split_into_subgraphs(data_tensor, spatial_coords, n_x, n_y)


            for sub_idx in range(len(sub_feature_list)):
                if sub_idx not in new_feature_dict[section]:
                    new_feature_dict[section][sub_idx] = {}  
                new_feature_dict[section][sub_idx][modality] = sub_feature_list[sub_idx]
                new_spatial_loc_dict[section][sub_idx] = sub_spatial_list[sub_idx]
                
    sub_data_dict = split_raw_data(data_dict, spatial_loc_dict, n_x, n_y)


    new_linkage_results = {}

    for (sec1, sec2), modality_pairs in linkage_indicator.items():
        if sec1 not in new_feature_dict or sec2 not in new_feature_dict:
            continue

        for sub1_idx in new_feature_dict[sec1]:
            for sub2_idx in new_feature_dict[sec2]:
                linkage_result = compute_linkages_per_subgraph(sub_data_dict, sec1, sub1_idx, sec2, sub2_idx, modality_pairs, num_hvg)

                if linkage_result is not None:
                    new_linkage_results[(sec1, sub1_idx, sec2, sub2_idx)] = linkage_result

    return new_feature_dict, new_spatial_loc_dict, new_linkage_results






def compute_linkages_per_subgraph(sub_data_dict, sec1, sub1_idx, sec2, sub2_idx, modality_pairs, num_hvg=3000):
    """
    
    Compute triplet linkages between two spatial subgraphs, each from a different section.

    This function operates on spatially partitioned data (e.g., from `split_raw_data`) and
    generates triplet-based cross-section linkages across paired subgraphs. It supports both
    strong and weak modality linkage based on the specified modality pairs.


    Parameters
    ----------
    sub_data_dict : dict
        A nested dictionary containing split AnnData objects for each modality within each subgraph per section.
    
    sec1 : str
        Name of the first section (e.g., `'s1'`).
    
    sub1_idx : int
        Index of the subgraph within the first section.
    
    sec2 : str
        Name of the second section (e.g., `'s2'`).
    
    sub2_idx : int
        Index of the subgraph within the second section.
    
    modality_pairs : list of tuple of str
        List of modality pairs to link across the two subgraphs.
        Each pair is a tuple like `('RNA', 'RNA')` or `('RNA', 'Protein')`.
    
    num_hvg : int, optional
        Number of highly variable features to retain for linkage construction.
        Default is 3000.

    Returns
    -------
    triplets : np.ndarray
        A NumPy array of shape `(n_triplets, 3)`, where each row is a triplet:
        `(anchor_index, positive_index, negative_index)`.
    """

    all_knn_pairs = []  
    shared_neg_samples = None  

    for modality1, modality2 in modality_pairs:
        if sec1 not in sub_data_dict or sec2 not in sub_data_dict:
            continue

        if sub1_idx not in sub_data_dict[sec1] or sub2_idx not in sub_data_dict[sec2]:
            continue

        if modality1 not in sub_data_dict[sec1][sub1_idx] or modality2 not in sub_data_dict[sec2][sub2_idx]:
            continue

        adata1 = sub_data_dict[sec1][sub1_idx][modality1]
        adata2 = sub_data_dict[sec2][sub2_idx][modality2]

        print(f"Computing linkage between [{modality1}] ({sec1}-{sub1_idx}) and [{modality2}] ({sec2}-{sub2_idx})")

        if modality1 == modality2:
            embedding_key = f'{modality1}_harmony'
            if embedding_key in adata1.obsm and embedding_key in adata2.obsm:
                knn_pair = perform_strong_linkage_knn(adata1.obsm[embedding_key], adata2.obsm[embedding_key])
            else:
                raise ValueError(f"Harmony embedding '{embedding_key}' not found in both subgraphs {sec1}-{sub1_idx}, {sec2}-{sub2_idx}")

        else:
            knn_pair = perform_weak_linkage_knn(adata1, adata2, modality1, modality2, num_hvg=num_hvg)

        if shared_neg_samples is None:
            shared_neg_samples = knn_pair[:, 2].copy()  
        else:
            knn_pair[:, 2] = shared_neg_samples  

        all_knn_pairs.append(knn_pair)  

    if len(all_knn_pairs) > 0:
        return np.vstack(all_knn_pairs)
    else:
        return None




###