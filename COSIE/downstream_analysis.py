import os
from matplotlib import cm
from matplotlib.colors import to_hex
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from anndata import AnnData
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import defaultdict


import torch
import anndata as ad
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize


import matplotlib as mpl
import scanpy as sc
from matplotlib.cm import get_cmap
from matplotlib import patches
from pandas.api.types import CategoricalDtype
from matplotlib.colors import to_rgb
from .utils import nn_approx



mpl.rcParams['pdf.fonttype'] = 42





def cluster_and_visualize_superpixel(
    final_embeddings,
    data_dict,
    n_clusters,
    mode="joint",  # 'joint' or 'independent' or "defined"
    defined_labels=None,
    vis_basis="spatial",
    random_state=0,
    colormap=None,
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    offset=False,
    save_path=None,
    dpi=300,
    remove_title = False,
    remove_legend = False,
    remove_spine = False,
    figscale = 35
):
    """
    Perform clustering on superpixel embeddings across multiple tissue sections and visualize the results.

    Supports three clustering modes:
    
    - 'joint': All sections' embeddings are clustered together.
    
    - 'independent': Each section is clustered independently.
    
    - 'defined': Uses user-specified cluster labels.


    Parameters
    ----------
    final_embeddings : dict
        Dictionary of {section_id: np.ndarray} representing cell embeddings.

    data_dict : dict
        Dictionary of {modality: list of AnnData}, where each AnnData contains spatial coordinates.

    n_clusters : int
        Number of clusters to generate.

    mode : str, default "joint"
        Clustering mode: "joint", "independent", or "defined".

    defined_labels : dict or None
        Required if mode is "defined". A dictionary of {section_id: np.ndarray of cluster labels}.

    vis_basis : str, default "spatial"
        Key in `obsm` indicating spatial coordinates.

    random_state : int, default 0
        Random seed for KMeans clustering.

    colormap : str or list or None
        Color map used to assign RGB colors to clusters.

    swap_xy : bool, default False
        Whether to swap x and y coordinates.

    invert_x : bool, default False
        Whether to flip the image horizontally.

    invert_y : bool, default False
        Whether to flip the image vertically.

    offset : bool, default False
        Whether to shift coordinates to (0, 0).

    save_path : str or None, default None
        If specified, saves the figure(s) with this filename prefix.

    dpi : int, default 300
        DPI for the saved figure.

    remove_title : bool, default False
        Whether to remove figure title.

    remove_legend : bool, default False
        Whether to remove cluster legend.

    remove_spine : bool, default False
        Whether to remove axis borders.

    figscale : int, default 35
        Controls image figure size.

    Returns
    -------
    cluster_labels : dict
        Dictionary of {section_id: np.ndarray of cluster labels}.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    import os

    adata_list = []
    embeddings = []
    coords_all = []
    section_names = []

    for section, embedding in final_embeddings.items():
        idx = int(section[1:]) - 1
        for modality, adata_list_per_mod in data_dict.items():
            if idx < len(adata_list_per_mod) and adata_list_per_mod[idx] is not None:
                adata = adata_list_per_mod[idx]
                adata_list.append(adata)
                embeddings.append(embedding)
                coords = adata.obsm[vis_basis].copy()
                if swap_xy:
                    coords = coords[:, [1, 0]]
                coords = coords.astype(int)
                if offset:
                    offset_value = coords.min(axis=0)     
                    coords -= offset_value               
                coords_all.append(coords)
                section_names.append(section)
                break

    cluster_labels = {}

    if mode == "joint":
        print("Perform joint clustering...")
        combined_embedding = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        all_clusters = kmeans.fit_predict(combined_embedding)
        start = 0
        for section, emb in zip(section_names, embeddings):
            end = start + emb.shape[0]
            cluster_labels[section] = all_clusters[start:end]
            start = end
    elif mode == "independent":
        print("Perform independent clustering...")
        for section, emb in zip(section_names, embeddings):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels[section] = kmeans.fit_predict(emb)
    elif mode == 'defined':
        if defined_labels is None:
            raise ValueError("If mode='defined', you must provide `defined_labels`.")
        cluster_labels = defined_labels
    else:
        raise ValueError("mode must be 'joint' or 'independent'")

    for section, coords, labels in zip(section_names, coords_all, cluster_labels.values()):
        max_y, max_x = coords.max(axis=0) + 1
        image = np.full((max_y, max_x), fill_value=-1, dtype=int)
        for (y, x), label in zip(coords, labels):
            image[y, x] = label
        if invert_x:
            image = image[:, ::-1]
        if invert_y:
            image = image[::-1, :]
        section_save_path = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            section_save_path = f"{base}_section_{section}{ext or '.png'}"
        plot_histology_clusters(
            he_clusters_image=image,
            num_he_clusters=n_clusters,
            section_title=f"Section {section} ({mode})",
            colormap=colormap,
            save_path=section_save_path,
            dpi=dpi,
            figscale = figscale,
            remove_title = remove_title,
            remove_legend = remove_legend,
            remove_spine=remove_legend, 
        )

    return cluster_labels



def plot_histology_clusters(he_clusters_image,
                            num_he_clusters,
                            section_title=None,
                            colormap=None,
                            save_path=None,
                            figscale = 35,
                            remove_title = False,
                            remove_legend = False,
                            remove_spine=False, 
                            dpi=300):
    """
    Visualize cluster maps from 2D cluster masks.

    Parameters
    ----------
    he_clusters_image : np.ndarray
        2D array of shape (H, W) where each pixel holds an integer cluster ID.

    num_he_clusters : int
        Total number of clusters (used for color assignment).

    section_title : str or None, optional
        Title shown on the figure.

    colormap : str, list, or None, optional
        Colormap for cluster coloring. If None, a default color list is used.

    save_path : str or None, optional
        Path to save the resulting image. If None, no image is saved.

    figscale : int, default 35
        Controls image figure size.

    remove_title : bool, default False
        Whether to remove the title.

    remove_legend : bool, default False
        Whether to remove the cluster legend.

    remove_spine : bool, default False
        Whether to remove the axis frame/spines.

    dpi : int, default 300
        DPI of saved image.

    Returns
    -------
    None
    """

    if colormap is None:
        color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                      [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                      [23,190,207],[174,199,232],[255,187,120],[152,223,138],
                      [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                      [199,199,199],[219,219,141],[158,218,229],[16,60,90],
                      [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                      [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]

    elif isinstance(colormap, list):
        color_list = colormap

    else:
        cmap = cm.get_cmap(colormap)
        color_list = [ [int(255 * c) for c in to_rgb(cmap(i))] for i in range(len(cmap.colors)) ]

    image_rgb = 255 * np.ones([he_clusters_image.shape[0], he_clusters_image.shape[1], 3])
    for cluster in range(num_he_clusters):
        image_rgb[he_clusters_image == cluster] = color_list[cluster]
    image_rgb = np.array(image_rgb, dtype='uint8')

    plt.figure(figsize=(he_clusters_image.shape[1] // figscale, he_clusters_image.shape[0] // figscale))
    if remove_title:
        plt.title("")
    else:
        title = section_title if section_title else "Histology Clusters"
        plt.title(title, fontsize=18)
    plt.imshow(image_rgb, interpolation='none')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if not remove_legend:
        legend_elements = [patches.Patch(facecolor=np.array(color_list[i]) / 255,
                                         label=f'Cluster {i}')
                           for i in range(num_he_clusters)]
        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0.,
                   fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()





def cluster_and_visualize(
    final_embeddings,
    data_dict,
    n_clusters,
    mode="joint",  # 'joint' or 'independent'
    vis_basis="spatial",
    cluster_key="cluster_labels",
    random_state=0,
    s=50,
    alpha=0.9,
    colormap="tab20",
    plot_style="original",  # 'equal' or 'original'
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    save_path=None,
    dpi=300
):
    """
    Cluster cell embeddings and visualize the results for each tissue section at the spot level.
    Supports both joint and per-section clustering modes, and offers flexible
    visualization controls (axis flip, equal scaling, saving, etc.).

    Parameters
    ----------
    final_embeddings : dict
        A dictionary mapping section names (e.g., `'s1'`, `'s2'`, ...) to 2D NumPy arrays of shape (n_cells, latent_dim), representing cell embeddings for each section.
    
    data_dict : dict
        A dictionary where each key is a modality (e.g., `'RNA'`, `'Protein'`) and each value is a list of AnnData objects, one per tissue section. If a modality is missing in a section, use `None` as a placeholder.
    
    n_clusters : int
        Number of clusters to assign using k-means.
    
    mode : str, optional
        Clustering mode. Must be one of {'joint', 'independent'}.
        
        - 'joint': Cluster all sections together.
        - 'independent': Cluster each section separately. Default is 'joint'.
    
    vis_basis : str, optional
        The key in `.obsm` to use for visualization (e.g., `'spatial'`). Default is `'spatial'`.
    
    cluster_key : str, optional
        Column name in `.obs` to store cluster assignments. Default is `'cluster_labels'`.
    
    random_state : int, optional
        Random seed for k-means reproducibility. Default is 0.
    
    s : int, optional
        Dot size in scatter plots. Default is 50.
    
    alpha : float, optional
        Point transparency (0 to 1). Default is 0.9.
    
    colormap : str, optional
        Matplotlib colormap name for cluster coloring. Default is `'tab20'`.
    
    plot_style : str, optional
        Must be one of {'equal', 'original'}.
        
        - 'equal': Enforce equal axis aspect ratio.
        - 'original': Retain raw coordinate scale. Default is `'original'`.
    
    swap_xy : bool, optional
        If True, swap x and y axes in the scatter plot. Default is False.
    
    invert_x : bool, optional
        If True, invert the x-axis. Default is False.
    
    invert_y : bool, optional
        If True, invert the y-axis. Default is False.
    
    save_path : str or None, optional
        If provided, save the figures using this prefix. Individual files will be saved for each section. Default is None (no saving).
    
    dpi : int, optional
        Resolution of saved figures in DPI (dots per inch). Default is 300.

    Returns
    -------
    cluster_labels : dict
        A dictionary mapping section IDs to arrays of assigned cluster labels.
    """
    adata_list = []
    embeddings = []
    section_names = []

    for section, embedding in final_embeddings.items():
        idx = int(section[1:]) - 1
        for modality, adata_list_per_mod in data_dict.items():
            if idx < len(adata_list_per_mod) and adata_list_per_mod[idx] is not None:
                adata_list.append(adata_list_per_mod[idx])
                embeddings.append(embedding)
                section_names.append(section)
                break

    if mode == "joint":
        print("Perform joint clustering...")
        combined_embedding = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        all_clusters = kmeans.fit_predict(combined_embedding).astype(str)

        # fixed order
        cluster_order = [str(i) for i in range(n_clusters)]
        color_list = [to_hex(cm.get_cmap(colormap)(i)) for i in range(n_clusters)]
        color_mapping = {str(i): color_list[i] for i in range(n_clusters)}
        # print("Color Mapping (joint):", color_mapping)

        start = 0
        for adata, emb in zip(adata_list, embeddings):
            end = start + emb.shape[0]
            adata.obs[cluster_key] = all_clusters[start:end]
            adata.obs[cluster_key] = adata.obs[cluster_key].astype(
                CategoricalDtype(categories=cluster_order, ordered=True)
            )
            adata.uns[f"{cluster_key}_colors"] = [color_mapping[cat] for cat in cluster_order]
            start = end

    elif mode == "independent":
        print("Perform independent clustering...")
        for adata, emb in zip(adata_list, embeddings):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            clusters = kmeans.fit_predict(emb).astype(str)
            cluster_order = [str(i) for i in range(n_clusters)]
            adata.obs[cluster_key] = clusters
            adata.obs[cluster_key] = adata.obs[cluster_key].astype(
                CategoricalDtype(categories=cluster_order, ordered=True)
            )
            adata.uns[f"{cluster_key}_colors"] = [
                to_hex(cm.get_cmap(colormap)(i)) for i in range(n_clusters)
            ]
    else:
        raise ValueError("mode must be 'joint' or 'independent'")

    # Visualization
    cluster_labels = {}
    for adata, section in zip(adata_list, section_names):
        cluster_labels[section] = adata.obs[cluster_key]
        vis_coords = adata.obsm[vis_basis].copy()
        if swap_xy:
            vis_coords = vis_coords[:, [1, 0]]
            adata.obsm["__temp_basis__"] = vis_coords
            basis_to_plot = "__temp_basis__"
        else:
            basis_to_plot = vis_basis
        
        title = f"Section {section} ({mode})"

      
        fig = sc.pl.embedding(
            adata,
            basis=basis_to_plot,
            color=cluster_key,
            title=title,
            s=s,
            alpha=alpha,
            show=False,
            return_fig=True
        )
        
        ax = fig.axes[0]
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        if invert_x:
            ax.invert_xaxis()
        if invert_y:
            ax.invert_yaxis()
        if plot_style == "equal":
            ax.set_aspect("equal")
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir != "":
                os.makedirs(save_dir, exist_ok=True)
            file_root, file_ext = os.path.splitext(save_path)
            if file_ext == "":
                file_ext = ".pdf"
            section_save_path = f"{file_root}_section_{section}{file_ext}"
            print(f"Saving figure to: {section_save_path}")
            fig.savefig(section_save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        

        if "__temp_basis__" in adata.obsm:
            del adata.obsm["__temp_basis__"]


    return cluster_labels








def perform_imputation(
    data_dict,
    final_embeddings,
    target_section,
    target_modality,
    K_num=50,
    source_sections=None,
    target_molecules='All',
    block_size=None,
    metric='euclidean',
    accelerate=False,
    n_trees=10    
):
    """
    Perform KNN-based imputation for a specific modality in a target tissue section.

    The imputed values are computed by identifying the K nearest neighbors in the embedding space
    from source sections and averaging their expression values for the specified molecules. If accelerate = True, PCA will be conducted and imputation will be performed based on the PCA embedding instead. 

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key is a modality (e.g., `'RNA'`, `'Protein'`) and each value is a list of AnnData objects, one per tissue section. If a modality is missing in a section, use `None` as a placeholder.
    
    final_embeddings : dict
        A dictionary mapping section names (e.g., `'s1'`, `'s2'`, ...) to 2D NumPy arrays of shape (n_cells, latent_dim), representing cell embeddings for each section.
    
    target_section : str
        The name of the section to impute (e.g., `'s1'`).
    
    target_modality : str
        The modality to impute (e.g., `'RNA'`, `'Protein'`, `'Metabolite'`).
    
    K_num : int, optional
        Number of nearest neighbors used for imputation. Default is 50.
    
    source_sections : list of str or None, optional
        A list of section names to serve as the source data. If None, all sections with the target modality will be used. Default is None.
    
    target_molecules : str or list, optional
        Features (genes, proteins, metabolites, etc.) to impute:
        
        - `'All'`: Impute the intersection of all shared features across source sections.  
        - list: A specific list of features to impute (e.g., `['CD4', 'CD68']`).  
        Default is `'All'`.
    
    block_size : int or None, optional
        If set, perform block-wise imputation across features to reduce memory usage. Each block contains up to `block_size` features. Default is None.
    
    metric : str, optional
        Distance metric used in approximate nearest neighbor search (via Annoy). Must be one of: {'euclidean', 'manhattan', 'angular', 'hamming', 'dot'}. Default is `'euclidean'`.

    accelerate : bool, optional
        Whether to perform joint PCA dimensionality reduction (to 50 dimensions) on all embeddings before imputation. 
        This can accelerate nearest neighbor search and reduce memory usage. Default is False.

    
    n_trees : int, optional
        Number of trees used to build the Annoy index. Larger values increase accuracy at the cost of indexing time. Default is 10.

    Returns
    -------
    new_adata : AnnData
        A new AnnData object containing the imputed data. Includes:
        
        - `.X`: Imputed expression matrix as a dense NumPy array.  
        - `.obs`: Metadata copied from the target section’s reference AnnData.  
        - `.var`: Feature names associated with the imputed modality.  
        - `.obsm['spatial']`: Copied spatial coordinates (if present).
    """
    from sklearn.decomposition import PCA
    section_names = list(final_embeddings.keys())

    # Optional PCA acceleration
    if accelerate:
        print("Performing joint PCA on all embeddings for acceleration...")
        all_embeddings_list = [final_embeddings[sec] for sec in section_names]
        stacked_embeddings = np.vstack(all_embeddings_list)
    
        # Perform PCA
        pca = PCA(n_components=50)
        pca_embeddings = pca.fit_transform(stacked_embeddings)
    
        # Split back PCA results to each section
        sizes = [final_embeddings[sec].shape[0] for sec in section_names]
        pca_split = np.split(pca_embeddings, np.cumsum(sizes)[:-1])
        final_embeddings = {sec: emb for sec, emb in zip(section_names, pca_split)}
    
        print("PCA completed. All embeddings reduced to 50 dimensions.")
        for sec, emb in final_embeddings.items():
            print(f"  Section {sec}: PCA embedding shape = {emb.shape}")
    
    
    target_idx = section_names.index(target_section)

    # Find an AnnData as reference of spatial / obs
    for mod, adatas in data_dict.items():
        if target_idx < len(adatas) and adatas[target_idx] is not None:
            target_ref = adatas[target_idx]
            print(f"Using modality [{mod}] in section [{target_section}] as spatial/obs reference")
            break
    else:
        raise ValueError(f"No reference AnnData found for section {target_section}.")

    # Find source sections
    if source_sections is None:
        source_sections = [
            sec for sec in section_names 
            if target_modality in data_dict and
               section_names.index(sec) < len(data_dict[target_modality]) and
               data_dict[target_modality][section_names.index(sec)] is not None
        ]
        print(f"[{target_modality}] exists in {source_sections}, which will be used as source data section")
    else:
        print(f"Manually specify {source_sections} as source data")


    # Find overlap features
    if isinstance(target_molecules, str) and target_molecules.lower() == 'all':
        feature_sets = [
            set(data_dict[target_modality][section_names.index(sec)].var_names)
            for sec in source_sections
        ]
        feature_list = sorted(set.intersection(*feature_sets))
    else:
        feature_list = list(target_molecules)

    if len(feature_list) == 0:
        raise ValueError("No shared features found across source sections.")

    all_embeddings, all_expressions = [], []
    for sec in source_sections:
        idx = section_names.index(sec)
        ad = data_dict[target_modality][idx]
        emb = final_embeddings[sec]

        
        feat_idx = [ad.var_names.get_loc(f) for f in feature_list]
        X = ad.X[:, feat_idx]
        X = np.asarray(X.toarray() if hasattr(X, 'toarray') else X)
        all_embeddings.append(emb)
        all_expressions.append(np.asarray(X))

    source_emb = np.vstack(all_embeddings)
    source_X = np.vstack(all_expressions)

    target_emb = final_embeddings[target_section]

    if target_emb.shape[1] != source_emb.shape[1]:
        raise ValueError("Target and source embedding dimensions do not match.")



    n_cells, n_features = target_emb.shape[0], len(feature_list)
    imputed_X = np.zeros((n_cells, n_features))

    indices = nn_approx(target_emb, source_emb, knn=K_num, metric=metric, n_trees=n_trees)
    
    if block_size is None:
        for i, nn_ids in enumerate(indices):
            imputed_X[i] = source_X[nn_ids].mean(axis=0)
    else:
        for start in range(0, n_features, block_size):
            end = min(start + block_size, n_features)
            for i, nn_ids in enumerate(indices):
                imputed_X[i, start:end] = source_X[nn_ids, start:end].mean(axis=0)

    ref_var = data_dict[target_modality][section_names.index(source_sections[0])].var.loc[feature_list]
    new_adata = AnnData(X=imputed_X, obs=target_ref.obs.copy(), var=ref_var.copy())
    if 'spatial' in target_ref.obsm:
        new_adata.obsm['spatial'] = target_ref.obsm['spatial'].copy()

    return new_adata




def create_normalized_adata(adata):
    """
    Create a new AnnData object by min-max scaling the expression matrix of the input `adata`.

    The values in `.X` are scaled to the range [0, 1]. If `.X` is stored in sparse format,
    it will be converted to a dense NumPy array before normalization. The original `.obs`,
    `.var`, and `.obsm` fields are preserved in the new AnnData object.

    Parameters
    ----------
    
    adata : AnnData
        The input AnnData object containing expression data in `.X`.

    Returns
    -------
    
    new_adata : AnnData
        A new AnnData object with min-max normalized `.X`, while retaining the original `.obs`, `.var`, and `.obsm` attributes.
    """
    dense_X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    scaler = MinMaxScaler()
    normalized_X = scaler.fit_transform(dense_X)
    new_adata = AnnData(X=normalized_X, obs=adata.obs.copy(), var=adata.var.copy(), obsm=adata.obsm.copy())
    return new_adata




def plot_marker_comparison(
    molecule_name: str,
    adata1,
    adata2,
    section1_label: str = 'Section 1',
    section2_label: str = 'Section 2',
    basis: str = 'spatial',
    s: int = 50,
    alpha: float = 0.9,
    colormap: str = "turbo",
    plot_style: str = "original",  # 'equal' or 'original'
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
    save_path: str = None,
    dpi: int = 500,
    remove_legend = False,
    remove_spine = False,
    remove_title = False
):
    """
    Compare the spatial expression pattern of a specified molecule (e.g., gene, protein..)
    across two tissue sections at the spot level, each represented by an AnnData object.


    Parameters
    ----------
    molecule_name : str
        The molecule name to visualize. Must be present in `.var` of both AnnData objects.
    
    adata1 : AnnData
        The first AnnData object, e.g., for imputed data.
    
    adata2 : AnnData
        The second AnnData object, e.g., for observed data.
    
    section1_label : str, optional
        Plot title label for the first section. Default is `'Section 1'`.
    
    section2_label : str, optional
        Plot title label for the second section. Default is `'Section 2'`.
    
    basis : str, optional
        Key in `.obsm` specifying the spatial coordinate basis (e.g., `'spatial'`). Default is `'spatial'`.
    
    s : int, optional
        Dot size in the scatter plot. Default is 50.
    
    alpha : float, optional
        Transparency level of plotted points (between 0 and 1). Default is 0.9.
    
    colormap : str, optional
        Name of the matplotlib colormap used to represent expression intensity. Default is turbo.
    
    plot_style : str, optional
        Must be one of {'equal', 'original'}.
        
        - `'equal'`: Enforces equal aspect ratio on axes.  
        - `'original'`: Keeps raw coordinate scale.  
        Default is `'original'`.
    
    swap_xy : bool, optional
        If True, swaps x and y coordinates in both sections. Default is False.
    
    invert_x : bool, optional
        If True, inverts the x-axis direction. Default is False.
    
    invert_y : bool, optional
        If True, inverts the y-axis direction. Default is False.
    
    save_path : str or None, optional
        If provided, saves the resulting figure to the specified path. The file format
        is inferred from the extension (e.g., `.pdf`, `.png`). Default is None.
    
    dpi : int, optional
        Resolution of the saved figure in dots per inch. Default is 300.

    Returns
    -------
    None
        This function does not return any value. It displays a side-by-side comparison plot of molecule expression across the two sections at the spot level. If `save_path` is specified, the figure is also saved to disk.
    """
    
    # Swap XY if requested
    for adata in [adata1, adata2]:
        if swap_xy:
            coords = adata.obsm[basis][:, [1, 0]].copy()
            adata.obsm["__temp_basis__"] = coords
        else:
            adata.obsm["__temp_basis__"] = adata.obsm[basis].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (adata, label, ax) in enumerate(zip([adata1, adata2],
                                               [section1_label, section2_label],
                                               axes)):
        sc.pl.embedding(
            adata,
            basis="__temp_basis__",
            color=molecule_name,
            title=None if remove_title else f'{label} - {molecule_name}',
            s=s,
            alpha=alpha,
            ax=ax,
            show=False,
            colorbar_loc=None,
            cmap=colormap
        )
    
        if invert_x:
            ax.invert_xaxis()
        if invert_y:
            ax.invert_yaxis()
        if plot_style == 'equal':
            ax.set_aspect('equal')
    
        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)
    
        if remove_title:
            ax.set_title("")
    
        # Remove legend and colorbar manually if possible
        if remove_legend:
            legend = ax.get_legend()
            if legend:
                legend.remove()
    
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        file_root, file_ext = os.path.splitext(save_path)
        if file_ext == "":
            file_ext = ".pdf"
        save_path_final = f"{file_root}{file_ext}"
        print(f"Saving marker comparison to: {save_path_final}")
        plt.savefig(save_path_final, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    for adata in [adata1, adata2]:
        if "__temp_basis__" in adata.obsm:
            del adata.obsm["__temp_basis__"]


def prepare_image(adata, molecule_name, basis, swap_xy, invert_x, invert_y, offset):
    """
    Prepare a 2D image from molecule expression and spatial coordinates in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial coordinates in `obsm[basis]` and molecule expression in `X`.

    molecule_name : str
        Name of the molecule to visualize.

    basis : str
        The key in `obsm` to use for spatial coordinates (e.g., "spatial").

    swap_xy : bool
        Whether to swap x and y coordinates.

    invert_x : bool
        Whether to flip the image horizontally.

    invert_y : bool
        Whether to flip the image vertically.

    offset : bool
        Whether to shift coordinates so that the minimum becomes (0, 0).

    Returns
    -------
    image : np.ndarray
        2D array of shape (height, width) representing the molecule intensity at each spatial location.
    """
    
    coords = adata.obsm[basis].copy()
    if swap_xy:
        coords = coords[:, [1, 0]]
    coords = coords.astype(int)
    if offset:
        offset_value = coords.min(axis=0)
        coords -= offset_value 


    values = adata[:, molecule_name].X
    if hasattr(values, "toarray"):
        values = values.toarray().flatten()
    else:
        values = np.array(values).flatten()

    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), np.nan, dtype=float)
    for (y, x), val in zip(coords, values):
        image[y, x] = val

    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]

    return image



def plot_marker_comparison_superpixel(
    molecule_name: str,
    adata1,
    adata2,
    section1_label: str = 'Section 1',
    section2_label: str = 'Section 2',
    basis: str = 'spatial',
    colormap: str = "turbo",
    plot_style: str = "original",
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
    offset: bool = False,
    figscale: int = 35,
    dpi: int = 300,
    remove_title: bool = False,     
    remove_spine: bool = False,    
    remove_legend: bool = False,      
    save_path: str = None
):
    """
    Plot side-by-side spatial expression comparison of a target molecule at the superpixel level.

    Parameters
    ----------
    molecule_name : str
        Name of the molecule to visualize.

    adata1 : AnnData
        First AnnData object with molecule expression and spatial coordinates.

    adata2 : AnnData
        Second AnnData object with molecule expression and spatial coordinates.

    section1_label : str, default 'Section 1'
        Title label for the first section.

    section2_label : str, default 'Section 2'
        Title label for the second section.

    basis : str, default 'spatial'
        The key in `obsm` specifying spatial coordinates.

    colormap : str, default "turbo"
        Name of matplotlib colormap to use for intensity.

    plot_style : str, default "original"
        If "equal", enforce equal aspect ratio for square spatial representation.

    swap_xy : bool, default False
        Whether to swap x and y axes.

    invert_x : bool, default False
        Whether to flip the image horizontally.

    invert_y : bool, default False
        Whether to flip the image vertically.

    offset : bool, default False
        Whether to shift coordinates to align to (0, 0) origin.

    figscale : int, default 35
        Scaling factor for figure size.

    dpi : int, default 300
        Dots-per-inch for saved figure resolution.

    remove_title : bool, default False
        Whether to remove plot titles.

    remove_spine : bool, default False
        Whether to remove axes spines.

    remove_legend : bool, default False
        Whether to remove colorbar.

    save_path : str or None, default None
        If provided, save the figure to this path.

    Returns
    -------
    None
    """

    img1 = prepare_image(adata1, molecule_name, basis, swap_xy, invert_x, invert_y, offset)
    img2 = prepare_image(adata2, molecule_name, basis, swap_xy, invert_x, invert_y, offset)


    figsize1 = (img1.shape[1] / figscale, img1.shape[0] / figscale)
    figsize2 = (img2.shape[1] / figscale, img2.shape[0] / figscale)
    figsize = (figsize1[0] + figsize2[0], max(figsize1[1], figsize2[1]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, title in zip(axes, [img1, img2], [section1_label, section2_label]):
        im = ax.imshow(img, cmap=colormap, interpolation='none')
        if not remove_title:
            ax.set_title(f"{title} - {molecule_name}", fontsize=16)
        else:
            ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)
        if plot_style == "equal":
            ax.set_aspect("equal")

        if not remove_legend:
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)  

    if save_path:
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = ".png"
        save_path = base + ext
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving marker comparison to: {save_path}")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()








def highlight_joint_clusters_all_sections(
    cluster_labels,
    data_dict,
    n_clusters,
    highlight_labels,
    vis_basis="spatial",
    colormap=None,
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    offset=False,
    save_dir=None,
    figscale=35,
    dpi=300,
    remove_title=True,
    remove_legend=True,
    remove_spine=True,
    bg_color = [200, 200, 200]
):
    """
    Visualize and highlight specified clusters across all tissue sections. For each section, this function renders a spatial plot of cell clusters, highlighting the clusters specified in `highlight_labels` using distinct colors, while rendering all other clusters in a uniform background color. 

    Parameters
    ----------
    cluster_labels : dict
        Dictionary of {section_id: np.ndarray of cluster labels} for each section.

    data_dict : dict
        Dictionary of input data.

    n_clusters : int
        Total number of clusters.

    highlight_labels : list of int
        List of cluster labels to highlight. Other clusters are rendered with background color.

    vis_basis : str, default "spatial"
        Key in `obsm` specifying the coordinate basis to use.

    colormap : str or list or None, default None
        Name of matplotlib colormap to use, or a list of RGB values. If None, uses a default palette.

    swap_xy : bool, default False
        Whether to swap x and y axes in the coordinate system.

    invert_x : bool, default False
        Whether to flip the image horizontally.

    invert_y : bool, default False
        Whether to flip the image vertically.

    offset : bool, default False
        Whether to shift coordinates to (0, 0) minimum before rendering.

    save_dir : str or None, default None
        If provided, saves each figure as a JPEG to the specified directory.

    figscale : float, default 35
        Controls the scaling of the figure size.

    dpi : int, default 300
        Resolution of the saved figure.

    remove_title : bool, default True
        Whether to remove the figure title.

    remove_legend : bool, default True
        Whether to remove the cluster legend from the plot.

    remove_spine : bool, default True
        Whether to remove the axis spines (borders around the plot).

    bg_color : list of int, default [200, 200, 200]
        RGB color used for non-highlighted clusters.

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os
    from matplotlib.colors import to_rgb
    from matplotlib import cm

    if colormap is None:
        base_colors = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                       [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                       [23,190,207],[174,199,232],[255,187,120],[152,223,138],
                       [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                       [199,199,199],[219,219,141],[158,218,229],[16,60,90],
                       [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                       [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]
    elif isinstance(colormap, list):
        base_colors = colormap
    else:
        cmap = cm.get_cmap(colormap)
        base_colors = [[int(255 * c) for c in to_rgb(cmap(i))] for i in range(len(cmap.colors))]

    

    for section, labels in cluster_labels.items():
        idx = int(section[1:]) - 1
        coords = None
        for modality, adata_list in data_dict.items():
            if idx < len(adata_list) and adata_list[idx] is not None:
                coords = adata_list[idx].obsm[vis_basis].copy()
                if swap_xy:
                    coords = coords[:, [1, 0]]
                coords = coords.astype(int)
                if offset:
                    coords -= coords.min(axis=0)
                break
        if coords is None:
            print(f"Warning: Coordinates not found for section {section}.")
            continue

        max_y, max_x = coords.max(axis=0) + 1
        image = np.full((max_y, max_x), fill_value=-1, dtype=int)
        for (y, x), label in zip(coords, labels):
            image[y, x] = label
        if invert_x:
            image = image[:, ::-1]
        if invert_y:
            image = image[::-1, :]

        color_list = []
        for i in range(n_clusters):
            if i in highlight_labels:
                color_list.append(base_colors[i % len(base_colors)])
            else:
                color_list.append(bg_color)

        image_rgb = 255 * np.ones((image.shape[0], image.shape[1], 3))
        for cluster in range(n_clusters):
            image_rgb[image == cluster] = color_list[cluster]
        image_rgb = image_rgb.astype("uint8")

        fig, ax = plt.subplots(figsize=(image.shape[1] // figscale, image.shape[0] // figscale))

        if not remove_title:
            ax.set_title(f"Section {section} - Highlighted Clusters", fontsize=18)

        ax.imshow(image_rgb, interpolation='none')
        ax.set_xticks([]); ax.set_yticks([])

        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)

        if not remove_legend:
            legend_elements = [
                patches.Patch(facecolor=np.array(color_list[i]) / 255, label=f'Cluster {i}')
                for i in highlight_labels
            ]
            ax.legend(handles=legend_elements,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      borderaxespad=0.,
                      fontsize=12)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"highlighted_{section}.jpg")
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()
        plt.close()




def visualize_global_cluster_centroid_dendrogram(final_embeddings, cluster_label):
    """
    Visualize dendrogram for clusters shared across sections.

    Parameters
    ----------
    final_embeddings : dict
        Section-wise cell embeddings.
    
    cluster_label : dict
        Section-wise cluster label arrays (joint clustering).

    Returns
    -------
    None
    """
    # concatenate embedding from the same cluster across all sections
    cluster_embeddings = defaultdict(list)

    for section in final_embeddings:
        embeddings = final_embeddings[section]
        labels = cluster_label[section]
        for c in np.unique(labels):
            cluster_embeddings[c].append(embeddings[labels == c])

    # calculate the cluster centroid for each cluster
    cluster_ids = sorted(cluster_embeddings.keys())
    centroids = []
    for c in cluster_ids:
        all_points = np.vstack(cluster_embeddings[c])
        centroid = all_points.mean(axis=0)
        centroids.append(centroid)

    centroids = np.vstack(centroids)

    # visualization
    Z = linkage(centroids, method="average")
    plt.figure(figsize=(8, 4))
    dendrogram(Z, labels=[f"cluster_{c}" for c in cluster_ids], leaf_rotation=90)
    plt.title("Dendrogram of Global Cluster Centroids")
    plt.xlabel("Cluster ID")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()




def visualize_section_cluster_centroids_dendrogram(embedding, labels, section_name="section"):
    """
    Visualize hierarchical clustering dendrogram within a single section.

    Parameters
    ----------
    embedding : np.ndarray
        Array of shape (n_cells, d), the embedding of cells from one section.
    
    labels : np.ndarray or list
        Array of shape (n_cells,), the cluster labels for each cell in the section.
    
    section_name : str
        Name of the section, used for labeling.

    Returns
    -------
    None
    """
    centroids = []
    cluster_ids = []

    for c in sorted(np.unique(labels)):
        cluster_mask = labels == c
        centroid = embedding[cluster_mask].mean(axis=0)
        centroids.append(centroid)
        cluster_ids.append(f"{section_name}_cluster_{c}")

    centroids = np.vstack(centroids)
    Z = linkage(centroids, method="average")

    plt.figure(figsize=(6, 4))
    dendrogram(Z, labels=cluster_ids, leaf_rotation=90)
    plt.title(f"Dendrogram of Clusters ({section_name})")
    plt.xlabel("Cluster ID")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()




def merge_clusters_to_new_ids(cluster_label_dict, merge_groups):
    """
    Merge specified cluster IDs into new classes with IDs starting from current max + 1.

    Parameters
    ----------
    cluster_label_dict : dict
        Dictionary of {section: np.ndarray of cluster labels}.
    
    merge_groups : list of list
        List of groups to be merged, e.g., [[1, 2, 3], [12, 15]].

    Returns
    -------
    new_label_dict : dict
        Updated cluster label dictionary with merged labels.
    """
    # Find global max label to start assigning new IDs
    all_labels = np.concatenate(list(cluster_label_dict.values()))
    current_max = int(all_labels.max())
    next_id = current_max + 1

    # Build merge map: old ID → new merged ID
    merge_map = {}
    for group in merge_groups:
        for cid in group:
            merge_map[cid] = next_id
        next_id += 1

    # Apply mapping to each section
    new_label_dict = {}
    for section, labels in cluster_label_dict.items():
        labels = np.array(labels)
        mapped_labels = np.array([merge_map.get(lbl, lbl) for lbl in labels])
        new_label_dict[section] = mapped_labels

    return new_label_dict



def relabel_clusters_sequentially(cluster_label_dict):
    """
    Relabel all cluster IDs across sections to contiguous integers starting from 0.

    Parameters
    ----------
    cluster_label_dict : dict
        Dictionary of {section: np.ndarray of cluster labels}.

    Returns
    -------
    relabeled_dict : dict
        Dictionary with same keys, but cluster labels relabeled to 0, 1, 2, ...
    """
    all_labels = np.concatenate(list(cluster_label_dict.values()))
    unique_labels = sorted(np.unique(all_labels))
    
    relabel_map = {old: new for new, old in enumerate(unique_labels)}

    relabeled_dict = {}
    for section, labels in cluster_label_dict.items():
        relabeled_labels = np.vectorize(relabel_map.get)(labels)
        relabeled_dict[section] = relabeled_labels

    return relabeled_dict



        

###