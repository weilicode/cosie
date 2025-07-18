import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from .model_component import GraphAutoencoder, Prediction_mlp
from .linkage_construction import *
from .data_preprocessing import reconstruct_metacell_to_original
from .loss import *
from .utils import *
from tqdm import tqdm





class COSIE_model(nn.Module):
    """
    The core model class of the COSIE framework, designed for spatial multimodal integration, imputation, and enhancement across multiple tissue sections. This class defines the model structure, and handles both training and inference in full-graph or subgraph settings.

    Parameters
    ----------
    config : dict
        Configuration dictionary defining architecture and training hyperparameters.
        Includes sub-configs for GraphAutoencoder, Prediction modules, and training settings.
        This can be customized in configure script. 
    
    feature_dict : dict
        A dictionary mapping section names (e.g., `'s1'`, `'s2'`) to a sub-dictionary of processed feature tensors for each modality (as `torch.FloatTensor`). 
        Format:
        {'s1': {'RNA': tensor[n_cells, d], 'Protein': tensor[n_cells, d], ...},'s2': {...}, ...}


    Attributes
    ----------
    autoencoders : nn.ModuleDict
        Dictionary of GraphAutoencoder models per modality.
    
    predictors : nn.ModuleDict
        Dictionary of Prediction modules for all available modality pairs.
    
    triplet_loss_fn : torch.nn.TripletMarginLoss
        Loss function used for cross-section integration.
    
    latent_dim : int
        Embedding dimension.
    
    all_modalities : list
        List of all detected modalities across sections.

    Methods
    -------
    to_device()
        Move the entire model, including all submodules, to a specified device.
    
    train_model()
        Train the model on either the full graph or spatially partitioned subgraphs, depending on the `n_x` and `n_y` grid splits.
    """
    def __init__(self, config, feature_dict):
        super(COSIE_model, self).__init__()
        self.config = config
        self.latent_dim = config['GraphAutoencoder']['hidden_dim'][-1]  
        self.all_modalities = []
        self.triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')

        modality_dims = {}  
        
        for section, modalities in feature_dict.items():
            for modality, features in modalities.items():
                if modality not in modality_dims:
                    modality_dims[modality] = features.shape[1]  
                    self.all_modalities.append(modality)

        print('All modalities:', self.all_modalities)

        self.autoencoders = nn.ModuleDict()
        print('-------- Encoder description --------')
        for modality, input_dim in modality_dims.items():
            encoder_dim = [input_dim] + config['GraphAutoencoder']['hidden_dim']
            self.autoencoders[modality] = GraphAutoencoder(
                encoder_dim,
                config['GraphAutoencoder']['activations'],
            )
            print(f"Encoder [{modality}]: Input {input_dim} → Hidden {encoder_dim[1:]}")

        self.predictors = nn.ModuleDict()
        created_predictors = set()  

        print('-------- Dual prediction module description --------')
        for section, modalities in feature_dict.items():
            modality_list = list(modalities.keys())  
            for i, mod1 in enumerate(modality_list):
                for mod2 in modality_list[i + 1:]:  
                    predictor_name_1 = f"{mod1}_to_{mod2}"
                    predictor_name_2 = f"{mod2}_to_{mod1}"
        
                    if predictor_name_1 not in created_predictors:
                        pred_dim = [self.latent_dim] + config['Prediction']['hidden_dim'] + [self.latent_dim]
                        self.predictors[predictor_name_1] = Prediction_mlp(pred_dim)
                        self.predictors[predictor_name_2] = Prediction_mlp(pred_dim) 
                        created_predictors.add(predictor_name_1)
                        created_predictors.add(predictor_name_2)
        
                        print(f"Predictor [{mod1} → {mod2}]: {pred_dim}")
                        print(f"Predictor [{mod2} → {mod1}]: {pred_dim}")

    def to_device(self, device):
        """
        Move all model parameters and submodules to the specified device.

        Parameters
        ----------
        device : str
            Target device identifier (e.g., `'cuda:0'`, `'cpu'`).

        Returns
        -------
        None
        """

        self.to(device)  
        for name, module in self.autoencoders.items():
            module.to(device)
        for name, module in self.predictors.items():
            module.to(device)
        print(f"Model moved to {device}!")




    def train_model(self, file_path, config, optimizer, device, feature_dict, spatial_loc_dict, data_dict, Linkage_indicator, num_hvg=3000, n_x=1, n_y=1):
        """
        Train the COSIE model on spatial multimodal data.

        Supports both full-graph and subgraph-level training. The training mode is selected
        based on the `n_x` and `n_y` values.

        Parameters
        ----------
        file_path : str
            Directory path where final embeddings will be saved as `.npy` files.
        
        config : dict
            Configuration dictionary defining model and training hyperparameters.
        
        optimizer : torch.optim.Optimizer
            Optimizer for updating model parameters.
        
        device : str
            Device identifier (e.g., `'cuda:0'`, `'cpu'`).
        
        feature_dict : dict
            A dictionary mapping each section name (e.g., `'s1'`, `'s2'`) to a sub-dictionary of processed feature tensors for each modality. Each feature is a `torch.FloatTensor`.
        
        spatial_loc_dict : dict
            A dictionary mapping each section name to a 2D NumPy array of spatial coordinates.
        
        data_dict : dict
            A dictionary mapping each modality name (e.g., `'RNA'`, `'Protein'`) to a list of AnnData objects, one per tissue section. Each AnnData should contain `.X`, `.obs`, `.var`, and `.obsm['spatial']`. If a modality is missing from a section, use `None` as a placeholder in the list.
        
        Linkage_indicator : dict
            A dictionary specifying which tissue section pairs and modality pairs should be linked.
            Format:
            
            {("s1", "s2"): [("RNA", "RNA"), ("RNA", "Protein")],("s2", "s3"): [("ATAC", "RNA")]}
            
            means: constructing linkage between section s1 and s2 using both RNA-RNA strong linkage and RNA-Protein weak linkage; constructing linkage between section s2 and s3 using ATAC-RNA linkage.

        num_hvg : int, optional
            Number of highly variable features to retain for feature matching during linkage construction. Default is 3000.
        
        n_x : int, optional
            Number of spatial partitions along the x-axis per section. Default is 1. If set to 1, no splitting is applied.
        
        n_y : int, optional
            Number of spatial partitions along the y-axis per section. Default is 1. If set to 1, no splitting is applied. When both `n_x = 1` and `n_y = 1` (default), the model runs in full-graph training mode. When either `n_x` or `n_y` is greater than 1, the model will switch to subgraph training mode and partitions each section into a grid of `n_x × n_y` subregions.

        Returns
        -------
        final_embeddings : dict
            Dictionary mapping section names to their final learned embedding matrices as NumPy arrays.
            These embeddings are also saved to `{file_path}/s1_embedding.npy`, etc.
            Format:
            
            {'s1': np.ndarray of shape (n1_cells, latent_dim), 's2': np.ndarray of shape (n2_cells, latent_dim), ...}
            
        """
        

        if n_x == 1 and n_y == 1:
            print("-------- Running Full-graph training mode --------")
            linkage_results = compute_linkages(data_dict, Linkage_indicator,  num_hvg=num_hvg)

                
            self.to_device(device)
    
            k_neighs = config['training']['knn_neighbors_spatial'] 
            k_neighs_feature = config['training']['knn_neighbors_feature'] 
            complete_graph = {}
            spatial_graph = {}
            print('-------- Construction of input graphs --------')
            for section, modalities in feature_dict.items():
                complete_graph[section] = {}
        
                
                print('-------- Constructing spatial graph for {} --------'.format(section))
                spatial_knn = compute_knn_graph(spatial_loc_dict[section], k_neighs).to(device)
                spatial_graph[section] = spatial_knn
        
                for modality, features in modalities.items():
                    print(f"Constructing feature graph for [{section} - {modality}]...")
        

                    feature_knn = compute_knn_graph(features, k_neighs_feature).to(device)
        
                    combined_knn = torch.cat([spatial_knn, feature_knn], dim=1)
                    # print(combined_knn)
        
                    complete_graph[section][modality] = combined_knn
    
            for section, modalities in feature_dict.items():
                for modality, features in modalities.items():
                    feature_dict[section][modality] = features.to(device)  
            
            print("Training started!")
        
        
            
            for epoch in tqdm(range(config['training']['epoch']), desc="Training Epochs"):
                optimizer.zero_grad()  
                
                total_reconstruction_loss = 0.
                total_contrastive_loss = 0.
                total_prediction_loss = 0.
                total_triplet_loss = 0.
                embedding_dict = {}  
                
                for section, modalities in feature_dict.items():
                    embeddings = {}  
                            
                    for modality, features in modalities.items():

                        encoder = self.autoencoders[modality]  
                        graph = complete_graph[section][modality]  
                        

                        z = encoder.encoder(features, graph)
                        embeddings[modality] = z  
        
                        reconstructed = encoder.decoder(z, graph)
                        recon_loss = F.mse_loss(reconstructed, features)  
                        total_reconstruction_loss += recon_loss
                    
                    embedding_dict[section] = embeddings
        
                    modality_list = list(modalities.keys())  
                    for i, mod1 in enumerate(modality_list):
                        for mod2 in modality_list[i + 1:]:  
                            
                            contrastive_loss = crossview_contrastive_Loss(embeddings[mod1], embeddings[mod2], config['training']['gamma'])
                            total_contrastive_loss += contrastive_loss
        
                            if epoch >= config['training']['start_dual_prediction']:
                                pred_mod1_to_mod2 = self.predictors[f"{mod1}_to_{mod2}"](embeddings[mod1])
                                pred_mod2_to_mod1 = self.predictors[f"{mod2}_to_{mod1}"](embeddings[mod2])
                                prediction_loss = F.mse_loss(pred_mod1_to_mod2, embeddings[mod2]) + F.mse_loss(pred_mod2_to_mod1, embeddings[mod1])
                                total_prediction_loss += prediction_loss
        
                
                if epoch >= config['training']['start_cross_section_integration']:
                
                    final_embedding_dict = {}  
                
                    
                
                    for section, embeddings in embedding_dict.items():
                
                        recovered_embeddings = {}  
                
                        for mod in self.all_modalities:
                            if mod in embeddings:
                                recovered_embeddings[mod] = embeddings[mod]  
                            else:
                                candidate_embeddings = []  
                
                                for src_mod in embeddings.keys():
                                    predictor_key = f"{src_mod}_to_{mod}"
                                    if predictor_key in self.predictors:
                                        candidate_embeddings.append(self.predictors[predictor_key](embeddings[src_mod]))
                
                                if len(candidate_embeddings) > 1:
                                    recovered_embedding = torch.mean(torch.stack(candidate_embeddings), dim=0)  
                                elif len(candidate_embeddings) == 1:
                                    recovered_embedding = candidate_embeddings[0]
                                else:
                                    print(f"No valid predictor found to recover [{mod}] in Section [{section}], using zero tensor!")
                                    if len(embeddings) > 0:
                                        sample_embedding = next(iter(embeddings.values()))  
                                        recovered_embedding = torch.zeros_like(sample_embedding)  
                                    else:
                                        raise Val
                
                                recovered_embeddings[mod] = recovered_embedding
                
                        concatenated_embedding = torch.cat([recovered_embeddings[mod] for mod in self.all_modalities], dim=1)
                        final_embedding_dict[section] = concatenated_embedding
                
                    
    
    
                    for section_pair, triplet_data in linkage_results.items():
                        sec1, sec2 = section_pair.split("_")  
                
                        neighborhood_s1 = compute_neighborhood_embedding(spatial_graph[sec1], final_embedding_dict[sec1], device)
                        neighborhood_s2 = compute_neighborhood_embedding(spatial_graph[sec2], final_embedding_dict[sec2], device)
                
                        neighborhood_embedding_matrix = torch.cat([neighborhood_s1, neighborhood_s2], dim=0)
                
                        anchor_arr = neighborhood_embedding_matrix[triplet_data[:, 0]]
                        positive_arr = neighborhood_embedding_matrix[triplet_data[:, 1]]
                        negative_arr = neighborhood_embedding_matrix[triplet_data[:, 2]]
                
                        triplet_loss = self.triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
                
                
                        total_triplet_loss += triplet_loss
            
    
                

    
                loss = total_contrastive_loss + total_reconstruction_loss * config['training']['lambda1']
                if epoch >= config['training']['start_dual_prediction']:
                    loss += total_prediction_loss * config['training']['lambda2']
                if epoch >= config['training']['start_cross_section_integration']:
                    loss += total_triplet_loss * config['training']['lambda3']
        
                loss.backward()
                optimizer.step()
        
    
            print("Running Evaluation...")
        
                
            for module in self.modules():
                module.eval()
        
            
            with torch.no_grad():
                final_embeddings = {}
        
                for section, modalities in feature_dict.items():
                    embeddings = {}
        
                    for modality, features in modalities.items():
                        encoder = self.autoencoders[modality]
                        graph = complete_graph[section][modality]
                        z = encoder.encoder(features, graph)
                        embeddings[modality] = z  

        
                    recovered_embeddings = {}
                    for mod in self.all_modalities:
                        if mod in embeddings:
                            recovered_embeddings[mod] = embeddings[mod] 
                        else:
                            print(f"Missing modality [{mod}] in Section [{section}]")
                            candidate_embeddings = [] 
            
                            for src_mod in embeddings.keys():
                                predictor_key = f"{src_mod}_to_{mod}"
                                if predictor_key in self.predictors:
                                    print(f"Using predictor [{src_mod} → {mod}] to recover missing embedding...")
                                    candidate_embeddings.append(self.predictors[predictor_key](embeddings[src_mod]))
            
                            if len(candidate_embeddings) > 1:
                                recovered_embedding = torch.mean(torch.stack(candidate_embeddings), dim=0)  
                            elif len(candidate_embeddings) == 1:
                                recovered_embedding = candidate_embeddings[0]
                            else:
                                print(f"No valid predictor found to recover [{mod}] in Section [{section}], using zero tensor!")
                                if len(embeddings) > 0:
                                    sample_embedding = next(iter(embeddings.values())) 
                                    recovered_embedding = torch.zeros_like(sample_embedding)  
                                else:
                                    raise Val
                            recovered_embeddings[mod] = recovered_embedding
    
    

    
                    concatenated_embedding = torch.cat([recovered_embeddings[mod] for mod in self.all_modalities], dim=1)
                    # print('calculate neighborhood {}'.format(section))
                    neighborhood_embedding = compute_neighborhood_embedding(spatial_graph[section], concatenated_embedding, device)
                    bi_embedding = (concatenated_embedding+neighborhood_embedding)*0.5
                    bi_embedding_numpy = bi_embedding.cpu().numpy()
    
                    # If used metacell, reverse to original cell-level
                    section_idx = int(section[1:]) - 1 
                    for modality, adata_list in data_dict.items():
                        if section_idx < len(adata_list) and adata_list[section_idx] is not None:
                            adata_tmp = adata_list[section_idx]
                            if 'meta_to_original' in adata_tmp.uns and 'original_cell_num' in adata_tmp.uns:
                                print(f"Mapping metacell embedding back to original cells for Section {section} using modality [{modality}]")
                                bi_embedding_numpy = reconstruct_metacell_to_original(adata_tmp, bi_embedding_numpy)
                            break  
                    
                    final_embeddings[section] = bi_embedding_numpy
                    np.save(os.path.join(file_path, f"{section}_embedding.npy"), bi_embedding_numpy)
                    
                print("All embeddings have been saved to {}".format(file_path))
        
            return final_embeddings

        else:
            print(f"-------- Running Sub-graph training mode, n_x is {n_x}, n_y is {n_y} --------")
            new_feature_dict, new_spatial_loc_dict, new_linkage_results = preprocess_data_for_subgraphs(data_dict,
                feature_dict, spatial_loc_dict, Linkage_indicator, n_x=n_x, n_y=n_y, num_hvg=num_hvg)
    


            self.to_device(device)
    
    
            k_neighs = config['training']['knn_neighbors_spatial'] 
            k_neighs_feature = config['training']['knn_neighbors_feature']  
            complete_graph = {}
            spatial_graph = {}
            full_complete_graph = {}  #  Full graph for inference
            full_spatial_graph = {}   #  Full spatial graph for inference
    
            print('---------------- Constructing Full Graph ----------------')
    
            for section, modalities in feature_dict.items():
                print(f"-------- Constructing full spatial graph for {section} --------")
                full_complete_graph[section] = {}
        
                spatial_knn_full = compute_knn_graph(spatial_loc_dict[section], k_neighs).to(device)
                full_spatial_graph[section] = spatial_knn_full 
        
                for modality, features in modalities.items():
                    print(f" Constructing full feature graph for [{section} - {modality}]...")
        
                    feature_knn_full = construct_knn_graph_hnsw(features, k=k_neighs_feature).to(device)
                    combined_knn_full = torch.cat([spatial_knn_full, feature_knn_full], dim=1)
        
                    full_complete_graph[section][modality] = combined_knn_full
    
            print('---------------- Graph Construction in Subgraph Level ----------------')
        
            for section, subgraphs in new_feature_dict.items():
                complete_graph[section] = {}
                spatial_graph[section] = {}
        
                for sub_idx, modalities in subgraphs.items():
                    print(f"-------- Constructing spatial graphs for {section} - Subgraph {sub_idx} --------")
        
                    spatial_knn = compute_knn_graph(new_spatial_loc_dict[section][sub_idx], k_neighs).to(device)
                    spatial_graph[section][sub_idx] = spatial_knn  
        
                    complete_graph[section][sub_idx] = {}
        
                    for modality, features in modalities.items():

        
                        print(f"-------- Constructing feature graph for [{section} - Subgraph {sub_idx} - {modality}] --------")
        
                        feature_knn = construct_knn_graph_hnsw(features, k = k_neighs_feature).to(device)
        
                        combined_knn = torch.cat([spatial_knn, feature_knn], dim=1)
        
                        complete_graph[section][sub_idx][modality] = combined_knn
        
            for section, subgraphs in new_feature_dict.items():
                for sub_idx, modalities in subgraphs.items():
                    for modality, features in modalities.items():

                        new_feature_dict[section][sub_idx][modality] = features.to(device) 
        
            for section, modalities in feature_dict.items():
                for modality, features in modalities.items():
                    feature_dict[section][modality] = features.to(device)  
        

            
            print("Training started!")
            
            
            num_subgraphs = len(next(iter(new_feature_dict.values())))  
            for epoch in tqdm(range(config['training']['epoch']), desc="Training Epochs"):
                optimizer.zero_grad()  
            
                total_reconstruction_loss = 0.
                total_contrastive_loss = 0.
                total_prediction_loss = 0.
                total_triplet_loss = 0.
            
                shuffled_subgraph_indices = {section: random.sample(range(num_subgraphs), num_subgraphs) for section in new_feature_dict}
            
                for batch_idx in range(num_subgraphs):  
                    embedding_dict = {}
            
                    batch_subgraphs = {section: shuffled_subgraph_indices[section][batch_idx] for section in new_feature_dict}

            
                    for section, sub_idx in batch_subgraphs.items():
                        features_dict = new_feature_dict[section][sub_idx]  
                        embeddings = {}
            
                        for modality, features in features_dict.items():
                            encoder = self.autoencoders[modality]  
                            graph = complete_graph[section][sub_idx][modality]  
                            
                            z = encoder.encoder(features, graph)
                            embeddings[modality] = z  
                            
                            reconstructed = encoder.decoder(z, graph)
                            recon_loss = F.mse_loss(reconstructed, features) 
                            total_reconstruction_loss += recon_loss
                        
                        embedding_dict[section] = embeddings  
            
                        modality_list = list(features_dict.keys())  
                        for i, mod1 in enumerate(modality_list):
                            for mod2 in modality_list[i + 1:]:  
                                
                                contrastive_loss = crossview_contrastive_Loss(embeddings[mod1], embeddings[mod2], config['training']['gamma'])
                                total_contrastive_loss += contrastive_loss
            
                                if epoch >= config['training']['start_dual_prediction']:
                                    pred_mod1_to_mod2 = self.predictors[f"{mod1}_to_{mod2}"](embeddings[mod1])
                                    pred_mod2_to_mod1 = self.predictors[f"{mod2}_to_{mod1}"](embeddings[mod2])
                                    prediction_loss = F.mse_loss(pred_mod1_to_mod2, embeddings[mod2]) + F.mse_loss(pred_mod2_to_mod1, embeddings[mod1])
                                    total_prediction_loss += prediction_loss
            
                    if epoch >= config['training']['start_cross_section_integration']:
                        final_embedding_dict = {}
            
                        for section, embeddings in embedding_dict.items():
                            recovered_embeddings = {} 
            
                            for mod in self.all_modalities:
                                if mod in embeddings:
                                    recovered_embeddings[mod] = embeddings[mod]  
                                else:
                                    candidate_embeddings = [] 
                                    for src_mod in embeddings.keys():
                                        predictor_key = f"{src_mod}_to_{mod}"
                                        if predictor_key in self.predictors:
                                            candidate_embeddings.append(self.predictors[predictor_key](embeddings[src_mod]))
            
                                    
                                    if len(candidate_embeddings) > 1:
                                        recovered_embedding = torch.mean(torch.stack(candidate_embeddings), dim=0)  
                                    elif len(candidate_embeddings) == 1:
                                        recovered_embedding = candidate_embeddings[0]
                                    else:
                                        print(f" No valid predictor found to recover [{mod}] in Section [{section}], using zero tensor!")
                                        sample_embedding = next(iter(embeddings.values())) if len(embeddings) > 0 else None
                                        recovered_embedding = torch.zeros_like(sample_embedding) if sample_embedding is not None else None
            
                                    recovered_embeddings[mod] = recovered_embedding
            
                            concatenated_embedding = torch.cat([recovered_embeddings[mod] for mod in self.all_modalities], dim=1)
                            final_embedding_dict[section] = concatenated_embedding
            
                       
                        batch_subgraph_pairs = {(sec1, sub1, sec2, sub2) for sec1, sub1 in batch_subgraphs.items() for sec2, sub2 in batch_subgraphs.items() if sec1 != sec2}
                        
                        for sec1, sub1, sec2, sub2 in batch_subgraph_pairs:
                            if (sec1, sub1, sec2, sub2) in new_linkage_results:
                                triplet_data = new_linkage_results[(sec1, sub1, sec2, sub2)]
                            else:
                                continue  
                            
                        
                            neighborhood_s1 = compute_neighborhood_embedding(spatial_graph[sec1][sub1], final_embedding_dict[sec1], device)
                            neighborhood_s2 = compute_neighborhood_embedding(spatial_graph[sec2][sub2], final_embedding_dict[sec2], device)
                            neighborhood_embedding_matrix = torch.cat([neighborhood_s1, neighborhood_s2], dim=0)
                        
                            anchor_arr = neighborhood_embedding_matrix[triplet_data[:, 0]]
                            positive_arr = neighborhood_embedding_matrix[triplet_data[:, 1]]
                            negative_arr = neighborhood_embedding_matrix[triplet_data[:, 2]]
                        
                            triplet_loss = self.triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
                            total_triplet_loss += triplet_loss
    

                loss = total_contrastive_loss + total_reconstruction_loss * config['training']['lambda1']
                if epoch >= config['training']['start_dual_prediction']:
                    loss += total_prediction_loss * config['training']['lambda2']
                if epoch >= config['training']['start_cross_section_integration']:
                    loss += total_triplet_loss * config['training']['lambda3']
            
                loss.backward()
                optimizer.step()
            
    
    
            print(" Running Evaluation...")
        
                
            for module in self.modules():
                module.eval()
    
            
            with torch.no_grad():
                final_embeddings = {}
        
                for section, modalities in feature_dict.items():
                    embeddings = {}
        
                    for modality, features in modalities.items():
                        encoder = self.autoencoders[modality]
                        graph = full_complete_graph[section][modality]
                        z = encoder.encoder(features, graph)
                        embeddings[modality] = z  

        
                    recovered_embeddings = {}
                    for mod in self.all_modalities:
                        if mod in embeddings:
                            recovered_embeddings[mod] = embeddings[mod]  
                        else:
                            print(f"Missing modality [{mod}] in Section [{section}]")
                            candidate_embeddings = []  
            
                            for src_mod in embeddings.keys():
                                predictor_key = f"{src_mod}_to_{mod}"
                                if predictor_key in self.predictors:
                                    print(f"Using predictor [{src_mod} → {mod}] to recover missing embedding...")
                                    candidate_embeddings.append(self.predictors[predictor_key](embeddings[src_mod]))
            
                            if len(candidate_embeddings) > 1:
                                recovered_embedding = torch.mean(torch.stack(candidate_embeddings), dim=0)  
                            elif len(candidate_embeddings) == 1:
                                recovered_embedding = candidate_embeddings[0]
                            else:
                                print(f"No valid predictor found to recover [{mod}] in Section [{section}], using zero tensor!")
                                if len(embeddings) > 0:
                                    sample_embedding = next(iter(embeddings.values())) 
                                    recovered_embedding = torch.zeros_like(sample_embedding)  
                                else:
                                    raise Val
                            recovered_embeddings[mod] = recovered_embedding
    
    

    
                    concatenated_embedding = torch.cat([recovered_embeddings[mod] for mod in self.all_modalities], dim=1)
                    # print('calculate neighborhood {}'.format(section))
                    neighborhood_embedding = compute_neighborhood_embedding(full_spatial_graph[section], concatenated_embedding, device)
                    bi_embedding = (concatenated_embedding+neighborhood_embedding)*0.5
                    bi_embedding_numpy = bi_embedding.cpu().numpy()
    
                    section_idx = int(section[1:]) - 1  
                    for modality, adata_list in data_dict.items():
                        if section_idx < len(adata_list) and adata_list[section_idx] is not None:
                            adata_tmp = adata_list[section_idx]
                            if 'meta_to_original' in adata_tmp.uns and 'original_cell_num' in adata_tmp.uns:
                                print(f"Mapping metacell embedding back to original cells for Section {section} using modality [{modality}]")
                                bi_embedding_numpy = reconstruct_metacell_to_original(adata_tmp, bi_embedding_numpy)
                            break  
                    
                    final_embeddings[section] = bi_embedding_numpy
                    np.save(os.path.join(file_path, f"{section}_embedding.npy"), bi_embedding_numpy)
                    
                print("All embeddings have been saved to {}".format(file_path))
        
            return final_embeddings

    
        
    

















###