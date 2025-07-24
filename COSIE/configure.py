def get_default_config():

    """
    Returns the hyperparameters configuration dictionary used to initialize and train the COSIE model.

    Returns
    -------
    config : dict
        A dictionary containing the following sections:
        
        - GraphAutoencoder:

            * hidden_dim (list of int): Hidden layer dimensions for the graph autoencoder.

            * activations (str): Activation function used (default: 'relu').

        - Prediction:

            * hidden_dim (list of int): List of hidden layer dimensions used in the dual-prediction module.


        - training:

            * seed (int): Random seed for reproducibility.

            * start_dual_prediction (int): Epoch to start dual-prediction loss.

            * start_cross_section_integration (int): Epoch to start cross-section integration.

            * epoch (int): Total number of training epochs.

            * lr (float): Learning rate for optimizer.

            * gamma (float): Weight for entropy regularization in contrastive loss.

            * lambda1 (float): Weight for contrastive loss.

            * lambda2 (float): Weight for prediction loss.

            * lambda3 (float): Weight for triplet loss.

            * knn_neighbors_spatial (int): Number of neighbors in spatial graph construction.

            * knn_neighbors_feature (int): Number of neighbors in feature graph construction.
            
            * print_num (int): Interval (in epochs) to print training progress.
    """

    return dict(
        Prediction=dict(
            hidden_dim=[512, 512]
        ),
        GraphAutoencoder=dict(
            hidden_dim=[256, 128],
            activations='relu',
        ),
        training=dict(
            seed=8,
            start_dual_prediction=100,
            start_cross_section_integration=200,
            epoch=600,
            lr=1.0e-4,
            gamma=5,
            lambda1=0.1,
            lambda2=0.2,
            lambda3=1.,
            knn_neighbors_spatial=5,
            knn_neighbors_feature=30,
            print_num=50,
        ),
    )


    