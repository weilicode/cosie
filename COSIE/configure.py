def get_default_config():

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


    