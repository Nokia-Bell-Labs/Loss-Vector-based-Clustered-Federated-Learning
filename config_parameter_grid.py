# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

# Define the grid over all parameters combinations of which the experiment will be run
# CAUTION: Allowed to vary all parameters except 'dataset'!

def get_config_parameter_grid():

    config_parameter_grid = {
        #### Environment configuration ####
        # 'number_of_data_classes': [],
        # 'number_of_clients_per_cluster': [1,2],
        # 'number_of_datapoints_per_client': [],

        #### Algorithm ####
        # 'algorithm': ['centralized', 'local', 'vanillaFL', 'IFCA', 'CLoVE'],
        # 'algorithm': ['vanillaFL', 'IFCA', 'CLoVE'],
        # 'algorithm': ['centralized', 'local', 'CLoVE'],
        # 'algorithm': ['IFCA', 'CLoVE'],
        # 'algorithm': ['centralized'],
        # 'algorithm': ['local'],
        # 'algorithm': ['IFCA'],
        # 'algorithm': ['CLoVE'],
        # 'initial_model_parameters': ['same', 'different'],
        # 'first_round_client_to_cluster_assignment': ['evaluation_based', 'random'],
        # 'cluster_to_model_matching_method': ['sort_clusters_wrt_min_client_ID', 'min_cost_matching_wrt_total_cluster_loss', 'min_cost_matching_wrt_cluster_overlap_with_previous_clusters'],
        # 'averaging_mode': ['model_averaging', 'gradient_averaging'],

        #### Data ###
        # 'dataset': ['MNIST'],
        # 'train_data_percentage': [0.9],
        # 'MNIST_all_digits': [],
        # 'FEMNIST_data_mode': [],
        # 'LINEAR_data_point_dimension': [],
        # 'LINEAR_num_cluster_pnts': [],
        # 'error_sigma': [],
        # 'delta': [],
        # 'theta_length': [],

        #### Model ####
        # 'autoencoder_middle_layer_size': [5, 50, 196],
        # 'autoencoder_middle_layer_size': [196],
        # 'optimizer_type': ["SGD", "Adam"],

        #### Hyperparameters ###
        # 'number_of_rounds': [3],
        # 'local_batch_size': [16,32,64,128],
        # 'local_epochs': [1],
        # 'local_learning_rate': [1e-3],
        # 'softmax_temperature': [1,2,5,10],
        # 'seed': [1, 2],
    }

    return config_parameter_grid
