# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

######### Shared config #########
import logging
import torch
from enum import IntFlag
from ml_collections import ConfigDict


class Status(IntFlag):
    NORMAL = 0
    ABNORMAL = 1


def get_config_dict():
    config = ConfigDict()

    #### Environment configuration ####
    config.number_of_data_classes = 3   # number of different data classes; choices: Int >= 1
    config.number_of_clients_per_cluster = 5        # By default, each cluster gets assigned data from one data class, unless specified otherwise through the data_mixture_mode parameter below
    config.number_of_datapoints_per_client = 50

    config.fraction_fit = 1.0    # Percentage of clients that participate at each next round; choices: Float <= 1
    # config.fraction_evaluate = 1.0   # choices: Float <= 1


    #### Algorithm ####
    config.algorithm = 'CLoVE'        # Choices: 'IFCA', 'CLoVE', 'centralized', 'local', 'vanillaFL'
    config.initial_model_parameters = 'different'     # whether the initial parameters of all models will be the same or different (randomly initialized); choices: 'same', 'different'; applied to all algorithms except for 'centralized'
    config.first_round_client_to_cluster_assignment = 'evaluation_based'    # 'evaluation_based' or 'random'
    config.cluster_to_model_matching_method = 'min_cost_matching_wrt_total_cluster_loss'     # Choices: 'sort_clusters_wrt_min_client_ID', 'min_cost_matching_wrt_total_cluster_loss', 'min_cost_matching_wrt_cluster_overlap_with_previous_clusters'; only applies if 'algorithm' == 'CLoVE'
    config.averaging_mode = 'model_averaging'        # Choices: 'model_averaging', 'gradient_averaging'
    config.clustering_alg = 'KMEANS'        # Choices: 'KMEANS', 'AgglomerativeClustering', 'FINCH'
    config.loss_transform = 'identity'      # Choices: 'identity', 'square_root'
    config.K_is_unknown = False   # Applies only for CLoVE. If True, then CLoVE will find the best number of models K up to the upper bound defined below.
    config.upper_bound_on_K = 15
    config.early_stopping_of_clustering = True
    config.number_of_rounds_for_clustering_stability = 3
    config.replicate_model_parameters_at_some_round = False
    config.round_of_replication = 15

    #### Data ###
    # Data mixing
    config.data_mixture_mode = 'no_mixture'  # Choices: 'no_mixture', 'dominant', 'zipped_no_overlap', 'zipped_with_overlap'
    config.percentage_of_points_from_other_data_classes = 0.2   # Applicable only when data_mixture_mode is 'dominant'.
    config.data_class_merge_factor = 2  # How many data classes to merge into a single cluster. Default: 1. Applicable only if data_mixture_mode is 'zipped_no_overlap' or 'zipped_with_overlap'.

    # Data folders
    config.data_root_dir = './data/'
    config.dataset = 'MNIST'             # Choices: 'MNIST', 'CIFAR10', 'FEMNIST', 'FMNIST', 'SYNTHETIC', 'LINEAR', 'AmazonReview', 'AG_news', 'TinyImageNet'
    config.data_dir = {
        'MNIST': config.data_root_dir,
        'CIFAR10': config.data_root_dir,
        'FEMNIST': config.data_root_dir + 'leaf/data/femnist/data/all_data',
        'FMNIST': config.data_root_dir,
        'SYNTHETIC': None,
        'LINEAR': None,
        'AmazonReview': config.data_root_dir,
        'AG_news': None,
        'TinyImageNet': config.data_root_dir + 'tiny-imagenet-200',
    }
    config.train_data_percentage = 0.9      # Does not apply if config.dataset == 'MNIST'
    config.assign_datapoints_to_clients_with_replacement = False     # Choices: True, False; Note that when it is False, if there is not enough distinct datapoints, this serves as the maximum possible number of datapoints per client and some clients might be assigned fewer datapoints

    # Dataset-specific config values
    # For config.dataset == 'MNIST':
    # config.MNIST_all_digits = [0, '0r3', 5]  # Format: mix of integers and strings of type '0r1' (digit 0 rotated 1 time by 90 degrees)
    config.MNIST_all_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # For config.dataset == 'FEMNIST':
    config.FEMNIST_data_mode = 'by_writer'      # Choices: 'by_writer', 'by_character'
    config.FEMNIST_num_data_classes = 62
    # config.FEMNIST_number_of_models = 10

    # For dataset == 'SYNTHETIC':
    config.SYNTHETIC_data_point_dimension = 100
    config.SYNTHETIC_number_of_overlap_basis_vectors = 20
    config.SYNTHETIC_number_of_no_overlap_basis_vectors = 20
    config.SYNTHETIC_overlap_dataset_size = 800
    config.SYNTHETIC_number_of_overlap_points_per_dataset = 100
    config.SYNTHETIC_number_of_no_overlap_points_per_dataset = 20

    # For dataset == 'LINEAR':        # For Linear regression
    config.LINEAR_data_point_dimension = 20
    config.LINEAR_num_cluster_pnts = 5000 # number of points per cluster
    # Linear regression parameters (for data generation)
    config.error_sigma = 0.0001         # Sigma of error in data generation
    config.delta = 1.0                  # Distance between models
    config.theta_length = 1.0           # Length of model theta vector

    # For dataset == 'AG_news'
    config.AG_news_max_len = 200


    #### Model ####
    config.model_type = 'CNN'      # Choices: 'AE', 'CNN', 'linear_regression', 'CIFAR10Autoencoder', 'Cifar10CNN', 'GenConvNet', 'AmazonMLP', 'TextCNN'
    # config.autoencoder_middle_layer_size = 5
    config.autoencoder_output_activation_function = torch.nn.Tanh() if config.dataset == 'SYNTHETIC' else torch.nn.Sigmoid()
    config.criterion = {        # Loss function
        'AE': torch.nn.MSELoss(reduction='mean'),
        'CNN': torch.nn.CrossEntropyLoss(),
        'linear_regression': torch.nn.MSELoss(reduction='mean'),
        'Cifar10CNN': torch.nn.CrossEntropyLoss(),
        'GenConvNet': torch.nn.CrossEntropyLoss(),
        'AmazonMLP': torch.nn.CrossEntropyLoss(),
        'TextCNN': torch.nn.CrossEntropyLoss(),
    }
    config.optimizer_type = "Adam"       # Choices: 'SGD', 'Adam' (extendable if needed)

    #### Hyperparameters ###
    config.number_of_rounds = 3  # number of communication rounds; When doing gradient based averaging, need to use very high number of rounds; for model averaging, a low number of rounds is fine
    config.local_batch_size = 100     # batch size for client training
    config.local_epochs = 1         # number of local epochs for client training; only applies to model averaging; Linear regression needs higher epochs for loss to decrease fast enough
    # if config.averaging_mode == 'gradient_averaging' and config.local_epochs != 1:
    #     print(f"Averaging mode set to gradient averaging. Number of local epochs will be set to 1.")
    #     config.local_epochs = 1
    config.local_learning_rate = 1e-3       # initial learning rate for each's client's model training
    config.lr_schedule_enabled = False   # whether to use a learning rate scheduler
    config.lr_scheduler_step_size = 100     # every how many epochs to apply a step in the learning rate scheduler (a separate count is kept for each client)
    config.lr_scheduler_gamma = 0.1     # multiplicative factor of learning rate decay

    config.seed = 12345 # set random seed

    #### GPU ####
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if config.device.type == "cuda":
    #     config.client_resources = {"num_gpus": 1}
    # else:
    #     config.client_resources = None


    #### Logging ####
    config.log_file = "pfl_experiments.log"
    config.logging_level = logging.INFO     # Choices: logging.(DEBUG, INFO, WARNING, ERROR, CRITICAL)
    config.logging_detail = "simple"    # Choices: "verbose" or "simple"
    config.mlflow_enabled = False
    config.mlflow_experiment_name = "PFL_experiment"
    config.mlflow_tracking_uri = "http://127.0.0.1:5000"
    config.mlflow_log_system_metrics = False
    config.results_dir = "results"     # Results folder for non-mlflow local experiments


    ### Plotting ####
    # config.plot_mode = 'minimal'           # 'all-plots' or 'minimal'
    # config.number_of_digits_in_IvO_plot = 10      # number of digits/pairs of Inputs vs Outputs (IvO) to be shown in the IvO figures

    return config


long_to_short_parameter_names = {
    'dataset':                                      'dataset',
    'MNIST_all_digits':                             'digits',
    'FEMNIST_data_mode':                            'Fdm',
    'number_of_data_classes':                       'dataclasses',
    'number_of_clients_per_cluster':                'cpc',
    'number_of_datapoints_per_client':              'dppc',
    'algorithm':                                    'alg',
    'averaging_mode':                               'am',
    'number_of_rounds': 'rounds',
    'initial_model_parameters':                     'imp',
    'first_round_client_to_cluster_assignment':     'frctca',
    'optimizer_type':                               'ot',
    'local_batch_size':                             'lbs',
    'local_epochs':                                 'le',
    'local_learning_rate':                          'llr',
    # 'cluster_to_model_matching_method':             'ctmmm',
    'train_data_percentage':                        'tts',  # train-test split
    # 'MNIST_all_digits':                             'digits',
    'autoencoder_middle_layer_size':                'AEmls',
    # 'criterion':                                    'crit',
    'seed':                                         'seed',
}
