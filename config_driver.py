# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import config
from config_parameter_grid import get_config_parameter_grid

import pandas as pd
import os
import sys
import ast


# If experiment driver file does not exist then it will exit
# Otherwise it will set config parameters for the experiment whose id is provided
class ExperimentDriver:
    def __init__(self, config_dict, config_parameter_grid, base_path, exp_drv_dir='experiment_driver', drvr_file_name='experiment_list.csv'):
        self.df = None
        self.base_path = base_path
        full_input_file_path = os.path.join(base_path, exp_drv_dir, drvr_file_name)

        if not os.path.exists(full_input_file_path):
            print("There is no driver file so exiting out")
            sys.exit(1)
        else:
            self.df = pd.read_csv(full_input_file_path)

        self.config_parameter_grid = config_parameter_grid
        self.config_dict = config_dict

    def set_config_params_for_experiment(self, experiment_id):
        result_row = self.find_row_by_experiment_id(experiment_id)
        if result_row is not None:
            print(f"Row found for Experiment ID {experiment_id}:\n{result_row}")
            self.set_config_params_for_row(result_row)
        else:
            raise ValueError(f"Experiment ID {experiment_id} was not found in the experiment driver file.")

    def set_config_params_for_row(self, row):
        if self.config_dict is None:
            config_dict = config.get_config_dict()
        else:
            config_dict = self.config_dict

        if self.config_parameter_grid is None:
            config_parameter_grid = get_config_parameter_grid()
        else:
            config_parameter_grid = self.config_parameter_grid

        config_dict['experiment_id'] = "_".join([str(int(item)) for item in parse_number_or_list(row['Experiment ID'])])
        config_dict['experiment_type'] = str(row['Experiment type'])
        config_dict['dataset'] = str(row['Dataset'])
        config_dict['number_of_data_classes'] = int(row['Data classes'])
        if config_dict['dataset'] == 'MNIST':
            config_dict['MNIST_all_digits'] = ast.literal_eval(row['MNIST_all_digits'])
        config_dict['number_of_clients_per_cluster'] = int(row['Clients per cluster'])
        config_dict['number_of_datapoints_per_client'] = int(row['Datapoints per client'])
        config_parameter_grid['algorithm'] = [str(item) for item in parse_string_as_list_of_strings(row['Algorithm'])]
        agg_scheme = parse_string_as_list_of_strings(row['MA vs GA'])
        agg_scheme_full = []
        if 'MA' in agg_scheme:
            agg_scheme_full.append('model_averaging')
        if 'GA' in agg_scheme:
            agg_scheme_full.append('gradient_averaging')
        config_parameter_grid['averaging_mode'] = agg_scheme_full
        config_parameter_grid['number_of_rounds'] = [int(row['Total rounds'])]
        config_dict['initial_model_parameters'] = str(row['initial_model_parameters'])
        config_dict['first_round_client_to_cluster_assignment'] = str(row['first_round_client_to_cluster_assignment'])
        # If running with this mode (expID is not None), then disable mlflow
        config_dict['mlflow_enabled'] = False

        if config_dict['dataset'] == 'LINEAR':
            config_dict['delta'] = float(row['Linear Delta'])

        config_dict['model_type'] = str(row['Model'])
        if config_dict['model_type'] in ['CNN', 'TextCNN', 'AmazonMLP']:
            config_dict['data_mixture_mode'] = str(row['Mix Mode'])
            if config_dict['data_mixture_mode'] == "zipped_with_overlap":
                config_dict['exp_selector'] = str(row['Exp Selector'])

        print("Setting K is unknown.")
        if str(row['K_is_unknown']) in ['True', 'TRUE']:
            config_dict['K_is_unknown'] = True
            if not pd.isna(row['Upper bound on K']):
                config_dict['upper_bound_on_K'] = int(row['Upper bound on K'])
            else:
                ValueError("Wrong config combination: K_is_unknown is True but no upper bound on K has been defined.")
        else:
            config_dict['K_is_unknown'] = False
        print("Setting early stopping.")
        if str(row['Early stopping']) in ['True', 'TRUE']:
            config_dict['early_stopping_of_clustering'] = True
        else:
            config_dict['early_stopping_of_clustering'] = False
        if not pd.isna(row['Participation Rate']):
            config_dict['fraction_fit'] = float(row['Participation Rate'])
            print(f"Setting participation rate to: {config_dict['fraction_fit']}")
        else:
            config_dict['fraction_fit'] = 1.0

        if not pd.isna(row['Ablation_Param']):  # For ablation study
            ablation_param = int(row['Ablation_Param'])
            if ablation_param == 1:     # Not using Matching
                print("Doing ablation:  Not using bipartite matching")
                config_dict['cluster_to_model_matching_method'] = 'sort_clusters_wrt_min_client_ID'     # essentially cluster to model assignment is random
            elif ablation_param == 2:   # Using AgglomerativeClustering instead of KMEANS
                print("Doing ablation:  Using AgglomerativeClustering instead of KMEANS")
                config_dict['clustering_alg'] = 'AgglomerativeClustering'
            elif ablation_param == 3:   # Using Square root of losses
                print("Doing ablation:  Using Square root of losses instead of losses direcly")
                config_dict['loss_transform'] = 'square_root'

        # Hard coded for now as they are not in the csv files
        config_parameter_grid['seed'] = [1, 2, 3]
        if config_dict['dataset'] == 'LINEAR' or config_dict['dataset'] == 'FEMNIST':
            config_dict['local_epochs'] = 25

        if config_dict['model_type'] == 'CNN':
            if config_dict['dataset'] == 'MNIST' or config_dict['dataset'] == 'FMNIST':
                config_dict['local_learning_rate'] = 1e-2
                config_dict['local_epochs'] = 1
            if config_dict['dataset'] == 'CIFAR10':
                config_dict['local_learning_rate'] = 1e-3
                config_dict['local_epochs'] = 1


        # Use a different log file for each experiment so we can run multiple of experiments in parallel
        config_dict['log_file'] = config_dict['experiment_id'] + '_' + config_dict['log_file']
        log_file_dir = os.path.join(self.base_path, config_dict['results_dir'], self.config_dict['experiment_id'])
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)
        config_dict['log_file'] = os.path.join(str(log_file_dir), config_dict['log_file'])

        print(f'config_parameter_grid = {config_parameter_grid}')

    def find_row_by_experiment_id(self, experiment_id):
        """
        Given an Experiment ID, find and return the row where this ID is between the first and last value in the 'Experiment ID' list.
        Assumes that the 'Experiment ID' list contains float values.
        """
        try:
            if self.df is not None:
                for index, row in self.df.iterrows():
                    # Extract the Experiment ID list for the current row
                    experiment_ids_list = parse_number_or_list(row['Experiment ID'])

                    # Ensure the Experiment ID is a list (and check that it's not empty)
                    if isinstance(experiment_ids_list, list) and len(experiment_ids_list) > 0:
                        # Check if the experiment_id is between the first and last value in the list
                        min_id = min(experiment_ids_list)
                        max_id = max(experiment_ids_list)

                        # Check if the experiment_id is between the first and last value in the list
                        if min_id <= experiment_id <= max_id:
                            return row

                print(f"Experiment ID {experiment_id} not found in any row.")
                return None
            else:
                print("Data has not been loaded yet. Please call 'read_file' first.")
                return None
        except ValueError:
            print("Error parsing file.")
            return None


def parse_number_or_list(input_string):
    try:
        # Safely evaluate the string
        result = ast.literal_eval(str(input_string))
        # Check if the result is a number or list of numbers
        if isinstance(result, (int, float)):
            return [result]
        elif isinstance(result, list) and all(isinstance(item, (int, float)) for item in result):
            return result
        else:
            raise ValueError("Input is not a valid number or list of numbers")
    except (ValueError, SyntaxError):
        raise ValueError("Invalid input string")


def parse_str_or_list(input_string):
    try:
        # Safely evaluate the string
        result = ast.literal_eval(str(input_string))
        # Check if the result is a number or list of numbers
        if isinstance(result, str):
            return [result]
        elif isinstance(result, list) and all(isinstance(item, str) for item in result):
            return result
        else:
            raise ValueError("Input is not a valid str or list of strs")
    except (ValueError, SyntaxError):
        raise ValueError("Invalid input string")


def parse_string_as_list_of_strings(input_string):
    """Parses a string containing a comma-separated list of strings.
    Args:
      input_string: The input string.

    Returns:
      A list of strings parsed from the cell value.
    """
    return [item.strip() for item in input_string.split(',')]
