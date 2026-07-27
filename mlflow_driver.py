# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import copy
import tempfile
from math import ceil
from pathlib import Path

from torch.utils.data import DataLoader
import mlflow

import config
from config_parameter_grid import get_config_parameter_grid
from model import get_models
from utils.algorithm_utils import *
from utils.data_utils import load_client_trainset_and_testset, assign_data_to_clients_RR_without_replacement, assign_data_to_clients_RR_with_replacement, mix_data_classes
from utils.logging_utils import logging, clear_logs, setup_logging
from utils.noop import NoOpClass
from utils.plotting_utils import loss_matrix_to_df, plot_heatmap, plot_MNIST_digit_reconstruction

import argparse
from metrics_printer import MetricsPrinter
from config_driver import ExperimentDriver

from utils.dataset_modifier import assign_data


def generate_run_name(config_dict, parameters_to_include):
    """Generates the run name including the specified parameters and their values from the config_dict"""
    averaging_mode = "MA" if config_dict['averaging_mode'] == 'model_averaging' else "GA"
    frctca = "eb" if config_dict['first_round_client_to_cluster_assignment'] == "evaluation_based" else "r"

    short_parameter_names = config.long_to_short_parameter_names

    config_dict_short = {}
    for parameter in short_parameter_names.keys():
        if parameter in parameters_to_include:
            if parameter == 'averaging_mode':
                config_dict_short['am'] = averaging_mode
            elif parameter == 'first_round_client_to_cluster_assignment':
                config_dict_short['frctca'] = frctca
            else:
                config_dict_short[short_parameter_names[parameter]] = config_dict[parameter]

    run_name = ", ".join([f"{parameter}={config_dict_short[parameter]}" for parameter in config_dict_short.keys()])

    return run_name


def main(exp_ID, base_path, exp_drv_dir):
    config_dict = config.get_config_dict()

    # Load the grid over all parameters combinations of which the experiment will be run
    config_parameter_grid = get_config_parameter_grid()

    # For running individual experiments
    # Sets the config parameters for experiment
    if exp_ID is not None:
        exp_driver = ExperimentDriver(config_dict, config_parameter_grid, base_path, exp_drv_dir)
        exp_driver.set_config_params_for_experiment(exp_ID)
        config_dict = exp_driver.config_dict
        config_parameter_grid = exp_driver.config_parameter_grid

    # Setup logging now since the log file name may have been changed by exp_driver
    setup_logging(config_dict)
    logger = logging.getLogger(__name__)

    if config_dict['mlflow_enabled'] is True:
        mlflow_ = mlflow
        mlflow_.set_tracking_uri(config_dict['mlflow_tracking_uri'])
        mlflow_.set_experiment(config_dict['mlflow_experiment_name'])
        if config_dict['mlflow_log_system_metrics'] is True:
            mlflow_.enable_system_metrics_logging()
            mlflow_.system_metrics.set_system_metrics_sampling_interval(1)   # interval in seconds
        else:
            mlflow_.disable_system_metrics_logging()
    else:
        # Use MLFlow_Noop if there is no connectivity or we do not want to log to an mlflow remote server
        mlflow_ = NoOpClass(logger=logger)


    with mlflow_.start_run():
        varying_config_parameter_names = [key for key, value in config_parameter_grid.items() if len(value) > 1]
        keys_with_only_one_value = list(set(config_parameter_grid.keys()) - set(varying_config_parameter_names))
        # Override those config_dict parameters with only one value in the config_parameter_grid
        for key in keys_with_only_one_value:
            config_dict[key] = config_parameter_grid[key][0]
        common_parameters_for_all_runs = list(set(config_dict.keys()) - set(varying_config_parameter_names))
        # Filter out from the grid those parameters with only one value
        if len(varying_config_parameter_names) > 0:
            config_parameter_grid = {key: value for key, value in config_parameter_grid.items() if key in varying_config_parameter_names}
        # Make 'seed' the first key in the dictionary, so that the loop is over 'seed' first
        if 'seed' in config_parameter_grid.keys():
            config_parameter_grid = {'seed': config_parameter_grid.pop('seed'), **config_parameter_grid}  # new order
            varying_config_parameter_names = config_parameter_grid.keys()
        run_name = generate_run_name(config_dict, parameters_to_include=common_parameters_for_all_runs)
        mlflow_.set_tag("mlflow.runName", run_name)

        previous_dataset = None
        previous_number_of_clients = None
        previous_number_of_data_classes = None
        previous_number_of_clusters = None
        previous_model_type = None
        previous_seed = None

        for varying_config_parameter_values in itertools.product(*config_parameter_grid.values()):
            # Create a copy of the config_dict that is specific to this run and apply configuration parameters from parameter grid
            config_dict_of_run = copy.deepcopy(config_dict)
            for index, config_parameter_name in enumerate(varying_config_parameter_names):
                config_dict_of_run[config_parameter_name] = varying_config_parameter_values[index]

            if config_dict_of_run['data_mixture_mode'] in ['zipped_no_overlap', 'zipped_with_overlap']:
                config_dict_of_run['number_of_clusters'] = ceil(config_dict_of_run['number_of_data_classes'] / config_dict_of_run['data_class_merge_factor'])
            else:
                config_dict_of_run['number_of_clusters'] = config_dict_of_run['number_of_data_classes']

            # Set number_of_clients and number_of_models, as they will be used by functions later
            config_dict_of_run['number_of_clients'] = config_dict_of_run['number_of_clusters'] * config_dict_of_run['number_of_clients_per_cluster']

            # Check that if 'dataset' is MNIST, the right number of MNIST digits have been given
            if (config_dict_of_run['dataset'] == 'MNIST') and (config_dict_of_run['number_of_data_classes'] != len(config_dict_of_run['MNIST_all_digits'])):
                raise ValueError(f"Given number of data classes ({config_dict_of_run['number_of_data_classes']}) does not match number of given MNIST digits ({len(config_dict_of_run['MNIST_all_digits'])}).")

            if config_dict['dataset'] == 'LINEAR' and config_dict['model_type'] != 'linear_regression':
                raise NotImplementedError("For now, LINEAR dataset can only be used when the model is set to linear_regression.")

            if config_dict['model_type'] == 'linear_regression' and config_dict['dataset'] != 'LINEAR':
                raise NotImplementedError("For now, LINEAR dataset can only be used when the model is set to linear_regression.")

            with mlflow_.start_run(nested=True):
                subrun_name = generate_run_name(config_dict_of_run, parameters_to_include=varying_config_parameter_names) if len(varying_config_parameter_names) > 0 else "single-run"
                mlflow_.set_tag("mlflow.runName", subrun_name)

                # Log the hyperparameters in mlflow (doing this early, so that the information is there in case the run fails)
                for key, value in config_dict_of_run.items():
                    mlflow_.log_param(key, value)

                logger.info("")
                logger.info("---------------------------------------")
                logger.info("")
                logger.info("Starting run with " + ",".join([f"{name}={value}" for name, value in zip(varying_config_parameter_names, varying_config_parameter_values)]) if len(varying_config_parameter_names) > 0 else "Starting single run")
                logger.info("")
                logger.info(f"Number of clients = {config_dict_of_run['number_of_clients']}")
                logger.info(f"Number of data classes = {config_dict_of_run['number_of_data_classes']}")
                logger.info(f"Number of clusters = {config_dict_of_run['number_of_clusters']}")
                # logger.info(f"Number of models for clustered training = {config_dict_of_run['number_of_models']}")
                logger.info(f"Number of clients per cluster = {config_dict_of_run['number_of_clients_per_cluster']}")
                logger.info(f"Number of datapoints per client = {config_dict_of_run['number_of_datapoints_per_client']}")
                logger.info(f"Model = {config_dict['model_type']}, Dataset = {config_dict['dataset']}")

                # Set randomness seeds
                random.seed(config_dict_of_run['seed'])
                np.random.seed(config_dict_of_run['seed'])
                torch.manual_seed(config_dict_of_run['seed'])
                torch.cuda.manual_seed_all(config_dict_of_run['seed'])
                torch.use_deterministic_algorithms(True)

                if (config_dict_of_run['dataset'] != previous_dataset) \
                    or (config_dict_of_run['dataset'] == 'LINEAR'):       # else, no need to reload the dataset
                    train_data_unmixed, test_data_unmixed = load_client_trainset_and_testset(config_dict_of_run)

                    train_data, test_data, new_to_original_data_class_labels_mapping_dict, client_dist_cfg = mix_data_classes(train_data_unmixed, test_data_unmixed, config_dict_of_run)
                
                config_dict_of_run['number_of_clusters'] = len(train_data.keys())
                # config_dict_of_run['number_of_models'] = len(train_data.keys())
                if config_dict_of_run['K_is_unknown']:
                    config_dict_of_run['total_number_of_models'] = config_dict_of_run['upper_bound_on_K']
                    config_dict_of_run['number_of_models'] = config_dict_of_run['total_number_of_models']  # This is the current number of models at each round, initialized at the total, but will be varied later.
                else:
                    config_dict_of_run['total_number_of_models'] = len(train_data.keys())
                    config_dict_of_run['number_of_models'] = config_dict_of_run['total_number_of_models']
                config_dict_of_run['number_of_clients'] = config_dict_of_run['number_of_clusters'] * config_dict_of_run['number_of_clients_per_cluster']
                logger.info(f"After update: New Number of clusters = {config_dict_of_run['number_of_clusters']}")
                logger.info(f"After update: New Number of models for clustered training = {config_dict_of_run['number_of_models']}")
                logger.info(f"After update: New Number of clients = {config_dict_of_run['number_of_clients']}")

                if ((config_dict_of_run['number_of_clients'] != previous_number_of_clients) or
                    (config_dict_of_run['number_of_data_classes'] != previous_number_of_data_classes) or
                    (config_dict_of_run['number_of_clusters'] != previous_number_of_clusters) or
                    (config_dict_of_run['model_type'] != previous_model_type) or
                    (config_dict_of_run['seed'] != previous_seed)):    # else, no need to redistribute dataset to clients or create/set new models
                    logger.info("")
                    logger.info("Training data points assignment to clients:")
                    if client_dist_cfg is None:
                        if config_dict_of_run['assign_datapoints_to_clients_with_replacement'] is True:
                            client_trainsets, client_to_cluster_mapping_train = assign_data_to_clients_RR_with_replacement(data=train_data, config_dict=config_dict_of_run)
                        else:
                            client_trainsets, client_to_cluster_mapping_train = assign_data_to_clients_RR_without_replacement(data=train_data, config_dict=config_dict_of_run)
                    else:
                        client_trainsets, client_to_cluster_mapping_train = assign_data(train_data, config_dict_of_run, client_dist_cfg, with_replacement=config_dict_of_run['assign_datapoints_to_clients_with_replacement'])
                    config_dict_of_run['number_of_clients'] = len(client_trainsets)
                    logger.info(f"After last update: New Number of clients = {config_dict_of_run['number_of_clients']}")     
                    trainloaders = [DataLoader(dataset=client_trainsets[client_index], batch_size=config_dict_of_run['local_batch_size'], shuffle=True) for client_index in range(config_dict_of_run['number_of_clients'])]
                    logger.info(f"Ground Truth Client to Cluster: {client_to_cluster_mapping_train}")

                    if (config_dict_of_run['algorithm'] == 'centralized') or (('algorithm' in config_parameter_grid.keys()) and ('centralized' in config_parameter_grid['algorithm'])):
                        # Create a trainloader with all the data for the centralized case
                        rng = np.random.default_rng(config_dict_of_run['seed'])
                        all_train_data = list(itertools.chain(*client_trainsets))
                        rng.shuffle(all_train_data)
                        all_data_trainloader = DataLoader(dataset=all_train_data, batch_size=config_dict_of_run['local_batch_size'], shuffle=True)

                    logger.info("")
                    logger.info("Test data point assignment to clients:")
                    if client_dist_cfg is None:
                        if config_dict_of_run['assign_datapoints_to_clients_with_replacement'] is True:
                            client_testsets, client_to_cluster_mapping_test = assign_data_to_clients_RR_with_replacement(data=test_data, config_dict=config_dict_of_run)
                        else:
                            client_testsets, client_to_cluster_mapping_test = assign_data_to_clients_RR_without_replacement(data=test_data, config_dict=config_dict_of_run)
                    else:
                        client_testsets, client_to_cluster_mapping_test = assign_data(test_data, config_dict_of_run, client_dist_cfg, with_replacement=config_dict_of_run['assign_datapoints_to_clients_with_replacement'])
                    testloaders = [DataLoader(dataset=client_testsets[client_index], batch_size=config_dict_of_run['local_batch_size'], shuffle=False) for client_index in range(config_dict_of_run['number_of_clients'])]

                previous_dataset = config_dict_of_run['dataset']
                previous_number_of_clients = config_dict_of_run['number_of_clients']
                previous_number_of_data_classes = config_dict_of_run['number_of_data_classes']
                previous_number_of_clusters = config_dict_of_run['number_of_clusters']
                previous_model_type = config_dict_of_run['model_type']
                previous_seed = config_dict_of_run['seed']

                data_point_dimension = prod(trainloaders[0].dataset[0][0].shape)
                config_dict_of_run['autoencoder_input_size'] = data_point_dimension
                config_dict_of_run['autoencoder_middle_layer_size'] = data_point_dimension // 4     # The maximum middle layer size allowed by our autoencoder model definition

                # Do the training depending on the algorithm
                logger.info("")
                logger.info(f"==== Algorithm: {config_dict_of_run['algorithm']} ====")
                logger.info(f"Model = {config_dict['model_type']}, Dataset = {config_dict['dataset']}")
                # Determine the actual number of models based on the algorithm
                if config_dict_of_run['algorithm'] == 'centralized':
                    # Only 1 model
                    config_dict_of_run['number_of_models'] = 1
                elif config_dict_of_run['algorithm'] == 'local':
                    # As many models as the clients
                    config_dict_of_run['number_of_models'] = config_dict_of_run['number_of_clients']
                elif config_dict_of_run['algorithm'] in ['IFCA', 'CLoVE', 'vanillaFL']:
                    # If vanillaFL, only one model will be aggregated; else, number of models as defined in the original config_dict
                    if config_dict_of_run['algorithm'] == 'vanillaFL':
                        config_dict_of_run['number_of_models'] = 1
                    else:
                        pass
                else:
                    raise ValueError(f"Invalid value given for config_dict['algorithm']: {config_dict_of_run['algorithm']}")

                # Create as many models as the actual number of models for this run
                nets = get_models(config_dict_of_run)

                if config_dict_of_run['dataset'] == 'LINEAR':
                    linear_class = config_dict_of_run['cur_linear_class_obj']
                    linear_class.print_ground_truth(config_dict_of_run, train_data=train_data)

                # Determine the initial parameters for the models (their weights from the net creation above will be ignored and overwritten)
                if config_dict_of_run['initial_model_parameters'] == 'same':
                    all_model_parameters_initial = {model_index: get_single_model_parameters(nets[0]) for model_index in range(len(nets))}  # if same, copy the weights of the first model to all models
                elif config_dict_of_run['initial_model_parameters'] == 'different':
                    all_model_parameters_initial = {model_index: get_single_model_parameters(nets[model_index]) for model_index in range(len(nets))}
                else:
                    raise ValueError(f"Invalid value given for config_dict['initial_model_parameters']: {config_dict_of_run['initial_model_parameters']}")

                # Run the appropriate training function depending on the algorithm
                if config_dict_of_run['algorithm'] == 'centralized':
                    all_model_parameters, client_to_selected_model_mapping_dict, selected_model_to_client_mapping_dict, metrics_per_round = centralized_only_training_function(
                        config_dict_of_run,
                        nets[0],
                        all_data_trainloader,
                        testloaders,
                        all_model_parameters_initial,
                        ground_truth=client_to_cluster_mapping_train
                    )
                elif config_dict_of_run['algorithm'] == 'local':
                    all_model_parameters, client_to_selected_model_mapping_dict, selected_model_to_client_mapping_dict, metrics_per_round = local_only_training_function(
                        config_dict_of_run,
                        nets,
                        trainloaders,
                        testloaders,
                        all_model_parameters_initial,
                        ground_truth=client_to_cluster_mapping_train
                    )
                else:       # if config_dict_of_run['algorithm'] in ['CLoVE', 'IFCA', 'vanillaFL']
                    all_model_parameters, client_to_selected_model_mapping_dict, selected_model_to_client_mapping_dict, metrics_per_round = clustered_federated_training_function(
                        config_dict_of_run,
                        nets,
                        trainloaders,
                        testloaders,
                        all_model_parameters_initial,
                        ground_truth=client_to_cluster_mapping_train
                    )

                # Log the metrics for all rounds in a text file if running experiments individually
                metrics_printer = MetricsPrinter(subrun_name, metrics_per_round, base_path, config_dict)
                metrics_printer.print_metrics()

                ## Log everything on mlflow (if it is enabled)

                if config_dict_of_run['mlflow_enabled'] is True:
                    # Generate the loss matrix of the data of each client (row) tested on each model (column), and plot the heatmap
                    loss_vectors_dict = generate_loss_vectors(config_dict_of_run, all_model_parameters, testloaders, list(range(config_dict_of_run['number_of_clients'])))
                    loss_matrix = np.array(list(loss_vectors_dict.values()))
                    losses_df = loss_matrix_to_df(loss_matrix)
                    logger.info(losses_df)
                    heatmap = plot_heatmap(loss_matrix)
                    # plt.show()
                    mlflow_.log_table(data=losses_df, artifact_file="final_loss_matrix.json")

                    # Plot reconstruction figures for (FE)MNIST
                    reconstruction_figs = []
                    if config_dict_of_run['dataset'] in ['MNIST', 'FEMNIST']:
                        reconstruction_figs = plot_MNIST_digit_reconstruction(config_dict_of_run, testloaders, all_model_parameters)

                    # Log the metrics for all rounds, directly in mlflow if the metric is a scalar, or in an artifact for the non-scalar metrics
                    non_scalar_metrics = set()
                    for metric in metrics_per_round.keys():
                        for step in range(len(metrics_per_round[metric])):
                            if np.isscalar(metrics_per_round[metric][step]):    # if a scalar metric; the metrics that do not satisfy this, such as the loss matrix, will be saved as an artifact below
                                mlflow_.log_metric(key=metric, value=metrics_per_round[metric][step], step=step)
                            else:
                                non_scalar_metrics = non_scalar_metrics.union({metric})

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        path = Path(tmp_dir, "non_scalar_metrics_per_round.txt")
                        path.write_text(repr({metric: metrics_per_round[metric] for metric in non_scalar_metrics}))
                        mlflow_.log_artifact(path.__str__())

                    # Log all the final model parameters for reproducibility
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        all_model_parameters_as_lists = {}
                        for model_index in range(len(all_model_parameters)):
                            all_model_parameters_as_lists[model_index] = [
                                layer_parameters.tolist() \
                                for layer_parameters in all_model_parameters[model_index]
                            ]
                        path = Path(tmp_dir, "all_model_parameters.txt")
                        path.write_text(repr(all_model_parameters_as_lists))
                        mlflow_.log_artifact(path.__str__())

                    # Log all the final models for reproducibility
                    for model_index in range(config_dict_of_run["number_of_models"]):
                        set_single_model_parameters(nets[model_index], all_model_parameters[model_index])

                        input_example = next(iter(trainloaders[0]))[0].to(torch.float32)
                        if config_dict_of_run['dataset'] in ['AmazonReview', 'AG_news']:  # text-based datasets
                            pass
                        else:  # image-based datasets
                            input_example = input_example.to(torch.float32)
                        if config_dict_of_run['dataset'] == 'CIFAR10' and config_dict_of_run['model_type'] == 'AE':
                            input_example = input_example.view(input_example.size(0), -1)  # Flatten input_example to feed to autoencoder
                        output_example = nets[model_index](input_example)
                        input_example_np = input_example.detach().numpy()
                        output_example_np = output_example.detach().numpy()
                        signature = mlflow_.models.infer_signature(input_example_np, output_example_np)

                        # mlflow_.pytorch.log_model(nets[model_index], f"model_{model_index}")
                        mlflow_.pytorch.log_model(
                            pytorch_model=nets[model_index],
                            artifact_path=f"model_{model_index}",
                            signature=signature
                        )
                        # mlflow_.pytorch.log_model(nets[model_index], f"model_{model_index}", input_example=next(iter(trainloaders[0]))[0].to(torch.float32).numpy())

                    # Log heatmap, and images and their reconstructions in case of (FE)MNIST
                    mlflow_.log_figure(heatmap, "heatmap.png")
                    for i, fig in enumerate(reconstruction_figs):
                        # plt.show()
                        mlflow_.log_figure(fig, f"digit_reconstruction_{i}.png")

        logger.info("")
        logger.info("Execution completed successfully.")
        if config_dict_of_run['mlflow_enabled'] is True:
            mlflow_.log_artifact(config_dict['log_file'])
        clear_logs(config_dict['log_file'])     # Delete log file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLoVE experiments.")

    parser.add_argument(
        "--id", 
        type=int, 
        help="ID of experiment that we want to run"
    )

    parser.add_argument(
        "--path", 
        default='.',
        type=str,
        help="Base path for experiment"
    )

    parser.add_argument(
        "--cdir", 
        default='experiment_driver',
        type=str, 
        help="Sub directory where experiment driver file is kept"
    )
    # Parse the arguments
    args = parser.parse_args()

    exp_ID = None
    if args.id:
        exp_ID = args.id

    base_path = None
    if args.path:
        base_path = args.path    

    exp_drv_dir = None
    if args.cdir:
        exp_drv_dir = args.cdir

    if exp_ID is not None:
        print(f'Args: id = {exp_ID}, base_path = {base_path}, exp_drv_dir = {exp_drv_dir}')
    else:
        raise ValueError("exp_ID was not provided.")

    main(exp_ID, base_path, exp_drv_dir)
