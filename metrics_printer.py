# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import json
import os


# Function to convert np.int32 (or any np scalar types) to regular int, including keys
def convert_np_types(obj):
    if isinstance(obj, dict):
        # Convert both keys and values
        return {convert_np_types(key): convert_np_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]  # Recursively convert list items
    elif isinstance(obj, np.integer):  # Check if it's a NumPy integer type
        return int(obj)  # Convert np.int32 to regular Python int
    return obj  # Return the object as is if it's not a numpy integer


def write_dict_to_file(obj, file, metric):
    if isinstance(obj, dict):
        # Convert all np.int32 values and keys to Python int before writing
        obj = convert_np_types(obj)
        try:
            json.dump(obj, file, indent=4)  # Write as pretty-formatted JSON
            file.write('\n')
            print(f"Dictionary {metric} successfully written to file.")
        except Exception as e:
            print(f"Error writing metric {metric} to file: {e}")
    else:
        print(f"The metric {metric} is not a dictionary.")


def manage_directory(dir_path):
    # Remove the directory if it exists
    # if os.path.exists(dir_path):
    #     shutil.rmtree(dir_path)  # Delete the directory and its contents
    #     print(f"Directory '{dir_path}' removed.")

    # Create the directory
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")


class MetricsPrinter:
    # def __init__(self, name, metrics, all_model_parameters, base_path, config_dict):
    def __init__(self, name, metrics, base_path, config_dict):
        self.name = name
        self.metrics = metrics
        # self.all_model_parameters = all_model_parameters
        self.config_dict = config_dict
        self.base_path = os.path.join(base_path)
        # self.log_all_model_parameters = False

    def print_metrics(self):
        if isinstance(self.metrics, dict):

            # Define the file name using the provided 'name'
            file_name = f"{self.name}.txt"
            dir_path = os.path.join(self.base_path, self.config_dict['results_dir'], self.config_dict['experiment_id'])
            manage_directory(dir_path)
            file_path = os.path.join(str(dir_path), file_name)

            # Writing the metrics to the file
            with open(file_path, "w") as file:
                file.write(f'key=name, value={self.name}\n')
                file.write(f'key=experiment_id, value={self.config_dict["experiment_id"]}\n')
                file.write(f'key=experiment_type, value={self.config_dict["experiment_type"]}\n')

                non_scalar_metrics = set()
                for metric in self.metrics.keys():
                    for step in range(len(self.metrics[metric])):
                        if np.isscalar(self.metrics[metric][
                                           step]):  # if a scalar metric; the metrics that do not satisfy this, such as the loss matrix, will be saved as an artifact below
                            file.write(f"key={metric}, value={self.metrics[metric][step]}, step={step}\n")
                        else:
                            non_scalar_metrics = non_scalar_metrics.union({metric})

                non_scalar_dict = {}
                for metric in non_scalar_metrics:
                    non_scalar_dict[metric] = self.metrics[metric]
                write_dict_to_file(non_scalar_dict, file, 'Non Scalar Metrics')

                #[write_dict_to_file({metric: self.metrics[metric]}, file, metric) for metric in non_scalar_metrics]
                #[print(self.metrics[metric]) for metric in non_scalar_metrics]
                #
                # # Log config_dict to file
                # config_dict_to_log = dict(self.config_dict)
                # del config_dict_to_log['autoencoder_output_activation_function']
                # del config_dict_to_log['criterion']
                # del config_dict_to_log['device']
                # config_dict_to_log['data_dir'] = config_dict_to_log['data_dir'][config_dict_to_log['dataset']]
                # config_dict_to_log['model_type'] = config_dict_to_log['model_type'][config_dict_to_log['dataset']]
                # write_dict_to_file(config_dict_to_log, file, 'Config Dict')
                #
                # # non_scalar_dict['config_dict'] = config_dict_to_log
                # # write_dict_to_file(non_scalar_dict, file, 'Non Scalar Metrics')
                #
                #
                # # # Log all model parameters to file
                # if self.log_all_model_parameters is True:
                #     all_model_parameters_as_lists = {}
                #     for model_index in range(len(self.all_model_parameters)):
                #         all_model_parameters_as_lists[model_index] = [
                #             layer_parameters.tolist() \
                #             for layer_parameters in self.all_model_parameters[model_index]
                #         ]
                #     file.write(f'key=all_model_parameters, value={repr(all_model_parameters_as_lists)}\n')

            print(f"Metrics have been written to {file_name}")
        else:
            print("Provided object is not a dictionary.")
