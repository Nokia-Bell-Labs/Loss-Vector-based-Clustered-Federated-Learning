# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import sys
import copy
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.learning_utils import get_custom_optimizer, get_single_model_gradients, test
from utils.logging_utils import logging
logger = logging.getLogger(__name__)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, initial_weights=None, initial_bias=None):
        super(LinearRegressionModel, self).__init__()

        # A simple linear layer: y = X * w + b
        # Forcing it to not use any bias
        self.linear = nn.Linear(input_dim, 1, bias=False)

        # Initialize weights and bias if provided
        if initial_weights is not None:
            # Ensure weights are 2-D tensor: (out_features, in_features)
            self.linear.weight = nn.Parameter(initial_weights.reshape(1, -1))
            # self.linear.weight = nn.Parameter(initial_weights)

    def forward(self, X):
        return self.linear(X)


# Base class DataGenerator with a method load_data
class DataGenerator:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def load_data(self):
        raise NotImplementedError("Method load_data() should be implemented by the subclass.")


class LinearDataGenerator(DataGenerator):
    def __init__(self, config_dict) -> None:
        super().__init__(config_dict)  # Pass the initialization value to the base class
        self.theta_stars = None

    def set_theta_stars(self, theta_stars):
        self.theta_stars = theta_stars

    def generate_data_points_and_response_lists(self, n, d, theta, sigma):
        """
        Generate n data points and their corresponding responses.

        Parameters:
        n (int): Number of data points to generate.
        d (int): Dimensionality of the data points.
        theta (np.ndarray): A d-dimensional vector.
        sigma (float): Standard deviation of the noise.

        Returns:
        list of tuples: Each tuple contains a data point (X_i) and the corresponding response (y_i).
        """
        # Generate n samples from N(0, I_d)
        X = np.random.randn(n, d)

        # Generate n scalars from N(0, sigma^2)
        epsilon = np.random.normal(0, sigma, n)

        # Compute y_i = <X_i, v> + epsilon_i
        y = np.dot(X, theta) + epsilon

        return [X.tolist(), y.tolist()]

    def load_data(self):
        config_dict = self.config_dict
        np.random.seed(config_dict['seed'])

        number_of_datasets = config_dict['number_of_clusters']
        data_point_dimension = config_dict['LINEAR_data_point_dimension']
        num_points_per_cluster = config_dict['LINEAR_num_cluster_pnts']
        sigma = config_dict['error_sigma']
        train_data_percentage = config_dict['train_data_percentage']

        cluster_datasets = [self.generate_data_points_and_response_lists(num_points_per_cluster, data_point_dimension, self.theta_stars[i], sigma)
                            for i in range(number_of_datasets)]
        # cluster_labels = list(range(config_dict['number_of_clusters']))
        # cluster_labels = [[cluster_index] * len(cluster_datasets[cluster_index]) for cluster_index in range(number_of_datasets)]

        train_data, test_data = {}, {}
        for cluster_index in range(number_of_datasets):
            number_of_data_points_for_cluster = len(cluster_datasets[cluster_index][0])
            train_data_length = int(train_data_percentage * number_of_data_points_for_cluster)

            x = torch.Tensor(cluster_datasets[cluster_index][0])
            cluster_data_points = list(zip(x, cluster_datasets[cluster_index][1]))
            # random.shuffle(cluster_data_points)
            train_data[cluster_index] = cluster_data_points[:train_data_length]
            test_data[cluster_index] = cluster_data_points[train_data_length:]

        return train_data, test_data


class Linear:
    def __init__(self, config_dict) -> None:
        self.trainloaders = None
        self.config_dict = config_dict
        self.input_dim = config_dict['LINEAR_data_point_dimension']
        self.num_models = config_dict['number_of_clusters']
        self.delta = config_dict['delta']
        self.theta_length = config_dict['theta_length']
        # As numpy vectors
        self.theta_stars = generate_thetas(self.input_dim, self.num_models, self.delta, self.theta_length)
        # As torch tensor weight vector and a bias
        self.initial_thetas = [initialize_weights_and_bias(self.input_dim)[0] for _ in range(self.config_dict['number_of_models'])]
        self.initial_models = [LinearRegressionModel(self.input_dim, initial_weights=self.initial_thetas[i]).to(config_dict['device']) for i in range(self.config_dict['number_of_models'])]
        # An extra model used for loss computation/inference
        self.scratch_model = LinearRegressionModel(self.input_dim, initial_weights=initialize_weights_and_bias(self.input_dim)[0]).to(config_dict['device'])

    # def get_data(self):
    #     self.train_data, self.test_data = self.data_generator.load_data()
    #     self.print_ground_truth()
    #     return self.train_data, self.test_data

    # def create_initial_models(self):
    #     # As torch tensor weight vector and a bias
    #     self.initial_thetas = [initialize_weights_and_bias(self.input_dim)[0] for _ in range(self.config_dict['number_of_models'])]
    #     self.initial_models = [LinearRegressionModel(self.input_dim, initial_weights=self.initial_thetas[i]) for i in range(self.config_dict['number_of_models'])]
    #     # An extra model used for loss computation/inference
    #     self.net = LinearRegressionModel(self.input_dim, initial_weights=initialize_weights_and_bias(self.input_dim)[0])

    def get_initial_models(self):
        return self.initial_models

    def get_scratch_model(self):
        return self.scratch_model

    # def compare_thetas(self, iter, cur_thetas=None):
    #     print_theta_comparisons(iter, self.theta_stars, self.initial_thetas, cur_thetas)

    def print_ground_truth(self, config_dict, train_data):
        # At this point data is per model/cluster, not per client
        # Print the loss of the optimal model of each cluster on the cluster's data
        optimal_models = [LinearRegressionModel(self.input_dim, initial_weights=convert_to_tensor(self.theta_stars[i])).to(config_dict['device']) for i in range(self.num_models)]

        # Create one trainloader per cluster, holding all data of that cluster
        cluster_trainloaders = [DataLoader(dataset=train_data[cluster_index], batch_size=self.config_dict['local_batch_size'], shuffle=True) for cluster_index in range(self.num_models)]

        logger.info('Begin: Printing optimal model losses')
        # Feed the data of each cluster to the corresponding optimal model
        for cluster_index in range(self.num_models):
            loss = test(
                net=optimal_models[cluster_index],
                testloader=cluster_trainloaders[cluster_index],
                config=self.config_dict,
            )
            logger.info(f"For cluster {cluster_index}, optimal model test loss = {loss}")
        logger.info('End: Printing optimal model losses')

#########################################################################


# Initialize weights of norm 1 and bias = 0
def initialize_weights_and_bias(input_dim):
    # Create a random tensor for weights and normalize it
    random_weights = torch.randn(input_dim)  # Random tensor
    norm = random_weights.norm(p=2)  # Compute the L2 norm
    initial_weights = random_weights / norm  # Normalize the tensor

    # Create a random value for bias between 0 and 1
    # initial_bias = torch.tensor([torch.rand(1).item()])  # Single random value between 0 and 1
    initial_bias = torch.zeros(1)  # bias = 0

    return initial_weights, initial_bias


# Generate thetas with norm = norm_length and that are at least delta apart (pairwise)
def generate_thetas(d, n, delta, norm_length, c=5, max_attempts=1000000):
    vectors = []

    # Start with a random unit vector
    x = np.random.randn(d)
    x = (x / np.linalg.norm(x)) * norm_length
    vectors.append(x)

    attempts = 0
    while len(vectors) < n and attempts < max_attempts:
        # Generate a random perturbation of length delta
        y = np.random.randn(d)
        y = (y / np.linalg.norm(y)) * delta  # Scale to length delta

        # Compute new candidate vector
        x_new = vectors[np.random.randint(len(vectors))] + y
        x_new /= np.linalg.norm(x_new)  # Normalize

        # Compute pairwise distances
        distances = np.linalg.norm(np.array(vectors) - x_new, axis=1)

        # Check if all distances are within [delta, c*delta]
        if np.all(distances >= delta) and np.all(distances <= c * delta):
            vectors.append(x_new)

        attempts += 1

    if len(vectors) < n:
        print("Could not generate the required thetas within max_attempts")
        sys.exit(1)

    return np.array(vectors)


def generate_thetas_old(d, n, delta, norm_length):
    vectors = []

    while len(vectors) < n:
        # Generate a new random vector
        new_vector = np.random.randn(d)

        # Scale it to have norm much smaller than 1
        new_vector = new_vector * (norm_length / np.linalg.norm(new_vector))

        # Check if it satisfies the distance condition with all existing vectors
        valid = True
        for v in vectors:
            if np.linalg.norm(new_vector - v) < delta:
                valid = False
                # print('condition not satisfied)')
                break

        # If valid, add it to the list of vectors
        if valid:
            vectors.append(new_vector)

    return np.array(vectors)


####################     UTILITIES     ######################################

## Many of these are currently unused
def get_model_theta(model):
    model_theta = model.linear.weight.detach().numpy().flatten()  # Convert to numpy and flatten
    return model_theta


def find_closest_pairs(theta_star, theta_cur):
    # Convert lists to numpy arrays
    theta_star_array = np.array(theta_star)
    theta_cur_array = np.array(theta_cur)

    # Initialize a list to hold the closest pairs
    closest_pairs = []

    # Loop through each point in theta_star_array
    for point in theta_star_array:
        # Compute the Euclidean distances from the current point to all points in theta_cur_array
        distances = np.linalg.norm(theta_cur_array - point, axis=1)
        # Find the index of the closest point in theta_cur_array
        closest_index = np.argmin(distances)
        # Append the pair (point from theta_star_array, closest point from theta_cur_array) to the list
        closest_pairs.append((point, theta_cur_array[closest_index]))

    return closest_pairs


def convert_to_np(t_tensors):
    return np.array([t.numpy() for t in t_tensors])


def convert_to_tensor(v):
    return torch.tensor(v, dtype=torch.float32)


def print_theta_comparisons(iter, theta_stars, initial_thetas_tensors, cur_thetas=None):
    logger.debug(f"Iteration={iter}:\n\n")
    initial_thetas = convert_to_np(initial_thetas_tensors)
    if cur_thetas is not None:
        closest_pairs = find_closest_pairs(theta_stars, cur_thetas)

    for i in range(len(theta_stars)):
        if cur_thetas is not None:
            logger.debug(f"For model {i}: theta_star vs cur_theta")
            compare_vectors(closest_pairs[i][0], closest_pairs[i][1], 'star', 'cur')
        logger.debug(f"For model {i}: initial_theta vs cur_theta")
        compare_vectors(initial_thetas[i], cur_thetas[i], 'initial', 'cur')


# Compare the learned vector with the true vector
def compare_vectors(w, theta_star, name1='', name2=''):
    # Calculate the difference (Euclidean norm) between the learned vector and true vector
    diff = np.linalg.norm(w - theta_star)
    logger.debug(f"Difference between {name1} theta and {name2} theta: {diff:.11f}")
    print(f"Difference between {name1} theta and {name2} theta: {diff:.11f}")


def get_loss_vectors(vectors, datasets):
    """
    Create loss vectors for clustering

    Parameters:
    vectors (list of np.ndarray): List of n vectors, each of dimension d.
    datasets (list of list of tuples): List of m datasets, where each dataset is for one client. It is a list of tuples (x, y) with x being a d-dim array and y being a scalar.
    cluster_client_pairs (list of list of tuples): List of m lists, each containing original cluster id, client id pairs for a client.

    Returns:
    result: A list of m lists where i-th list is the loss vector for i-th client
    """
    n = len(vectors)  # of models
    m = len(datasets)  # of clients

    # Initialize the loss vector list
    result = []

    # Loop over each dataset
    for i, dataset in enumerate(datasets):
        # Compute total squared error for the entire dataset with respect to each vector
        MSEs = [sum((np.dot(v, x) - y) ** 2 for x, y in dataset) / len(dataset) for v in vectors]
        result.append(MSEs)

    return result


def print_loss_vectors(loss_vectors, updated_cluster_client_pairs):
    i = 0
    for cluster_client_ids in updated_cluster_client_pairs:
        logger.debug(f"Client {cluster_client_ids[1]} of cluster {cluster_client_ids[0]} loss =  {loss_vectors[i]}")
        i += 1


def greedy_assignment_of_clusters_to_models(model_thetas, datasets, cluster_assignments, cluster_client_pairs):
    excluded_model_indices = []
    excluded_client_indices = []

    num_models = len(model_thetas)

    # Initialize the result sets
    result_sets = [[] for _ in range(num_models)]

    # Create a copy of the cluster_client_pairs to update
    updated_cluster_client_pairs = copy.deepcopy(cluster_client_pairs)

    # Iteratively pick a model for which clients in one of the k-means clusters have lowest loss
    # Assign all those clients to that model and repeat for remaining models and client datasets
    while len(excluded_model_indices) < num_models:
        best_model_index, assigned_client_indices = get_best_dataset_and_model(
            model_thetas, datasets, cluster_assignments,
            excluded_model_indices, excluded_client_indices)

        # Mark that this model is already used up and also mark that these clients
        # are already assigned to a model so they can be excluded from next rounds
        excluded_model_indices.append(best_model_index)
        excluded_client_indices.extend(assigned_client_indices)

        # Assign the datasets of these clients to the model that we just found for the next round
        for indx in assigned_client_indices:
            updated_cluster_client_pairs[indx].append(best_model_index + 1)
            # Assign the entire dataset to the corresponding result set
            result_sets[best_model_index].extend(datasets[indx])

    return result_sets, updated_cluster_client_pairs


def get_best_dataset_and_model(vectors, datasets, cluster_assignments, exclude_vector_indices=None, exclude_dataset_indices=None):
    # Initialize default values for exclusion
    exclude_vector_indices = exclude_vector_indices or []
    exclude_dataset_indices = exclude_dataset_indices or []

    # Group datasets by their cluster labels and track indices
    combined_datasets = defaultdict(list)
    dataset_indices = defaultdict(list)

    for idx, (dataset, label) in enumerate(zip(datasets, cluster_assignments)):
        if idx in exclude_dataset_indices:
            continue  # Skip excluded datasets
        combined_datasets[label].append(dataset)
        dataset_indices[label].append(idx)

    best_loss = float('inf')
    best_label = None
    best_vector = None
    best_vector_index = None
    best_indices = []

    # Iterate through each label and combine corresponding datasets
    for label, dataset_group in combined_datasets.items():
        # Combine datasets
        combined_dataset = []
        for dataset in dataset_group:
            combined_dataset.extend(dataset)

        # Calculate loss for each vector
        for vector_index, v in enumerate(vectors):
            if vector_index in exclude_vector_indices:
                continue  # Skip excluded vectors

            if len(combined_dataset) == 0:  # Avoid division by zero
                continue

            mse = sum((np.dot(v, x) - y) ** 2 for x, y in combined_dataset) / len(combined_dataset)

            # Check for the best loss
            if mse < best_loss:
                best_loss = mse
                best_label = label
                best_vector = v
                best_vector_index = vector_index  # Update best vector index
                best_indices = dataset_indices[label]  # Update indices of best label

    return best_vector_index, best_indices


logger.debug("\n\n\nSTARTING UP ****************************************************************************************************************\n\n")
