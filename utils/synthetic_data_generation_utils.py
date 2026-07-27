# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def normalize_vector(vector):
    """
    Scale vector so its norm = 1.
    """
    # Compute the norm of the vector
    norm = np.linalg.norm(vector)

    # Check if the norm is not zero to avoid division by zero
    if norm == 0:
        return vector

    # Normalize the vector
    normalized_vector = vector / norm

    return normalized_vector


def scale_data(datasets):
    """
    Make max absolute value = 1 across all datasets by scaling them all by the same value.
    """
    scaling_factor = max(abs(np.max(datasets)), abs(np.min(datasets)))
    print(f"Scaling factor = {scaling_factor}")
    scaled_dataset = [dataset / scaling_factor for dataset in datasets]
    return scaled_dataset


def generate_points_on_sphere(data_point_dimension, number_of_points):
    # Generate a random vector of the given dimension
    vector = np.random.randn(number_of_points, data_point_dimension)

    # Normalize the vector so that its norm is 1
    normalized_vector = normalize_vector(vector)

    return normalized_vector


def generate_datasets(number_of_datasets, number_of_data_points, data_point_dimension, number_of_basis_vectors):
    """
    Generate datasets of random vectors according to the given parameters.
    """

    datasets = []

    for i in range(number_of_datasets):
        # Generate a set of random vectors of the given dimension to serve as the basis vectors
        basis_vectors = np.random.randn(data_point_dimension, number_of_basis_vectors)

        # Get random vectors by multiplying random norm-1 vectors by the basis
        dataset = generate_points_on_sphere(
            data_point_dimension=basis_vectors.shape[1],
            number_of_points=number_of_data_points
        ) @ basis_vectors.T

        # Normalize the norm of the generated vectors to 1
        normalized_dataset = np.vstack([normalize_vector(vector) for vector in dataset])

        datasets.append(normalized_dataset)

    return datasets



def add_overlap(datasets, overlap_dataset, number_of_overlap_points_per_dataset):
    """
    For each dataset in datasets, add number_of_overlap_points_per_dataset points randomly drawn from overlap_dataset to it.
    """

    # Get the total number of data points in the overlap dataset
    total_rows = overlap_dataset.shape[0]

    # Make sure number_of_overlap_points_per_dataset is no more than the overlap dataset size
    if number_of_overlap_points_per_dataset > total_rows:
        number_of_overlap_points_per_dataset = total_rows

    # Construct the datasets with overlap by choosing some data points at random from the overlap dataset and appending the no-overlap part
    datasets_with_overlap = [
        np.vstack(
            [
                overlap_dataset[np.random.choice(total_rows, size=number_of_overlap_points_per_dataset, replace=False)],
                M
            ]
        ) for M in datasets
    ]

    return datasets_with_overlap


def create_synthetic_datasets(number_of_datasets, data_point_dimension, number_of_overlap_basis_vectors, number_of_no_overlap_basis_vectors, overlap_dataset_size, number_of_overlap_points_per_dataset, number_of_no_overlap_points_per_dataset):

    # Create the overlap data, from which all eventual datasets will draw some points
    overlap_dataset = generate_datasets(
        number_of_datasets=1,
        number_of_data_points=overlap_dataset_size,
        data_point_dimension=data_point_dimension,
        number_of_basis_vectors=number_of_overlap_basis_vectors,
    )[0]    # We index the 0-th element as the function returns a list with a single element

    # Create the no-overlap data, which will be attached to the samples from the overlap to form all eventual datasets
    no_overlap_datasets = generate_datasets(
        number_of_datasets=number_of_datasets,
        number_of_data_points=number_of_no_overlap_points_per_dataset,
        data_point_dimension=data_point_dimension,
        number_of_basis_vectors=number_of_no_overlap_basis_vectors,
    )

    # Create datasets with overlap by adding points from the overlap dataset to the no-overlap ones
    datasets_with_overlap = add_overlap(
        datasets=no_overlap_datasets,
        overlap_dataset=overlap_dataset,
        number_of_overlap_points_per_dataset=number_of_overlap_points_per_dataset,
    )

    # Scale data values so that they are between -1 and 1, for neural net performance purposes
    scaled_datasets = scale_data(datasets_with_overlap)

    return scaled_datasets


def distribute_dataset_to_clients(dataset, number_of_clients, number_of_points_per_client):
    """
    Create a dataset for each client by randomly drawing points from the given dataset.
    """
    # Get the total number of data points in the dataset
    total_rows = dataset.shape[0]

    # Make sure number_of_points_per_client is no more than the dataset size
    if number_of_points_per_client > total_rows:
        number_of_points_per_client = total_rows

    # Construct each client's dataset by choosing some data points at random
    client_datasets = [
        dataset[
            np.random.choice(total_rows, size=number_of_points_per_client, replace=False)
        ] for _ in range(number_of_clients)
    ]

    return client_datasets


# def main():
#     import config_centralized
#     config_dict = config_centralized.get_config_dict()
#     datasets = create_synthetic_datasets(
#         number_of_datasets=config_dict['number_of_models'],
#         data_point_dimension=config_dict['SYNTHETIC_data_point_dimension'],
#         number_of_overlap_basis_vectors=config_dict['SYNTHETIC_number_of_overlap_basis_vectors'],
#         number_of_no_overlap_basis_vectors=config_dict['SYNTHETIC_number_of_no_overlap_basis_vectors'],
#         overlap_dataset_size=config_dict['SYNTHETIC_overlap_dataset_size'],
#         number_of_overlap_points_per_dataset=config_dict['SYNTHETIC_number_of_overlap_points_per_dataset'],
#         number_of_no_overlap_points_per_dataset=config_dict['SYNTHETIC_number_of_no_overlap_points_per_dataset'],
#     )
#
#     number_of_clusters = config_dict['number_of_models']
#     number_of_clients = config_dict['number_of_clients']
#
#     client_datasets = []
#     for cluster_index in range(number_of_clusters):
#         total_data_points_for_cluster = datasets[cluster_index].shape[0]
#         number_of_data_points_per_client = total_data_points_for_cluster // number_of_clients
#         client_datasets_for_cluster = distribute_dataset_to_clients(datasets[cluster_index], number_of_clients=number_of_clients, number_of_points_per_client=number_of_data_points_per_client)
#         client_datasets.extend(client_datasets_for_cluster)
#
#     # train_data, test_data = load_synthetic_data(config_dict)
#
# if __name__ == "__main__":
#     main()
