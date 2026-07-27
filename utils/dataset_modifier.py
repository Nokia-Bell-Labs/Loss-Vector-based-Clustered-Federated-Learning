# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import random
import sys
import numpy as np

from torch.utils.data import  Subset
import copy
from collections import defaultdict
from collections import Counter

from collections import defaultdict
from torch.utils.data import Subset


from utils.logging_utils import logging
logger = logging.getLogger(__name__)

###### BEGIN ######################## The following functions are for building clusters with random label skews ###########################

def print_label_assignments(label_assignments):
    """
    Prints label assignments for each party in sorted integer format.

    Args:
        label_assignments (list of sets): Each set contains labels assigned to a party.
    """
    logger.info("\n=== Label Assignments per Party ===")
    for i, label_set in enumerate(label_assignments):
        label_list = sorted(int(label) for label in label_set)
        logger.info(f"Cluster {i}: Labels {label_list}")


def convert_to_label_assignments(labels_groups):
    return [set(labels) for labels in labels_groups]


def print_party_label_distribution(group_indices_dict, label_assignments, dataset):
    """
    Prints number of data points per assigned label for each party.
    """
    logger.info("\n=== Party Label Distribution Summary ===")
    
    for party in sorted(group_indices_dict.keys()):
        indices = group_indices_dict[party]
        labels = [dataset[idx][1] for idx in indices]
        label_counts = Counter(labels)

        logger.info(f"\nParty {party}:")
        logger.info(f"  Total samples       = {len(indices)}")
        logger.info(f"  Assigned labels     = {sorted(label_assignments[party])}")
        logger.info(f"  Label counts (assigned labels only):")
        
        for label in sorted(label_assignments[party]):
            count = label_counts.get(label, 0)
            logger.info(f"    Label {label}: {count} sample{'s' if count != 1 else ''}")
            if count == 0:
                logger.info(f"    WARNING: Assigned label {label} has 0 samples!")


def generate_label_assignments(num_classess, n_parties, num_labels_per_cluster, num_label_overlap):
    assert num_labels_per_cluster <= num_classess, "Each party's label count can't exceed total classes."
    assert num_label_overlap < num_labels_per_cluster, "Overlap must be less than label count per party."

    party_labels = [set() for _ in range(n_parties)]
    all_labels = list(range(num_classess))

    for i in range(n_parties):
        assigned = False
        while not assigned:
            candidate_labels = set(np.random.choice(all_labels, num_labels_per_cluster, replace=False))
            if all(
                len(candidate_labels & party_labels[j]) == num_label_overlap
                for j in range(i)
            ):
                party_labels[i] = candidate_labels
                assigned = True
    return party_labels


def partition_dataset_with_label_overlap(dataset, n_parties, num_classes, num_labels_per_cluster,
                                         num_label_overlap, beta=0.5, reserve_pct=0.1,
                                         label_assignments=None):
    """
    Partitions a dataset among parties based on label assignments and Dirichlet sampling.

    Args:
        dataset: PyTorch-style dataset.
        n_parties: Number of parties (clusters for clients).
        K: Total number of classes.
        num_labels_per_cluster: Number of labels each party gets.
        num_label_overlap: Number of labels shared between any two parties.
        beta: Dirichlet concentration parameter.
        reserve_pct: % of label data to reserve for uniform allocation.
        label_assignments: Optional. A list of sets indicating assigned labels per party.

    Returns:
        group_indices_dict: {party_id: list of indices}
        subsets: {party_id: list of (image, label) tuples}
        label_assignments: {party_id: set of assigned labels}
    """

    y_train = [label for _, label in dataset]
    y_train = np.array(y_train)

    group_indices_dict = defaultdict(list)
    subsets = {}

    # Use provided label_assignments or generate new ones
    if label_assignments is None:
        label_assignments = generate_label_assignments(
            K, n_parties, num_labels_per_cluster, num_label_overlap
        )

    # Map each class to parties that hold it
    class_to_parties = defaultdict(list)
    for party, labels in enumerate(label_assignments):
        for label in labels:
            class_to_parties[label].append(party)

    for k in range(num_classes):
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)

        assigned_parties = class_to_parties[k]
        if not assigned_parties:
            continue

        # Reserve a portion for guaranteed distribution
        n_reserved = int(len(idx_k) * reserve_pct)
        logger.debug(f"n_reserved of {k}= {n_reserved}")
        reserved_idx = idx_k[:n_reserved]
        dirichlet_idx = idx_k[n_reserved:]

        proportions = np.random.dirichlet([beta] * len(assigned_parties))
        split_points = (np.cumsum(proportions) * len(dirichlet_idx)).astype(int)[:-1]
        split_indices = np.split(dirichlet_idx, split_points)

        for party, idx in zip(assigned_parties, split_indices):
            group_indices_dict[party].extend(idx.tolist())

        split_reserved = np.array_split(reserved_idx, len(assigned_parties))
        for party, idx in zip(assigned_parties, split_reserved):
            logger.debug(f"assigning extra reserved pnts for {k} to {party}:  {len(idx.tolist())}")
            group_indices_dict[party].extend(idx.tolist())

    for i in range(n_parties):
        group_indices = group_indices_dict[i]
        np.random.shuffle(group_indices)
        subset = Subset(dataset, group_indices)
        subsets[i] = list(subset)

    return subsets, group_indices_dict,  label_assignments


def try_partition_until_min(dataset, n_parties, num_classes, num_labels_per_cluster, num_label_overlap,
                            beta=0.5, initial_reserve_pct=0.05, max_reserve_pct=0.5, step=0.05, min_num=100, label_assignments=None):
    """
    Repeatedly tries partitioning the dataset with increasing reserve_pct until each party
    gets at least `min_num` samples, or reserve_pct exceeds limit.
    """
    reserve_pct = initial_reserve_pct
    while reserve_pct <= max_reserve_pct:
        logger.info(f"Trying with reserve_pct = {reserve_pct:.2f}")
        subsets, group_indices_dict, label_assignments = partition_dataset_with_label_overlap(
            dataset, n_parties, num_classes,
            num_labels_per_cluster, num_label_overlap,
            beta=beta, reserve_pct=reserve_pct, label_assignments=label_assignments
        )

        sizes = [len(group_indices_dict[i]) for i in range(n_parties)]
        if all(size >= min_num for size in sizes):
            logger.info("Minimum satisfied for all parties.")
            print_label_assignments(label_assignments)
            return  subsets, group_indices_dict

        logger.info(f"Some parties below min_num={min_num}. Retrying with higher reserve...")
        reserve_pct += step

    print(f"Failed to meet min_num={min_num} for all parties by reserve_pct={max_reserve_pct}")
    sys.exit(1)
    #raise ValueError(f"Failed to meet min_num={min_num} for all parties by reserve_pct={max_reserve_pct}")

# This is how to partition dataset into clusters
##############################
# labels_assignment = generate_label_assignments(K, n_parties, a, b)
# ##################Partition##############
# group_indices_dict, subsets, _ = try_partition_until_min(
#     dataset, n_parties=5, num_classes=K,
#     num_labels_per_cluster=a, num_label_overlap=b,
#     beta=beta, initial_reserve_pct=0.1,
#     max_reserve_pct=1.0, step=0.05, min_num=5100,
#     label_assignments=labels_assignment
# )

# ################# View result
# Use the same label_assignments generated in the function
# If not returned, modify the function to return it as well
##########################
#print_party_label_distribution(group_indices_dict, labels_assignment, dataset)


###### END ######################## The following functions are for building clusters with random label skews #############################

###### BEGIN ######################## The following functions are for distributing cluster data to clients #############################

def map_sorted_indices_with_clusters(client_ranks, client_ds_size):
    """
    Returns a list of (new_index, old_index, cluster_id)
    where new_index is the position after sorting by rank.
    The cluster sizes are inferred from client_ds_size.
    """
    # Build the mapping from original client index to cluster_id using client_ds_size
    client_to_cluster = []
    for cluster_id, clients in enumerate(client_ds_size):
        client_to_cluster.extend([cluster_id] * len(clients))  # Each cluster has clients as listed

    # Sort the client ranks and pair them with their original indices
    indexed_ranks = list(enumerate(client_ranks))  # (original_index, rank)
    sorted_indices = sorted(indexed_ranks, key=lambda x: x[1])  # Sort by rank

    # Map the sorted indices back to (new_index, old_index, cluster_id)
    index_map = []
    client_to_cluster_map = []
    for new_index, (old_index, _) in enumerate(sorted_indices):
        cluster_id = client_to_cluster[old_index]
        index_map.append((new_index, old_index, cluster_id))
        client_to_cluster_map.append(cluster_id)

    return index_map, client_to_cluster_map


def sort_indices_by_cluster(index_map, client_ds_size):
    """
    Returns a list where each entry contains sorted new indices for each cluster,
    sorted based on the original (old) indices of the clients.
    """
    # Initialize list of lists for each cluster
    client_to_cluster_list = [[] for _ in range(len(client_ds_size))]

    # Collect (new_index, old_index) for each cluster
    for new_index, old_index, cluster_id in index_map:
        client_to_cluster_list[cluster_id].append((new_index, old_index))

    # Sort each cluster's list by old_index, then extract only the new_index
    for cluster_id in range(len(client_to_cluster_list)):
        sorted_pairs = sorted(client_to_cluster_list[cluster_id], key=lambda x: x[1])  # sort by old_index
        client_to_cluster_list[cluster_id] = [new_index for new_index, _ in sorted_pairs]

    return client_to_cluster_list


def scale_down_sizes(sizes, available):
    total_required = sum(sizes)
    if total_required <= available:
        return sizes
    scale = available / total_required
    scaled_sizes = [int(s * scale) for s in sizes]
    while sum(scaled_sizes) > available:
        max_idx = scaled_sizes.index(max(scaled_sizes))
        scaled_sizes[max_idx] -= 1
    return scaled_sizes


def assign_without_replacement(data, sizes):
    data = copy.copy(data)  # safe to shuffle now
    random.shuffle(data)
    assignments = []
    start = 0
    for size in sizes:
        assignments.append(data[start:start + size])
        start += size
    return assignments


def assign_with_replacement(data, sizes):
    return [random.choices(data, k=size) for size in sizes]


def order_clients_by_rank(assignments, client_ranks):
    """Sort assignments based on global client ranks."""
    return [assignment for _, assignment in sorted(zip(client_ranks, assignments), key=lambda x: x[0])]


# If the input is None then set to default ranks
def update_client_ranks(client_ranks, num_clients):
    if client_ranks is None:
        return [i for i in range(num_clients)]
    return client_ranks


# If the input is None then set to default sizes
def update_client_ds_sizes(client_ds_size, num_clusters, num_clients_per_cluster, points_per_client):
    if client_ds_size is None:
        return [[points_per_client] *num_clients_per_cluster for _ in range(num_clusters)]
    return client_ds_size


def get_num_clients(client_ds_size):
    return sum(len(cluster_clients) for cluster_clients in client_ds_size)


def assign_data_to_clients(data, client_ds_size, client_ranks, with_replacement=False):
    all_assignments = []

    for i, _ in data.items():
        cluster_data = data[i] 
        sizes = client_ds_size[i]
        available = len(cluster_data)

        if not with_replacement:
            sizes = scale_down_sizes(sizes, available)
            assignments = assign_without_replacement(cluster_data, sizes)
        else:
            assignments = assign_with_replacement(cluster_data, sizes)

        all_assignments.extend(assignments)
    # print("Before shuffle")
    # for i, client_data in enumerate(all_assignments):
    #     print(f"Client {i}: {client_data}")
    # Final ordering by global rank
    return order_clients_by_rank(all_assignments, client_ranks)


def assign_data(data, config, data_distr, with_replacement=False):
    num_of_clusters = len(data)
    num_of_clients_per_cluster = config['number_of_clients_per_cluster']
    datapoints_per_client = config['number_of_datapoints_per_client']
    if data_distr is None:
        client_ds_size = None
        client_ranks = None
    else:
        client_ds_size = data_distr['client_ds_size_distr']
        client_ranks = data_distr['client_indx_permutation']

    client_ds_size = update_client_ds_sizes(client_ds_size, num_of_clusters, num_of_clients_per_cluster, datapoints_per_client)
    logger.info(f'client_ds_sizes = {client_ds_size}')
    num_clients = get_num_clients(client_ds_size)
    client_ranks = update_client_ranks(client_ranks, num_clients)
    logger.info(f'shuffle list: {client_ranks}')

    # Just for printing the new ordering of clients within clusters
    index_map, client_to_cluster_map = map_sorted_indices_with_clusters(client_ranks, client_ds_size)
    client_to_cluster_list = sort_indices_by_cluster(index_map, client_ds_size)
    # Print client_to_cluster_list (sorted new indices for each cluster)
    logger.info(f"Ground Truth: {client_to_cluster_list}")

    return assign_data_to_clients(data, client_ds_size, client_ranks, with_replacement), client_to_cluster_map

###### END ######################## The following functions are for distributing cluster data to clients #############################

######  BEGIN  ############### The following functions are for distributing data to clusters ########################################

# Define a function to generate swap pairs based on the list of labels
def generate_swap_pairs(label_group):
    """
    Generates pairs of adjacent labels to swap from the label group.
    
    Parameters:
    - label_group: List of labels to generate swap pairs for.
    
    Returns:
    - List of tuples (old_label, new_label) for swapping.
    """
    swap_pairs = []
    for i in range(0, len(label_group), 2):
        # Ensure there is an adjacent pair to swap
        if i + 1 < len(label_group):
            swap_pairs.append((label_group[i], label_group[i + 1]))
            swap_pairs.append((label_group[i+1], label_group[i]))
    return swap_pairs


# Define a function to swap labels in the group
def swap_labels_in_group(dataset, group_indices, label_swap_pairs):
    """
    Swaps labels for a given list of indices based on specified swap pairs.
    
    Parameters:
    - dataset: The dataset to operate on (a list of (image, label) tuples)
    - group_indices: Indices of the group within the dataset
    - label_swap_pairs: A list of tuples (label1, label2) to swap
    
    Returns:
    - A list of modified (image, swapped label) tuples
    """
    modified_group = []
    for idx in group_indices:
        image, label = dataset[idx]
        # Find the label swap for this particular label
        for (old_label, new_label) in label_swap_pairs:
            if label == old_label:
                label = new_label
                break
        modified_group.append((image, label))
    
    return modified_group


# Define a function to split dataset into N pieces based on label filtering and random sampling
def split_dataset(dataset, label_keys, label_groups, num_points_for_all_classes, swaps_lists, do_swap=False):
    """
    Splits dataset (random without replacement) into  pieces based on label filtering and random sampling
    
    Parameters:
    - dataset: The dataset to operate on (a list of (image, label) tuples)
    - label_keys: The set of labels of classes in this dataset
    - label_groups: 2-D list of rows (or groups). A row specifies labels of points that need to be included for this row.  
    - num_points_for_all_classes: list that specifies number of points to be included for a label in a row (if that label is in that row)
    - swap_lists: 2-D list of rows. Adjacent labels in a row need to be swapped in corresponding row of points 
    - do_swap: Flag to indicate whether label swapping is to be done
    
    An Example: 
    
    label_groups = [
        [0, 1,2,3,4,5,6,7,8,9],  # This row/group will have points from first 10 labels
        [0, 1,2,3,4,5,6,7,8,9],
        [0, 1,2,3,4,5,6,7,8,9],
    ] 
    swaps_lists = [
        [0, 3, 8, 9],  # For label swapping of label pairs (0, 3), (8,9) in first group/row
        [1, 4, 6, 9],
        [1, 2, 3, 6],
    ]
    
    Returns:
    - The list of rows of datapoints after splitting and optional label swapping
    """
    subsets = {}
    all_labels = list(label_keys)
    
    group_indices_dict = {}  # This will store the indices for each group
    
    # Get indices for each label
    label_indices = {i: [] for i in range(len(all_labels))}  # E.g. CIFAR-10 dataset will have upto 10 labels

    # Store data point indices by labels
    for idx, (image, label) in enumerate(dataset):
        label_indices[label].append(idx)

    selected_indices = set()

    for i, group in enumerate(label_groups):
        group_indices = []
        
        # For each label in the group, sample num_points_for_all_classes[label] data points
        for label in group:

            num_points_for_class = num_points_for_all_classes[label]
            
            # Filter out previously selected indices to ensure no duplicates: sampling without replacement
            available_indices = [idx for idx in label_indices[label] if idx not in selected_indices]

            if len(available_indices) < num_points_for_class:
                print("Failed for lack of data")
                print(f"Not enough samples for label {label}. Found {len(available_indices)} but need {num_points_for_class}.")
                sys.exit()
                raise ValueError(f"Not enough samples for label {label}. Found {len(available_indices)} but need {num_points_for_class}.")
            
            # Randomly sample num_points_for_class points from the available indices for this label
            sampled_indices = random.sample(available_indices, num_points_for_class)
            group_indices.extend(sampled_indices)

        # Add sampled indices to the selected set to avoid duplicates across subsets
        selected_indices.update(group_indices)

        # Store group indices in the dictionary
        group_indices_dict[i] = group_indices

        # Generate swap pairs if swapping is enabled
        if do_swap:
            # Generate swap pairs dynamically based on the group of labels
            label_swap_pairs = generate_swap_pairs(swaps_lists[i])
            # Create the modified group by swapping labels
            modified_group = swap_labels_in_group(dataset, group_indices, label_swap_pairs)
            subsets[i] = modified_group
        else:
            # No label swap, just add the subset
            subset = Subset(dataset, group_indices)
            subsets[i] = list(subset)
    
    return subsets, group_indices_dict


# To convert from dict form to list form
def convert_from_dict_to_list(dict_dataset):
    list_dataset = []
    for key, data_pnt_list in dict_dataset.items():
        list_dataset.extend(data_pnt_list)

    return list_dataset, dict_dataset.keys()


def split_dict_values(d, ratio=0.5):
    """
    Splits a dictionary into two dictionaries by randomly distributing the values of each key.
    
    Parameters:
        d (dict): The input dictionary where values are lists.
        ratio (float): Proportion of items to go into the first dictionary.
    
    Returns:
        dict1, dict2: Two dictionaries with randomly split values.
    """
    dict1, dict2 = {}, {}

    for key, value_list in d.items():
        random.shuffle(value_list)  # Shuffle to ensure randomness
        split_index = int(len(value_list) * ratio)
        dict1[key] = value_list[:split_index]  # First part
        dict2[key] = value_list[split_index:]  # Second part

    return dict1, dict2

######  END  ############### For distributing data to clusters ########################################

######  BEGIN  ############### For distributing data to clusters using label dominance #############

def partition_dataset_dominance_based(dataset, selected_labels=[0,1,2,3,4,5,6,7,8,9], p=0.2):
    """
    Partition dataset into groups for selected labels.
    Each group i (for label i) gets:
      - p% of its group_size from its own label
      - (100 - p)% of group_size equally from all other selected labels

    Ensures:
      - Each data point is used at most once
      - Group size = 90% of min size of selected labels

    Parameters:
    -----------
    dataset : Dataset
        dataset of (image, label) pairs.

    selected_labels : list
        Labels for which to form groups (e.g., [0, 1, 2]).

    p : float
        Percentage of each group to come from its labels.

    Returns:
    --------
    subsets : dict
        Mapping from label (as group key) to list of dataset items.

    group_indices_dict : dict
        Mapping from label to list of dataset indices used for that group.
    """
    subsets = {}
    group_indices_dict = {}

    # Step 1: Gather and shuffle indices by label
    label_indices = {l: [] for l in selected_labels}
    for idx, (image, label) in enumerate(dataset):
        if label in selected_labels:
            label_indices[label].append(idx)
    for l in label_indices:
        random.shuffle(label_indices[l])

    # Step 2: Determine group size
    min_label_count = min(len(indices) for indices in label_indices.values())
    group_size = int(0.9 * min_label_count)

    # Step 3: Determine per-label allocation
    own_label_count = int((p)/100.0  * group_size)
    other_label_count = group_size - own_label_count
    num_other_labels = len(selected_labels) - 1
    per_other_label_count = (
        int(other_label_count / num_other_labels) if num_other_labels > 0 else 0
    )

    # Step 4: Assign samples
    # Track available indices per label
    label_ptr = {l: 0 for l in selected_labels}

    for group_label in selected_labels:
        group_indices = []

        # Assign own-label samples
        start = label_ptr[group_label]
        end = start + own_label_count
        own_samples = label_indices[group_label][start:end]
        group_indices.extend(own_samples)
        label_ptr[group_label] = end

        # Assign from other labels
        for other_label in selected_labels:
            if other_label == group_label:
                continue
            start = label_ptr[other_label]
            end = start + per_other_label_count
            other_samples = label_indices[other_label][start:end]
            group_indices.extend(other_samples)
            label_ptr[other_label] = end

        random.shuffle(group_indices)
        group_indices_dict[group_label] = group_indices
        subsets[group_label] = list(Subset(dataset, group_indices))

    return subsets, group_indices_dict

######  END   ############### For distributing data to clusters using label dominance ###############

######  BEGIN  ############### For distributing data to clients for experiment ########################################

client_cfg_5_15_1000 = {
            "client_ds_size_distr" : [ # None or num_data_points_per_client
                [1000, 1000, 1000],  # Cluster 0 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [1000, 1000, 1000],  # Cluster 1 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [1000, 1000],  # Cluster 2 has 2 clients requesting 1000 samples each (will be scaled down if without replacement)
                [1000, 1000, 1000],  # Cluster 3 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [1000, 1000, 1000, 1000],  # Cluster 3 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
            ],
            "client_indx_permutation" : [2, 0, 4, 3, 1, 6, 5, 8, 9, 7, 12, 10, 11, 14, 13], # None, or Ranks for ordering clients in output list
        }
client_cfg_5_15_small = {
            "client_ds_size_distr" : [ # None or num_data_points_per_client
                [500, 500, 500],  # Cluster 0 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [500, 500, 1000],  # Cluster 1 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [500, 1000],  # Cluster 2 has 2 clients requesting 1000 samples each (will be scaled down if without replacement)
                [500, 1000, 500],  # Cluster 3 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [500, 1000, 500, 500],  # Cluster 3 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
            ],
            "client_indx_permutation" : [2, 0, 4, 3, 1, 6, 5, 8, 9, 7, 12, 10, 11, 14, 13], # None, or Ranks for ordering clients in output list
        }
client_cfg_4_16_small = {
            "client_ds_size_distr" : [ # None or num_data_points_per_client
                [500, 500, 500, 500, 1000],  # Cluster 0 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [500, 500, 1000, 500],  # Cluster 1 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [500, 1000, 500],  # Cluster 3 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
                [500, 1000, 500, 500],  # Cluster 3 has 3 clients requesting 1000 samples each (will be scaled down if without replacement)
            ],
            "client_indx_permutation" :  None, # or Ranks for ordering clients in output list
        }
dict_overlap_swap_exps = {

# Pairs **************************************************
    "exp4": { # Simulates no overlap for pairs
        "label_groups" : [
            [0, 1], [2,3], [4,5], [6,7], [8,9], # class labels for this cluster
        ],
        "swaps_lists" : [
            # No swap
        ],
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 2500, # Number of train samples per label in each group in label_groups
        "N_test" : 400 # Number of test samples per label in each group in label_groups
    },    

    "exp14": { # Swaps on pairs (exp4 baseline)
        "label_groups" : [
            [0,1], [0,1], [2,3], [2,3], [4,5], [4,5], [6,7],[6,7],[8,9], [8,9],
        ],
        "swaps_lists" : [
            [0,1], [], [2,3], [], [4,5],[], [6,7], [],[8,9],[],
        ],
        "swaps_flag" : 1, 
        "N_train" : 2500, 
        "N_test" : 400 
    },   
    "exp24": { # Overlaps on pairs
        "label_groups" : [
            [0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7],[7,8],[8,9],
        ],
        "swaps_lists" : [
            
        ],
        "swaps_flag" : 0, 
        "N_train" : 2500, 
        "N_test" : 400 
    },   

# Triples ************************************************************
    "exp5": { # Simulates no overlap for triples
        "label_groups" : [
            [0,1,2],  [3,4,5], [6,7,8],  # class labels for this cluster
        ],
        "swaps_lists" : [
            # No swap
        ],
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 2500, # Number of train samples per label in each group in label_groups
        "N_test" : 400 # Number of test samples per label in each group in label_groups
    },    
    "exp15": { # Swaps on triples (exp5 baseline)
        "label_groups" : [
            [0,1,2], [0,1,2], [3,4,5], [3,4,5], [6,7,8], [6,7,8],  # class labels for this cluster
        ],
        "swaps_lists" : [
            [0,1], [], [3,4], [], [6,7], [], 
        ],
        "swaps_flag" : 1, #1 means true, false otherwise
        "N_train" : 2500, # Number of train samples per label in each group in label_groups
        "N_test" : 400 # Number of test samples per label in each group in label_groups
    },    

    "exp25": { # Simulates  overlap on triples
        "label_groups" : [
            [0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0],
        ],
        "swaps_lists" : [
            # No swap
        ],
        "client_config": client_cfg_5_15_1000, 
        "swaps_flag" : 0, 
        "N_train" : 2500, 
        "N_test" : 400 
    },

# 4 per set ******************************************************************************
    "exp6": { # Simulates no overlap for 4s
        "label_groups" : [
            [0,1,2,3],  [4,5,6,7], 
        ],
        "swaps_lists" : [
            # No swap
        ],
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 2500, # Number of train samples per label in each group in label_groups
        "N_test" : 400 # Number of test samples per label in each group in label_groups
    },    
    "exp16": { # Swaps on 4s (exp6 baseline)
        "label_groups" : [
            [0,1,2,3],  [4,5,6,7], [0,1,2,3],  [4,5,6,7], 
        ],
        "swaps_lists" : [
            [0,1], [6,7], [],  [], 
        ],
        "swaps_flag" : 1, #1 means true, false otherwise
        "N_train" : 2500, # Number of train samples per label in each group in label_groups
        "N_test" : 400 # Number of test samples per label in each group in label_groups
    },    

    "exp26": { # Simulates bigger overlap
        "label_groups" : [
            [0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9], [8,9,0,1],
        ],
        "swaps_lists" : [
            # No swap
        ],
        "client_config": client_cfg_5_15_1000,
        "swaps_flag" : 0, 
        "N_train" : 2500, 
        "N_test" : 400 
    },

# 5 per set ******************************************************************************
    "exp7": { # Simulates no overlap for 5s
        "label_groups" : [
            [0,1,2,3,4],  [5,6,7,8,9], 
        ],
        "swaps_lists" : [
            # No swap
        ],
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 2500, # Number of train samples per label in each group in label_groups
        "N_test" : 400 # Number of test samples per label in each group in label_groups
    },    
    "exp17": { # Swaps (only 2) : Medium size
        "label_groups" : [
            [0,1,2,3,4], [0,1,2,3,4], [5,6,7,8,9], [5,6,7,8,9], 
        ],
        "swaps_lists" : [
            [0,1], [3,4], [6,7], [5,8], 
        ],
        "swaps_flag" : 1, 
        "N_train" : 1500, 
        "N_test" : 300 
    },   
    "exp27": { # Simulates bigger overlap
        "label_groups" : [
            [0,1,2,3,4], [2,3,4,5,6], [4,5,6,7,8], [6,7,8,9,0], 
        ],
        "swaps_lists" : [
            # No swap
        ],
        "swaps_flag" : 0, 
        "N_train" : 1500, 
        "N_test" : 300 
    },

# Full set ******************************************************************************
    "exp8": { # full set baseline
        "label_groups" : [
            [0, 1,2,3,4,5,6,7,8,9],  # class labels for this cluster
        ],
        "swaps_lists" : [
           
        ],
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 1500, # Number of train samples per label in each group in label_groups
        "N_test" : 300 # Number of test samples per label in each group in label_groups
    },        
    "exp18": { #full set: extreme  swaps and overlaps (baseline exp8)
        "label_groups" : [
            [0, 1,2,3,4,5,6,7,8,9],  # class labels for this cluster
            [0, 1,2,3,4,5,6,7,8,9],
            [0, 1,2,3,4,5,6,7,8,9],
        ],
        "swaps_lists" : [
            [0, 3, 8, 9],  # Pairs of adjacent labels that need to be swapped for this class (0,3) and (8,9)
            [1, 4, 6, 9],
            [1, 2, 3, 6],
        ],
        "swaps_flag" : 1, #1 means true, false otherwise
        "N_train" : 1500, # Number of train samples per label in each group in label_groups
        "N_test" : 300 # Number of test samples per label in each group in label_groups
    },

# Full set ******************************************************************************
    "exp9": { # baseline no   overlaps 
        "label_groups" : [
            [0,1,2,3,4,5],  # class labels for this cluster
            
        ],
        "swaps_lists" : [

        ],
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 1500, # Number of train samples per label in each group in label_groups
        "N_test" : 300 # Number of test samples per label in each group in label_groups
    },
    "exp19": { # baseline no   overlaps 
        "label_groups" : [
            [0,1,2,3,4,5],  # class labels for this cluster
            [1,2,3,4,5,8],
            [5,6,7,8,9,2],
            [6,7,8,9,0,4],
            [0,1,3,7,9,6],
            
        ],
        "swaps_lists" : [
            
        ],
        "client_config": client_cfg_5_15_small,
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 1500, # Number of train samples per label in each group in label_groups
        "N_test" : 300 # Number of test samples per label in each group in label_groups
    },
    "exp29": { # extreme   overlaps with 0ne swap per group
        "label_groups" : [
            [0,1,2,3,4,5],  # class labels for this cluster
            [1,2,3,4,5,8],
            [5,6,7,8,9,2],
            [6,7,8,9,0,4],
            [0,1,3,7,9,6],
        ],
        "swaps_lists" : [
            [1,2],  [3,4], [5,6], [7,8],[3,7],
 
        ],
        "client_config": client_cfg_5_15_small,
        "swaps_flag" : 1, #1 means true, false otherwise
        "N_train" : 1500, # Number of train samples per label in each group in label_groups
        "N_test" : 300 # Number of test samples per label in each group in label_groups
    },

# Compare with papers ******************************************************************************
# (1) Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints
# 20 clients, 4 clusters. Data uniformly distributed amongst all clients
# Label swapping based on random permutations
# Need 500 points per client for CIFAR to work
# Need many rounds (100)
# batch size 100, 3 epochs per round
# # of rounds is large as they have to do log K splits min. For each split have to wait certain number of rounds approx. 50
    "exp110": { #full set baseline
        "label_groups" : [
            [0, 1,2,3,4,5,6,7,8,9],  # class labels for this cluster
        ],
        "swaps_lists" : [
           
        ],
        "swaps_flag" : 0, #1 means true, false otherwise
        "N_train" : 1500, # Number of train samples per label in each group in label_groups
        "N_test" : 300 # Number of test samples per label in each group in label_groups
    },        
    "exp111": { #full set: extreme  swaps and overlaps (baseline exp8)
        "label_groups" : [
            [0, 1,2,3,4,5,6,7,8,9],  # class labels for this cluster
            [0, 1,2,3,4,5,6,7,8,9],
            [0, 1,2,3,4,5,6,7,8,9],
            [0, 1,2,3,4,5,6,7,8,9],
        ],
        "swaps_lists" : [
            [0, 3, 8, 9],  # Pairs of adjacent labels that need to be swapped for this class (0,3) and (8,9)
            [1, 4, 6, 9],
            [1, 2, 3, 6],
            [3, 5, 4, 7],
        ],
        "swaps_flag" : 1, # 1 means true, false otherwise
        "N_train" : 1250, # Number of train samples per label in each group in label_groups
        "N_test" : 200 # Number of test samples per label in each group in label_groups
    },

# FedGH: Heterogeneous Federated Learning with Generalized Global Header
# *** 120 and 122 is already covered using exp4 and exp121 resp.
    "exp121": { # Overlaps on pairs
        "label_groups" : [
            [0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7],[7,8],[8,9],[9,0]
        ],
        "swaps_lists" : [
            
        ],
        "swaps_flag" : 0, 
        "N_train" : 2500, 
        "N_test" : 400 
    },

# PERSONALIZED FEDERATED LEARNING WITH FEATURE ALIGNMENT AND CLASSIFIER COLLABORATION
# *** 130 is already covered with dominant, 131 is random permutation/swap using exp17 for 20 clients
# Label skew experiments
    "exp201": { # Simulates bigger overlap
        "label_skew" : True,
        "label_groups" : [
            # No label Groups, Let it generate one
        ],
        "swaps_lists" : [
            # No swap
        ],
        "num_labels_per_cluster": 4,
        "label_overlap_size": 2,
        "beta": 0.5, # For Dirichlet
        "reserve_frac": 0.1,
        'N_clusters': 5,
        "N_train" : 5100, # This is min # of train per cluster
        "N_test" : 510 # This is min # of test per cluster
    },

    "exp202": { # Simulates bigger overlap
        "label_skew" : True,
        "label_groups" : [
            [0,1,2,3,4,5,6,7,8],  # class labels for this cluster
            [1,2,3,4,5,6,7,8,9],
            [0,2,3,4,5,6,7,8,9],
            [0,1,3,4,5,6,7,8,9],
            [0,1,2,3,4,5,6,7,8,9],
            
        ],
        "swaps_lists" : [
            # No swap
        ],
        "num_labels_per_cluster": 7, # bet 7 and 8
        "label_overlap_size": 5, # approx varies between 4 and 6
        "beta": 0.5, # For Dirichlet
        "reserve_frac": 0.1,
        'N_clusters': 5,
        "N_train" : 6000, # This is min # of train per cluster
        "N_test" : 750 # This is min # of test per cluster
    }, 
      
    "exp203": { # Simulates bigger overlap
        "label_skew" : True,
        "label_groups" : [
             [0, 1], [2,3], [4,5], [6,7], [8,9], # No overlap
            
        ],
        "swaps_lists" : [
            # No swap
        ],
        "num_labels_per_cluster": 2,
        "label_overlap_size": 0,
        "beta": 0.5, # For Dirichlet
        "reserve_frac": 0.1,
        'N_clusters': 5,
        "N_train" : 6000, # This is min # of train per cluster
        "N_test" : 750 # This is min # of test per cluster
    },

    "exp204": { # Simulates bigger overlap
        "label_skew" : True,
        "label_groups" : [
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
        ],
        "swaps_lists" : [
            # No swap
        ],
        "num_labels_per_cluster": 10, # bet 7 and 8
        "label_overlap_size": 0, # approx varies between 4 and 6
        "beta": 0.5, # For Dirichlet
        "reserve_frac": 0.1,
        'N_clusters': 5,
        "N_train" : 4500, # This is min # of train per cluster
        "N_test" : 750 # This is min # of test per cluster
    },  

    "exp301": { # Simulates bigger overlap
        "feature_skew" : True,
        "other_rot_angles" : [
            90, 180, 270, 
        ],
        "swaps_lists" : [
            # No swap
        ],
        "N_train" : 10000, # This is min # of train per cluster
        "N_test" : 2000 # This is min # of test per cluster
    },  

    "exp401": { # For AG_News
        "label_skew" : True,
        "label_groups" : [
             [0, 1], [0,2], [0,3], [1,2],  [1,3], [2,3],  # No overlap
            
        ],
        "swaps_lists" : [
            # No swap
        ],
        "num_labels_per_cluster": 6,
        "label_overlap_size": 2,
        "beta": 0.5, # For Dirichlet
        "reserve_frac": 0.1,
        'N_clusters': 5,
        "N_train" : 300, # This is min # of train per cluster
        "N_test" : 300 # This is min # of test per cluster
    },  

    "exp501": { # dominance based label skew
        "dominance_label_skew" : True,
        "label_groups" : [
             [0,1,2,3,4,5,6,7,8,9],  # No overlap
            
        ],
        "swaps_lists" : [
            # No swap
        ],
        "dominance_percent": 66.6, # 2/3rd

    },

    "exp602": { # Simulates bigger overlap, used for ablation
        "label_skew" : True,
        "label_groups" : [
            [0,1,2,3,4,5,6,7,8],  # class labels for this cluster
            [1,2,3,4,5,6,7,8,9],
            [0,2,3,4,5,6,7,8,9],
            [0,1,3,4,5,6,7,8,9],
            [0,1,2,3,4,5,6,7,8,9],
            
        ],
        "swaps_lists" : [
            # No swap
        ],
        "num_labels_per_cluster": 7, # bet 7 and 8
        "label_overlap_size": 5, # approx varies between 4 and 6
        "beta": 0.5, # For Dirichlet
        "reserve_frac": 0.1,
        'N_clusters': 5,
        "N_train" : 6000, # This is min # of train per cluster
        "N_test" : 750 # This is min # of test per cluster
    }, 

    "exp701": { # Simulates bigger overlap, used for ablation
        "label_skew" : True,
        "label_groups" : [
             [0, 1], [2,3], [4,5], [6,7], [8,9], # No overlap
            
        ],
        "swaps_lists" : [
            # No swap
        ],
        "num_labels_per_cluster": 2,
        "label_overlap_size": 0,
        "beta": 0.5, # For Dirichlet
        "reserve_frac": 0.1,
        'N_clusters': 5,
        "N_train" : 6000, # This is min # of train per cluster
        "N_test" : 750 # This is min # of test per cluster
    }, 
}

def get_label_skew_data_split(expid,  train_data_unmixed, test_data_unmixed):
        train_dataset, label_keys = convert_from_dict_to_list(train_data_unmixed)
        test_dataset, _ = convert_from_dict_to_list(test_data_unmixed)
        num_classes = len(label_keys)
        exp_val_dict = dict_overlap_swap_exps[expid]

        label_groups = exp_val_dict["label_groups"]
        swaps_lists = exp_val_dict["swaps_lists"]
        beta = exp_val_dict["beta"]
        reserve_frac = exp_val_dict["reserve_frac"]
        min_N_train = exp_val_dict["N_train"] # Min. Number of train samples per cluster
        min_N_test = exp_val_dict["N_test"] # Min. Number of test samples per cluster
        num_clusters = exp_val_dict["N_clusters"]

        num_labels_per_cluster = None
        label_overlap_size = None
        if len(label_groups) <= 0:
            num_labels_per_cluster = exp_val_dict["num_labels_per_cluster"]
            label_overlap_size = exp_val_dict["label_overlap_size"]
            labels_assignment = generate_label_assignments(num_classes, num_clusters, num_labels_per_cluster, label_overlap_size)
        else:
            labels_assignment = convert_to_label_assignments(label_groups)
            num_clusters = len(labels_assignment)

        train_data, train_group_indices_dict = try_partition_until_min(train_dataset, num_clusters, num_classes, num_labels_per_cluster, label_overlap_size,
                            beta=beta, initial_reserve_pct=reserve_frac, max_reserve_pct=1.0, step=0.05, min_num=min_N_train, label_assignments=labels_assignment)

        test_data, test_group_indices_dict = try_partition_until_min(test_dataset, num_clusters, num_classes, num_labels_per_cluster, label_overlap_size,
                            beta=beta, initial_reserve_pct=reserve_frac, max_reserve_pct=1.0, step=0.05, min_num=min_N_test, label_assignments=labels_assignment)
        
        return train_data, train_group_indices_dict, test_data, test_group_indices_dict, swaps_lists


def get_dominance_data_split(expid, train_data_unmixed, test_data_unmixed):
        train_dataset, label_keys = convert_from_dict_to_list(train_data_unmixed)
        test_dataset, _ = convert_from_dict_to_list(test_data_unmixed)

        exp_val_dict = dict_overlap_swap_exps[expid]

        selected_labels = exp_val_dict["label_groups"][0] # should be of the form [ [selected labels]]
        p = exp_val_dict["dominance_percent"] # Fraction of data in the group that comes from dominant label 
        swaps_lists = exp_val_dict["swaps_lists"]

        train_data, train_group_indices_dict = partition_dataset_dominance_based(train_dataset, selected_labels, p)

        test_data, test_group_indices_dict = partition_dataset_dominance_based(test_dataset, selected_labels, p)
        
        return train_data, train_group_indices_dict, test_data, test_group_indices_dict, swaps_lists


def shuffled_indices(lst):
    indices = list(range(len(lst)))
    random.shuffle(indices)
    return indices


def get_feature_skew_data_split(expid, train_data_unmixed, test_data_unmixed, config_dict, loader_fn):
    train_data = {}
    test_data= {}
    exp_val_dict = dict_overlap_swap_exps[expid]
    angles = exp_val_dict['other_rot_angles']
    data_list=[(train_data_unmixed, test_data_unmixed)]
    for angle in angles:
        new_train, new_test = loader_fn(config_dict, angle)
        data_list.append((new_train, new_test))
    
    ds_list = []
    for data_items in data_list:
        train_dataset, label_keys = convert_from_dict_to_list(data_items[0])
        test_dataset, _ = convert_from_dict_to_list(data_items[1])
        ds_list.append((train_dataset, test_dataset))

    train_group_indices_dict = defaultdict(list)
    test_group_indices_dict = defaultdict(list)

    for idx, ds in enumerate(ds_list):
        train_ds = ds[0]
        test_ds = ds[1]

        indxs = shuffled_indices(train_ds)
        train_group_indices_dict[idx] = indxs
        subset = Subset(train_ds, indxs)
        train_data[idx] = list(subset)

        indxs = shuffled_indices(test_ds)
        test_group_indices_dict[idx] = indxs
        subset = Subset(test_ds, indxs)
        test_data[idx] = list(subset)

    swaps_lists = exp_val_dict["swaps_lists"]
    return train_data, train_group_indices_dict, test_data, test_group_indices_dict, swaps_lists
