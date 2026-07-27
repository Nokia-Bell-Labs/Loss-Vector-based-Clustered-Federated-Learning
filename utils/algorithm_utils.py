# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import itertools
import random
import torch
from math import prod
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from finch import FINCH, finch_fit

from model import get_model
from utils.learning_utils import get_single_model_parameters, set_single_model_parameters, parameters_to_arrays, wavg_aggregate, train, test, compute_gradient, get_custom_optimizer, set_up_LR_scheduler

from utils.logging_utils import logging
logger = logging.getLogger(__name__)
np.set_printoptions(legacy='1.25')


def find_best_K_by_silhouette(X, K_range=(3, 15), linkage='ward', random_state=0):
    """
    Finds the best number of clusters (K) using AgglomerativeClustering and silhouette score.

    Parameters:
        X (array-like): Feature matrix.
        K_range (tuple): (min_K, max_K) range to search.
        linkage (str): Linkage method ('ward', 'complete', 'average', 'single').
        random_state (int): Random state to use as seed.

    Returns:
        - best_K (int)
        - best_score (float)
        - best_labels (np.ndarray)
    """
    best_score = -1
    best_K = None
    best_labels = None

    logger.info("\nSearching for best K value:")

    if K_range[1] >= len(X):
        X = np.repeat(X, repeats=5, axis=0)     # Duplicate data points so that enough data points are present

    for K in range(K_range[0], K_range[1] + 1):
        model = AgglomerativeClustering(n_clusters=K, linkage=linkage)
        labels = model.fit_predict(X)

        if len(set(labels)) < 2:
            continue  # Skip degenerate clusterings

        score = silhouette_score(X, labels, random_state=random_state)
        logger.info(f"For K = {K} candidate clusters, silhouette score = {score}.")

        if score > best_score * 1.10:  # We add a tolerance of 10% to avoid picking a higher K that has only a marginally better silhouette score than a lower K (which would be more parsimonious and less likely to overfit)
            best_score = score
            best_K = K
            best_labels = labels

    if best_labels is None:
        raise ValueError("No valid clustering found in the specified K range.")

    logger.info(f"Best number of clusters K: {best_K}")
    logger.info(f"Best score: {best_score}")
    logger.info(f"Best labels: {best_labels}")

    return best_K, best_score, best_labels


def centralized_only_training_function(config_dict, net, all_data_trainloader, testloaders, initial_parameters, ground_truth):
    metrics_per_round = {}
    all_model_parameters = initial_parameters[0]
    set_single_model_parameters(net, all_model_parameters)

    selected_model_to_client_mapping_dict = {0: i for i in range(config_dict['number_of_clients'])}
    client_to_selected_model_mapping_dict = {i: 0 for i in range(config_dict['number_of_clients'])}

    for current_round in range(config_dict['number_of_rounds']):
        logger.info("")
        logger.info(f"Beginning of round {current_round}:")

        if config_dict['lr_schedule_enabled'] is True:
            config_dict = set_up_LR_scheduler(config_dict)

        round_loss = train(
            net=net,
            trainloader=all_data_trainloader,
            config=config_dict,
            client_index=0,
            model_index=0,
        )
        logger.info(f"Round {current_round}: Train loss of centralized model = {round_loss}")
        all_model_parameters = {0: get_single_model_parameters(net)}

        ## Evaluation of all models on each client's test data and calculation of metrics ##
        metric_results = calculate_metrics(config_dict, all_model_parameters, testloaders, client_to_selected_model_mapping_dict, ground_truth)
        metric_results['train_loss'] = round_loss
        metric_results['labels_pred_train'] = list(client_to_selected_model_mapping_dict.values())
        metric_results['client_to_selected_model_mapping_dict'] = client_to_selected_model_mapping_dict
        metric_results['selected_model_to_client_mapping_dict'] = selected_model_to_client_mapping_dict
        for metric in list(metric_results.keys()):
            if metric not in metrics_per_round.keys():
                metrics_per_round[metric] = []
            metrics_per_round[metric].append(metric_results[metric])

    return all_model_parameters, client_to_selected_model_mapping_dict, selected_model_to_client_mapping_dict, metrics_per_round


def local_only_training_function(config_dict, nets, trainloaders, testloaders, initial_parameters, ground_truth):
    all_model_parameters = initial_parameters
    metrics_per_round = {}

    selected_model_to_client_mapping_dict = {i: i for i in range(config_dict['number_of_clients'])}
    client_to_selected_model_mapping_dict = {i: i for i in range(config_dict['number_of_clients'])}

    for current_round in range(config_dict['number_of_rounds']):
        logger.info("")
        logger.info(f"Beginning of round {current_round}:")
        participating_clients = list(range(config_dict['number_of_clients']))   # For local only training, all clients participate at all rounds.

        if config_dict['lr_schedule_enabled'] is True:
            config_dict = set_up_LR_scheduler(config_dict)

        round_losses = {}
        for client_index in participating_clients:
            selected_model_index = client_to_selected_model_mapping_dict[client_index]
            set_single_model_parameters(nets[selected_model_index], all_model_parameters[selected_model_index])     # We set the parameters, and train() internally zeroes out the gradients, so what was inside the net from previous evaluation/training steps does not matter.
            round_loss = train(
                net=nets[selected_model_index],
                trainloader=trainloaders[client_index],
                config=config_dict,
                client_index=client_index,
                model_index=selected_model_index
            )
            logger.info(f"Round {current_round}: Train loss of client {client_index} on model {selected_model_index} = {round_loss}")
            all_model_parameters[client_index] = get_single_model_parameters(nets[selected_model_index])
            round_losses[client_index] = round_loss

        ## Evaluation of all models on each client's test data and calculation of metrics ##
        metric_results = calculate_metrics(config_dict, all_model_parameters, testloaders, client_to_selected_model_mapping_dict, ground_truth)
        for client_index in range(config_dict['number_of_clients']):    # log train loss for all clients (None if client did not participate in this round)
            metric_results[f'train_loss_client_{client_index}'] = round_losses[client_index] if client_index in participating_clients else None
        metric_results['labels_pred_train'] = list(client_to_selected_model_mapping_dict.values())
        metric_results['client_to_selected_model_mapping_dict'] = client_to_selected_model_mapping_dict
        metric_results['selected_model_to_client_mapping_dict'] = selected_model_to_client_mapping_dict
        for metric in list(metric_results.keys()):
            if metric not in metrics_per_round.keys():
                metrics_per_round[metric] = []
            metrics_per_round[metric].append(metric_results[metric])

    return all_model_parameters, client_to_selected_model_mapping_dict, selected_model_to_client_mapping_dict, metrics_per_round


def clustered_federated_training_function(config_dict, nets, trainloaders, testloaders, initial_parameters, ground_truth):
    all_model_parameters = initial_parameters
    metrics_per_round = {}
    client_to_selected_model_mapping_dict = dict()
    selected_model_to_client_mapping_dict = dict()
    # previous_client_to_selected_model_mapping_dict = None
    client_to_selected_model_mapping_history_dict = {client_index: [] for client_index in range(config_dict['number_of_clients'])}
    # consecutive_stable_rounds = 0
    clustering_stability_achieved = False
    full_client_to_selected_model_mapping_dict = dict()
    seen_client_IDs = [False] * config_dict['number_of_clients']
    all_seen_client_indices = []

    num_nets = len(nets)
    centroids = np.zeros((num_nets, num_nets))

    for current_round in range(config_dict['number_of_rounds']):
        logger.info("")
        logger.info(f"Beginning of round {current_round}:\n")

        participating_clients = random.sample(population=range(config_dict['number_of_clients']), k=int(config_dict['fraction_fit'] * config_dict['number_of_clients']))
        if config_dict['fraction_fit'] < 1.0:
            participating_clients.sort()
            logger.info(f"Participating clients: {participating_clients}")
            logger.info(f"Non-participating clients: {list(set(range(config_dict['number_of_clients'])) - set(participating_clients))}")

        # Mark all participating clients as "seen" by the algorithm
        for client_index in participating_clients:
            seen_client_IDs[client_index] = True

        # Print history for all clients
        for client_index in range(config_dict['number_of_clients']):
            client_history = client_to_selected_model_mapping_history_dict[client_index]
            logger.info(f"Client {client_index}: history: {client_history}")

        ## Check if the client-to-cluster assignment has stabilized ##
        if config_dict['early_stopping_of_clustering'] is True and clustering_stability_achieved is False:
            # Global way: all client to model assignments have been stable for X rounds
            # if client_to_selected_model_mapping_dict == previous_client_to_selected_model_mapping_dict:
            #     consecutive_stable_rounds += 1
            # else:
            #     consecutive_stable_rounds = 0
            # previous_client_to_selected_model_mapping_dict = client_to_selected_model_mapping_dict

            # logger.info(f"Consecutive stable rounds: {consecutive_stable_rounds} out of the {config_dict['number_of_rounds_for_clustering_stability']} required for stability.")
            # clustering_stability_achieved = consecutive_stable_rounds >= config_dict['number_of_rounds_for_clustering_stability']

            # # Print history for all clients
            # for client_index in range(config_dict['number_of_clients']):
            #     client_history = client_to_selected_model_mapping_history_dict[client_index]
            #     logger.info(f"Client {client_index}: history: {client_history}")

            # Fine-grained way: each of the last participated clients has been stable for X rounds
            client_is_stable = {client_index: True for client_index in participating_clients}
            for client_index in participating_clients:
                client_history = client_to_selected_model_mapping_history_dict[client_index]
                # logger.info(f"Client {client_index}: history: {client_history}")
                # client_is_stable = True
                r = 0
                while r < config_dict['number_of_rounds_for_clustering_stability'] and client_is_stable:
                    if len(client_history) >= config_dict['number_of_rounds_for_clustering_stability']:
                        client_is_stable[client_index] = client_history[-1-r] == client_history[-1]
                    else:
                        client_is_stable[client_index] = False
                    r += 1
                # if not client_is_stable:
                #     clustering_stability_achieved = False
                #     break
                # else:
                #     number_of_stable_clients += 1
            number_of_stable_clients = sum(client_is_stable.values())
            clustering_stability_achieved = all(client_is_stable.values())
            if clustering_stability_achieved:
                # Redefine the (full) client to model mapping to include the last model all of the clients that have participated so far were assigned to
                all_seen_client_indices = [i for i, val in enumerate(seen_client_IDs) if val]
                full_client_to_selected_model_mapping_dict = {client_index: client_to_selected_model_mapping_history_dict[client_index][-1] for client_index in all_seen_client_indices}
            logger.info(f"Clients stable for {config_dict['number_of_rounds_for_clustering_stability']} rounds: {number_of_stable_clients} out of the {len(participating_clients)} participating.")

        if config_dict['early_stopping_of_clustering'] is True and clustering_stability_achieved is True:
            # pass        # client_to_selected_model_mapping_dict is already populated from the previous round
            client_to_selected_model_mapping_dict = {client_index: full_client_to_selected_model_mapping_dict[client_index] for client_index in participating_clients}
            # selected_model_to_client_mapping_dict = {model_index: [] for model_index in range(config_dict['total_number_of_models'])}
            selected_model_to_client_mapping_dict = {}
            # if config_dict['K_is_unknown']:
            #     config_dict['total_number_of_models'] = config_dict['number_of_models']     # We can discard the rest of the models, as they will not be used anymore
            for client_index, model_index in client_to_selected_model_mapping_dict.items():
                if model_index not in selected_model_to_client_mapping_dict.keys():
                    selected_model_to_client_mapping_dict[model_index] = []
                selected_model_to_client_mapping_dict[model_index].append(client_index)

            # Deal with unseen clients that just arrived (i.e. arrived after stability was reached)
            for client_index in participating_clients:
                if seen_client_IDs[client_index] is False:
                    loss_vector = generate_loss_vectors(config_dict, all_model_parameters, testloaders, participating_clients=[client_index])
                    loss_vector = loss_vector[client_index]
                    distances = np.linalg.norm(centroids - loss_vector, axis=1)
                    best_model_index = np.argmin(distances)
                    client_to_selected_model_mapping_dict[client_index] = best_model_index
                    selected_model_to_client_mapping_dict[best_model_index].append(client_index)
                    all_seen_client_indices.append(client_index)
                    logger.info(f"Client {client_index} participated for the first time after stability was achieved and was assigned to model {best_model_index}.")

            logger.info("Cluster stability has been achieved. Not computing loss vectors and modifying clusters anymore.")
            logger.info(f"selected_model_to_client_mapping = {selected_model_to_client_mapping_dict}")
        else:
            ## Evaluation step of all models on each client's training data ##
            loss_vectors_dict = generate_loss_vectors(
                config_dict=config_dict,
                all_model_parameters=all_model_parameters,
                dataloaders=trainloaders,
                participating_clients=participating_clients
            )

            logger.info("")
            logger.info("The following losses are on training dataset")
            logger.info(f'Loss vectors =')
            loss_vectors_printable = [[round(elem, 7) for elem in row] for row in loss_vectors_dict.values()]
            logger.info('\n'.join([str(row) for row in loss_vectors_printable]))


            ## Client to model assignment step (centrally at the server for CLoVE, locally for the other algorithms) ##
            if not config_dict['first_round_client_to_cluster_assignment'] in ['evaluation_based', 'random']:
                raise ValueError(f"Invalid value for config_dict['first_round_client_to_cluster_assignment']: {config_dict['first_round_client_to_cluster_assignment']}")

            selected_model_to_client_mapping_dict = {}
            client_to_selected_model_mapping_dict = {}
            if (current_round == 0) and (config_dict['first_round_client_to_cluster_assignment'] == 'random'):  # if this is the first round and the assignment is set to random, then ignore the previously computed loss vectors and assign clients to clusters randomly
                selected_models_vector = random.choices(population=list(range(config_dict['number_of_models'])), k=len(participating_clients))
                for model_index in range(config_dict['number_of_models']):
                    if model_index not in selected_models_vector:  # if some model got assigned 0 clients, then assign a random client to it
                        selected_models_vector[random.randint(0, len(selected_models_vector) - 1)] = model_index
                client_clusters = [[participating_clients[item[0]] for item in clients_in_group] for model_index, clients_in_group in itertools.groupby(sorted(enumerate(selected_models_vector), key=lambda x: x[1]), lambda x: x[1])]
                client_clusters.sort()  # Sort clusters according to the smallest client ID in each of them
                selected_model_to_client_mapping_dict = {model_index: cluster for model_index, cluster in enumerate(client_clusters)}
                for cluster_index, client_cluster in enumerate(client_clusters):
                    for client in client_cluster:
                        client_to_selected_model_mapping_dict[client] = cluster_index
                        client_to_selected_model_mapping_history_dict[client].append(cluster_index)
                logger.info(f"Initial random selected_model_to_client_mapping = {selected_model_to_client_mapping_dict}")


            else:  # for server_round >= 1, or if server_round == 0 and self.config_dict['first_round_client_to_cluster_assignment'] == 'evaluation_based'
                if config_dict['algorithm'] in ['IFCA', 'vanillaFL']:
                    for client_index in participating_clients:
                        if config_dict['algorithm'] == 'IFCA':
                            best_model_index, selected_model_index = select_model_IFCA(model_losses_on_client_data=loss_vectors_dict[client_index])
                        elif config_dict['algorithm'] == 'vanillaFL':
                            if config_dict["number_of_models"] != 1:
                                raise ValueError(f"Invalid value for config_dict['number_of_models']: {config_dict['number_of_models']}. It has to be 1, since config_dict['algorithm'] is vanillaFL.")
                            best_model_index, selected_model_index = 0, 0
                        else:
                            raise ValueError(f"Invalid value given for config_dict['algorithm']: {config_dict['algorithm']}")

                        # logger.info(f"best_model_index = {best_model_index}")
                        # logger.info(f"selected_model_index = {selected_model_index}\n")

                        client_to_selected_model_mapping_dict[client_index] = selected_model_index
                        selected_model_to_client_mapping_dict = {
                            int(model_index): [item[0] for item in clients_with_this_selected_model]
                            for model_index, clients_with_this_selected_model in itertools.groupby(sorted(client_to_selected_model_mapping_dict.items(), key=lambda x: x[1]), lambda x: x[1])
                        }
                elif config_dict['algorithm'] == 'CLoVE':
                    if (config_dict["replicate_model_parameters_at_some_round"] is True) and (current_round == config_dict["round_of_replication"]):
                        model_index = get_most_used_model(client_to_selected_model_mapping_history_dict)
                        all_model_parameters = replicate_model_parameters(all_model_parameters, model_index)
                        logger.info(f"Replicated model parameters for model {model_index}.")
                    modified_loss_vectors_dict = loss_vectors_dict
                    if config_dict['loss_transform'] == 'square_root':
                        modified_loss_vectors_dict = {key: np.sqrt(value) for key, value in modified_loss_vectors_dict.items()}

                    modified_loss_vectors_array = np.array(list(modified_loss_vectors_dict.values()))

                    
                    if config_dict['K_is_unknown']:
                        if config_dict['clustering_alg'] != 'FINCH':
                            best_K, best_score, best_labels = find_best_K_by_silhouette(
                                modified_loss_vectors_array,
                                K_range=(2, min(len(participating_clients), config_dict['upper_bound_on_K'])),
                                random_state=config_dict['seed']
                            )
                        else:
                            c, best_K_list, req_c = FINCH(data=modified_loss_vectors_array,  distance='euclidean', verbose=False)
                            best_K = best_K_list[0]
                            
                        config_dict['number_of_models'] = best_K

                    ## Clustering step at server ##
                    if config_dict['clustering_alg'] == 'KMEANS':
                        clustering = KMeans(n_clusters=config_dict["number_of_models"], random_state=config_dict['seed'], n_init="auto").fit(modified_loss_vectors_array)
                    elif config_dict['clustering_alg'] == 'FINCH':
                        clustering = finch_fit(modified_loss_vectors_array, req_clust=config_dict["number_of_models"], distance='euclidean', verbose=False)
                    elif config_dict['clustering_alg'] == 'AgglomerativeClustering':
                        clustering = AgglomerativeClustering(n_clusters=config_dict["number_of_models"], linkage='ward', metric='euclidean').fit(modified_loss_vectors_array)
                    else:
                        raise ValueError(f"Invalid value given for config_dict['clustering_alg']: {config_dict['clustering_alg']}")

                    centroids = clustering.cluster_centers_

                    selected_model_to_client_mapping_dict, client_to_selected_model_mapping_dict = match_clusters_to_models(config_dict, clustering.labels_, modified_loss_vectors_dict, participating_clients)

                    for client_index, model_index in client_to_selected_model_mapping_dict.items():
                        client_to_selected_model_mapping_history_dict[client_index].append(model_index)

                else:
                    raise ValueError(f"Invalid value given for config_dict['algorithm']: {config_dict['algorithm']}")
                logger.info(f"Loss-based selected_model_to_client_mapping = {selected_model_to_client_mapping_dict}")


        ## Training step at each client on the selected model ##
        if config_dict['lr_schedule_enabled'] is True:
            config_dict = set_up_LR_scheduler(config_dict)

        updated_model_parameters = {}
        updated_gradients = {}
        round_losses = {}
        for client_index in participating_clients:
            selected_model_index = client_to_selected_model_mapping_dict[client_index]
            set_single_model_parameters(nets[selected_model_index], all_model_parameters[selected_model_index])     # We set the parameters, and train() internally zeros out the gradients, so what was inside the net from previous evaluation/training steps does not matter.
            if config_dict['averaging_mode'] == 'model_averaging':
                round_loss = train(
                    net=nets[selected_model_index],
                    trainloader=trainloaders[client_index],
                    config=config_dict,
                    client_index=client_index,
                    model_index=selected_model_index
                )
                logger.info(f"Round {current_round}: Train loss of client {client_index} on selected model {selected_model_index} = {round_loss}")
                updated_model_parameters[client_index] = get_single_model_parameters(nets[selected_model_index])
                round_losses[client_index] = round_loss
            elif config_dict['averaging_mode'] == 'gradient_averaging':
                gradient_of_selected_model = compute_gradient(
                    net=nets[selected_model_index],
                    trainloader=trainloaders[client_index],
                    config=config_dict,
                    client_index=client_index,
                    model_index=selected_model_index
                )
                updated_gradients[client_index] = gradient_of_selected_model
            else:
                raise ValueError(f"Invalid value given for config_dict['averaging_mode']: {config_dict['averaging_mode']}")


        ## Aggregation step at server ##
        for j in range(num_nets):
            if j in selected_model_to_client_mapping_dict.keys():    # if model j has been updated by any client at the current round, aggregate the weights of the clients that used it (else do nothing and model j keeps its parameters from the previous round)

                logger.info(f'============= Server: Aggregating model {j} based on local models of clients: {sorted(selected_model_to_client_mapping_dict[j])}.')

                # Fetch client data sizes for clients that used model j
                train_data_sizes_for_aggregation = [len(trainloaders[client_index].dataset) for client_index in participating_clients if client_index in selected_model_to_client_mapping_dict[j]]

                if len(train_data_sizes_for_aggregation) <= 0:
                    # No client mapped to this model. We should not update this model. Rather let us keep it same.
                    logger.info("No client mapped to this model. Not applying updates to it")
                    continue

                if config_dict['averaging_mode'] == 'model_averaging':
                    # Fetch updated weights from clients that used model j
                    parameters_for_model_j_to_be_aggregated = [updated_model_parameters[client_index] for client_index in participating_clients if client_index in selected_model_to_client_mapping_dict[j]]

                    # Aggregate the weights of the clients that used model j
                    all_model_parameters[j] = wavg_aggregate(parameters_for_model_j_to_be_aggregated, train_data_sizes_for_aggregation)
                elif config_dict['averaging_mode'] == 'gradient_averaging':
                    # Fetch updated gradients from clients that used model j
                    gradients_for_model_j_to_be_aggregated = [updated_gradients[client_index] for client_index in participating_clients if client_index in selected_model_to_client_mapping_dict[j]]

                    # Aggregate the gradients of the clients that used model j
                    gradients_aggregated_for_model_j = wavg_aggregate(gradients_for_model_j_to_be_aggregated, train_data_sizes_for_aggregation)

                    params = [torch.nn.Parameter(torch.tensor(layer, requires_grad=True)) for layer in all_model_parameters[j]]
                    for layer_index, layer in enumerate(params):
                        layer.grad = torch.tensor(gradients_aggregated_for_model_j[layer_index])
                    optimizer = get_custom_optimizer(iter(params), config_dict)
                    optimizer.step()
                    all_model_parameters[j] = parameters_to_arrays(params)
                else:
                    raise ValueError(f"Invalid value given for config_dict['averaging_mode']: {config_dict['averaging_mode']}")


        ## Evaluation of all models on each client's test data and calculation of metrics ##
        last_model_dict = get_last_model_dict(client_to_selected_model_mapping_history_dict)
        if len(last_model_dict) == 0:
            # Fall back to the current round assignment when history is not populated yet.
            last_model_dict = client_to_selected_model_mapping_dict.copy()
        metric_results = calculate_metrics(config_dict, all_model_parameters, testloaders, last_model_dict, ground_truth, participating_clients)
        for client_index in range(config_dict['number_of_clients']):    # log train loss for all clients (None if client did not participate in this round)
            metric_results[f'train_loss_client_{client_index}'] = round_losses[client_index] if ((client_index in participating_clients) and (config_dict['averaging_mode'] == 'model_averaging')) else None
        metric_results['labels_pred_train'] = list(client_to_selected_model_mapping_dict.values())
        metric_results['client_to_selected_model_mapping_dict'] = client_to_selected_model_mapping_dict
        metric_results['selected_model_to_client_mapping_dict'] = selected_model_to_client_mapping_dict
        for metric in list(metric_results.keys()):
            if metric not in metrics_per_round.keys():
                metrics_per_round[metric] = []
            metrics_per_round[metric].append(metric_results[metric])

    return all_model_parameters, client_to_selected_model_mapping_dict, selected_model_to_client_mapping_dict, metrics_per_round


def select_model_IFCA(model_losses_on_client_data):
    best_model_index = np.argmin(model_losses_on_client_data)
    selected_model_index = best_model_index

    return best_model_index, selected_model_index


def match_clusters_to_models(config_dict, clustering_labels, loss_vectors_dict, participating_clients):
    client_index_clusters = [[item[0] for item in clients_in_group] for model_index, clients_in_group in itertools.groupby(sorted(enumerate(clustering_labels), key=lambda x: x[1]), lambda x: x[1])]     # resulting clusters contain IDs based on the enumerate operation, i.e. coming an increasing index that does not necessarily coincide with the client IDs. We fix this in the next line.
    client_clusters = [[participating_clients[c] for c in client_index_cluster] for client_index_cluster in client_index_clusters]
    client_clusters.sort()  # Sort clusters according to the smallest client ID in each of them

    if config_dict['cluster_to_model_matching_method'] == 'sort_clusters_wrt_min_client_ID':
        cluster_to_selected_model_matching = list(range(config_dict['total_number_of_models']))

    elif config_dict['cluster_to_model_matching_method'] == 'min_cost_matching_wrt_total_cluster_loss':
        biadjacency_matrix = np.zeros(shape=(config_dict['number_of_models'], config_dict['total_number_of_models']))
        for model_index in range(config_dict['total_number_of_models']):
            for cluster_ID, cluster in enumerate(client_clusters):
                for client_ID in cluster:
                    biadjacency_matrix[cluster_ID, model_index] += loss_vectors_dict[client_ID][model_index]

        cluster_to_selected_model_matching = linear_sum_assignment(biadjacency_matrix)[1]

    elif config_dict['cluster_to_model_matching_method'] == 'min_cost_matching_wrt_cluster_overlap_with_previous_clusters':
        raise NotImplementedError(f"The option for config_dict['cluster_to_model_matching_method'] == {config_dict['cluster_to_model_matching_method']} has not been implemented yet.")

    else:
        raise ValueError(f"Invalid value for config_dict['cluster_to_model_matching_method']: {config_dict['cluster_to_model_matching_method']}")

    client_to_selected_model_mapping_dict = {}
    for cluster_index, client_cluster in enumerate(client_clusters):
        for client in client_cluster:
            client_to_selected_model_mapping_dict[client] = cluster_to_selected_model_matching[cluster_index]

    selected_model_to_client_mapping_dict = {model_index: client_clusters[cluster_ID] for cluster_ID, model_index in enumerate(cluster_to_selected_model_matching)}

    return selected_model_to_client_mapping_dict, client_to_selected_model_mapping_dict


def generate_loss_vectors(config_dict, all_model_parameters, dataloaders, participating_clients):
    """Evaluates each of the given models on each of the given client test data"""
    data_point_dimension = prod(dataloaders[0].dataset[0][0].shape)
    config_dict['autoencoder_input_size'] = data_point_dimension
    net = get_model(config_dict)    # Only one model object is needed, as we feed the appropriate parameters into it each time and save them after each step. So the model object carries no history that we use.

    actual_number_of_models = len(all_model_parameters.keys())      # Might be different from config_dict['number_of_models'] if the training mode is centralized or local only.
    loss_vectors_dict = {client_ID: [] for client_ID in participating_clients}

    for model_index in range(actual_number_of_models):
        set_single_model_parameters(net, all_model_parameters[model_index])     # We set the parameters, and train() internally zeros out the gradients, so what was inside the net from previous evaluation/training steps does not matter.
        for client_index in participating_clients:
            test_loss = test(
                net=net,
                testloader=dataloaders[client_index],
                config=config_dict,
                client_index=client_index,
                model_index=model_index,
            )
            loss_vectors_dict[client_index].append(test_loss)

    return loss_vectors_dict


def calculate_metrics(config_dict, all_model_parameters, testloaders, full_client_to_selected_model_mapping_dict, ground_truth, participating_clients=None):
    if participating_clients is None:
        participating_clients = range(config_dict['number_of_clients'])
    all_seen_clients = [int(client_id) for client_id in full_client_to_selected_model_mapping_dict.keys()]
    # Generate the loss matrix of the data of each client (row) tested on each model (column)
    loss_vectors_dict = generate_loss_vectors(config_dict, all_model_parameters, testloaders, all_seen_clients)
    # Set each client's cluster label based on its test data to the cluster/model for which it has the lowest loss
    loss_matrix = np.array(list(loss_vectors_dict.values()))
    labels_pred_test = np.argmin(loss_matrix, axis=1).tolist()
    labels_pred_train = list(full_client_to_selected_model_mapping_dict.values())
    labels_true = [ground_truth[client_ID] for client_ID in all_seen_clients]

    # Define the metrics which require only the true labels and estimated labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.rand_score,
        metrics.adjusted_rand_score,
        metrics.mutual_info_score,
        metrics.adjusted_mutual_info_score,
    ]
    metric_names = [m.__name__ for m in clustering_metrics]
    metric_results = {}
    metric_results.update({metric_name + "_train": metric(labels_true, labels_pred_train) for metric_name, metric in zip(metric_names, clustering_metrics)})
    metric_results.update({metric_name + "_test": metric(labels_true, labels_pred_test) for metric_name, metric in zip(metric_names, clustering_metrics)})
    metric_results['labels_pred_test'] = labels_pred_test
    metric_results['loss_matrix_test'] = loss_matrix.tolist()

    return metric_results


def get_last_model_dict(client_to_selected_model_mapping_history_dict):
    last_model_dict = {client_index: client_to_selected_model_mapping_history_dict[client_index][-1]
                       for client_index in client_to_selected_model_mapping_history_dict.keys()
                       if len(client_to_selected_model_mapping_history_dict[client_index]) > 0}
    return last_model_dict


def get_most_used_model(client_to_selected_model_mapping_history_dict):
    """
    Finds the model ID that appears in the most client history lists.

    Parameters:
        client_to_selected_model_mapping_history_dict (dict): Dictionary mapping client indices to lists of model IDs.

    Returns:
        int: The model ID that appears in the most client lists.
    """
    model_count = {}

    # Count how many clients have each model in their history
    for client_index, model_history in client_to_selected_model_mapping_history_dict.items():
        # Get unique models for this client to count each model only once per client
        unique_models_for_client = set(model_history)
        for model_id in unique_models_for_client:
            model_count[model_id] = model_count.get(model_id, 0) + 1

    # Find the model that appears in the most client lists
    if not model_count:
        return None

    most_used_model_id = max(model_count, key=model_count.get)
    return most_used_model_id


def replicate_model_parameters(all_model_parameters, selected_model_id):
    """
    Replicates the model parameters for a specific model ID across all clients.

    Parameters:
        all_model_parameters (dict): Dictionary mapping model IDs to their parameters.
        selected_model_id (int): The model ID to replicate.

    Returns:
        dict: A dictionary mapping client indices to the replicated model parameters.
    """
    if selected_model_id not in all_model_parameters:
        logger.warning(f"Model ID {selected_model_id} not found in all_model_parameters.")
        return {}

    replicated_parameters = all_model_parameters[selected_model_id]
    for model_id in all_model_parameters.keys():
        all_model_parameters[model_id] = replicated_parameters

    return all_model_parameters
