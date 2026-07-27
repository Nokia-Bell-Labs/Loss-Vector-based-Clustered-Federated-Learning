# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from functools import reduce
from collections import OrderedDict
from typing import Dict, Union, List, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ml_collections import ConfigDict


from utils.logging_utils import logging
logger = logging.getLogger(__name__)


def parameters_to_arrays(params: List[torch.nn.Parameter]) -> List[np.ndarray]:
    return [layer_params.clone().detach().cpu().numpy() for layer_params in params]


def get_single_model_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    return [val.clone().cpu().numpy() for _, val in net.state_dict().items()]


def get_single_model_gradients(net: torch.nn.Module) -> List[np.ndarray]:
    gradient_tensors = [(param.grad if param.grad is not None else torch.zeros_like(param, dtype=torch.float32)) for name, param in net.named_parameters()]
    gradient_ndarrays = [gradient_tensor.clone().detach().cpu().numpy() for gradient_tensor in gradient_tensors]
    return gradient_ndarrays


def get_multiple_model_parameters(config: Dict[str, Union[int, np.ndarray]], nets: List) -> List[np.ndarray]:
    """Returns a long list of the concatenation of the parameters of all neural nets."""

    params_to_return = []
    start = 0
    for j in range(config["number_of_models"]):
        j_th_model_params = get_single_model_parameters(nets[j])
        params_to_return.append(j_th_model_params)
        config["start_indices"][j] = start
        config["end_indices"][j] = start + len(params_to_return[j])
        start = config["end_indices"][j]
    return list(itertools.chain(*params_to_return))


def get_multiple_model_parameters_and_gradients(config: Dict[str, Union[int, np.ndarray]], nets: List) -> List[np.ndarray]:
    """
    Returns a long list of the concatenation of the parameters of all neural nets and the concatenation of the gradients of all neural nets.
    The start and end indices apply for indexing the parameters of each model as they are (e.g. [config["start_indices"][j] : config["start_indices"][j]] for model j), and for indexing the gradients of each model if we add an offset of len(params_to_return) (e.g. [config["start_indices"][j] + len(params_to_return) : config["start_indices"][j] + len(params_to_return)] for model j).
    """

    params_to_return = []
    grads_to_return = []
    start = 0
    for j in range(config["number_of_models"]):
        j_th_model_params = get_single_model_parameters(nets[j])
        params_to_return.append(j_th_model_params)
        j_th_model_grads = get_single_model_gradients(nets[j])
        grads_to_return.append(j_th_model_grads)
        config["start_indices"][j] = start
        config["end_indices"][j] = start + len(params_to_return[j])
        start = config["end_indices"][j]
    return list(itertools.chain(*params_to_return, *grads_to_return))


def get_multiple_model_gradients(config: Dict[str, Union[int, np.ndarray]], nets: List) -> List[np.ndarray]:
    """Returns a long list of the concatenation of the gradients of all neural nets."""

    params_to_return = []
    grads_to_return = []
    start = 0
    for j in range(config["number_of_models"]):
        j_th_model_grads = get_single_model_gradients(nets[j])
        grads_to_return.append(j_th_model_grads)
        config["start_indices"][j] = start
        config["end_indices"][j] = start + len(params_to_return[j])
        start = config["end_indices"][j]
    return list(itertools.chain(*grads_to_return))


def set_single_model_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    # Convert NumPy arrays back to tensors with proper shapes
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32).reshape(net.state_dict()[k].shape) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def wavg_aggregate(arrays: List[List[np.ndarray]], weights: List[int]):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    total_weight = sum(weights)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_arrays = [
        [layer * num_examples for layer in array] for array, num_examples in zip(arrays, weights)
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / total_weight
        for layer_updates in zip(*weighted_arrays)     # layer_updates are the weights of one particular layer from all models
    ]
    return weights_prime


def train(
    net: torch.nn.Module,
    trainloader: DataLoader,
    config: Union[ConfigDict, Dict[str, Union[bool, bytes, float, int, str]]],
    client_index,
    model_index,
) -> float:
    """Train the neural network on the training set."""

    criterion = config['criterion'][config['model_type']]

    optimizer = get_custom_optimizer(net.parameters(), config)

    if config['lr_schedule_enabled'] is True:
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=config['lr_scheduler_step_size'],
            gamma=config['lr_scheduler_gamma']
        )
        optimizer.param_groups[0]['lr'] = config['latest_learning_rate'][client_index]
        if config['latest_scheduler_state_dict'][client_index] is not None:
            scheduler.load_state_dict(config['latest_scheduler_state_dict'][client_index])

    epoch_loss = np.inf

    net.train()
    for epoch in range(config['local_epochs']):
        epoch_loss = 0.0
        accuracy = 0.0
        for inputs, labels in trainloader:
            if config['dataset'] in ['AmazonReview', 'AG_news']:       # text-based datasets
                pass
            else:   # image-based datasets
                inputs = inputs.to(torch.float32)
                labels = labels.to(torch.float32)
            if config['dataset'] == 'CIFAR10' and config['model_type'] == 'AE':
                inputs = inputs / 255.0  # Normalize to [0, 1]
                inputs = inputs.view(inputs.size(0), -1)    # Flatten inputs to feed to autoencoder
            # # Enable storing of gradients if in gradient averaging mode
            # if config['averaging_mode'] == 'gradient_averaging':
            #     inputs.requires_grad = True
            # Load data to the active device
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
            # Reset the gradients back to zero
            optimizer.zero_grad()
            # Compute outputs
            outputs = net(inputs)
            # Compute loss
            if config['model_type'] == 'AE':    # Unsupervised case: loss computed based on the inputs
                loss = criterion(outputs, inputs)
            elif config['model_type'] in ['CNN', 'Cifar10CNN', 'GenConvNet', 'AmazonMLP', 'TextCNN']:  # Supervised case: loss computed based on the labels
                loss = criterion(outputs, labels.long())
            elif config['model_type'] == 'linear_regression':  # Supervised case: loss computed based on the labels
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
            else:
                raise ValueError(f"Invalid value given for config_dict['model_type']: {config['model_type']}")
            # Compute accumulated gradients
            loss.backward()
            # Perform parameter update based on current gradients
            optimizer.step()
            # Update metrics
            if config['model_type'] == 'AE':    # Unsupervised case: loss computed based on the inputs
                epoch_loss += sum([criterion(inputs[i], outputs[i]).tolist() for i in range(len(inputs))])
            elif config['model_type'] in ['CNN', 'Cifar10CNN', 'GenConvNet', 'AmazonMLP', 'TextCNN']:  # Supervised case: loss computed based on the labels
                epoch_loss += sum([criterion(outputs[i], labels[i].long()).tolist() for i in range(len(labels))])
                # logger.info(f'epoch_loss_add = {sum([criterion(outputs[i], labels[i].long()).tolist() for i in range(len(labels))])}')
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # logger.info(f"PROBABILITIES={probabilities}")
                predicted_labels = torch.argmax(probabilities, dim=1).to(torch.float32)
                accuracy += (predicted_labels == labels).sum().item()
            elif config['model_type'] == 'linear_regression':  # Supervised case: loss computed based on the labels
                epoch_loss += sum([criterion(labels[i], outputs[i]).tolist() for i in range(len(labels))])
            else:
                raise ValueError(f"Invalid value given for config_dict['model_type']: {config['model_type']}")

        epoch_loss /= len(trainloader.dataset)
        accuracy /= len(trainloader.dataset)

        if config["lr_schedule_enabled"] is True:
            scheduler.step()
            logger.debug(f"Epoch {epoch}: learning rate = {scheduler.get_last_lr()[0]}")
        logger.debug(f"Training client {client_index}, model {model_index}, epoch {epoch + 1}: train loss = {epoch_loss}")
        logger.debug(f"Training client {client_index}, model {model_index}, epoch {epoch + 1}: train accuracy = {accuracy}")
        # if writer is not None:
        #     writer.add_scalar('train/train loss - epochs', epoch_loss, (current_round - 1) * epochs + epoch + 1)
        # train_metrics[f"train_loss_client_{config['client_ID']}_model_{config['selected_model_index']}_epoch_{(config['current_round'] - 1) * config['local_epochs'] + epoch + 1}"] = epoch_loss
    round_loss = epoch_loss

    # if writer is not None:
    #     writer.add_scalar('train/train loss - rounds', round_loss, current_round+1)
    # train_metrics[f"train_loss_client_{config['client_ID']}_model_{config['selected_model_index']}_round_{config['current_round']}"] = round_loss

    if config['lr_schedule_enabled'] is True:
        config['latest_scheduler_state_dict'][client_index] = scheduler.state_dict()
        config['latest_learning_rate'][client_index] = scheduler.get_last_lr()[0]

    # return train_metrics
    logger.debug(f"For client {client_index}, after training model {model_index}: loss = {round_loss}.")
    logger.debug(f"For client {client_index}, after training model {model_index}: accuracy = {accuracy}.")
    return round_loss


def test(
    net: torch.nn.Module,
    testloader: DataLoader,
    config: Union[ConfigDict, Dict[str, Union[bool, bytes, float, int, str]]],
    client_index=None,
    model_index=None
) -> float:
    """Test the neural network on the test set."""

    criterion = config['criterion'][config['model_type']]
    loss = 0.0
    accuracy = 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            if config['dataset'] in ['AmazonReview', 'AG_news']:       # text-based datasets
                pass
            else:   # image-based datasets
                inputs = inputs.to(torch.float32)
                labels = labels.to(torch.float32)
            if config['dataset'] == 'CIFAR10' and config['model_type'] == 'AE':
                inputs = inputs / 255.0  # Normalize to [0, 1]
                inputs = inputs.view(inputs.size(0), -1)    # Flatten inputs to feed to autoencoder
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
            # if config['model_type'] == 'CNN':
            #     inputs = inputs.permute(0, 3, 1, 2)
            outputs = net(inputs)
            if config['model_type'] == 'AE':    # Unsupervised case: loss computed based on the inputs
                loss += sum([criterion(inputs[i], outputs[i]).tolist() for i in range(len(inputs))])
            elif config['model_type'] in ['CNN', 'Cifar10CNN', 'GenConvNet', 'AmazonMLP', 'TextCNN']:      # Supervised case: loss computed based on the labels
                loss += sum([criterion(outputs[i], labels[i].long()).tolist() for i in range(len(labels))])
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1).to(torch.float32)
                # logger.info(f"PREDICTED LABELS: {predicted_labels}")
                # logger.info(f"GROUND TRUTH LABELS: {labels}")
                accuracy += (predicted_labels == labels).sum().item()
            elif config['model_type'] == 'linear_regression':   # Supervised case: loss computed based on the labels
                outputs = outputs.squeeze()
                loss += sum([criterion(labels[i], outputs[i]).tolist() for i in range(len(labels))])
            else:
                raise ValueError(f"Invalid value given for config_dict['model_type']: {config['model_type']}")
    loss /= len(testloader.dataset)
    accuracy /= len(testloader.dataset)
    if client_index is not None and model_index is not None:
        logger.debug(f"For client {client_index} and model {model_index}, test loss = {loss}")
    if config['model_type'] in ['CNN', 'Cifar10CNN', 'GenConvNet', 'AmazonMLP', 'TextCNN']:
        logger.info(f"For client {client_index} and model {model_index}, test accuracy = {accuracy}")
    return loss


def compute_gradient(
    net: torch.nn.Module,
    trainloader: DataLoader,
    config: Union[ConfigDict, Dict[str, Union[bool, bytes, float, int, str]]],
    client_index,
    model_index
) -> List[np.ndarray]:
    """Get average gradients of the neural network evaluated on the data (without doing any training or updating the net's weights)."""

    # Ignore batch size and repackage all data into a single-batch trainloader, so that the returned gradient is the one after all data has been through the net at once.
    trainloader = DataLoader(dataset=trainloader.dataset, batch_size=len(trainloader.dataset), shuffle=False)

    criterion = config['criterion'][config['model_type']]
    net.eval()
    net.zero_grad()
    test_loss = 0.0
    accuracy = 0.0
    for inputs, labels in trainloader:
        if config['dataset'] in ['AmazonReview', 'AG_news']:  # text-based datasets
            pass
        else:  # image-based datasets
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
        if config['dataset'] == 'CIFAR10' and config['model_type'] == 'AE':
            inputs = inputs / 255.0  # Normalize to [0, 1]
            inputs = inputs.view(inputs.size(0), -1)    # Flatten inputs to feed to autoencoder
        inputs.requires_grad = True     # Enable storing of gradients
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])
        # net.zero_grad()       # Not zeroing gradients at each round on purpose
        outputs = net(inputs)
        if config['model_type'] == 'AE':  # Unsupervised case: loss computed based on the inputs
            loss = criterion(outputs, inputs)
            loss.backward()
            test_loss += sum([criterion(inputs[i], outputs[i]).tolist() for i in range(len(inputs))])
        elif config['model_type'] in ['CNN', 'Cifar10CNN', 'GenConvNet', 'AmazonMLP', 'TextCNN']:  # Supervised case: loss computed based on the labels
            loss = criterion(outputs, labels.long())
            loss.backward()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).to(torch.float32)
            test_loss += sum([criterion(outputs[i], labels[i].long()).tolist() for i in range(len(labels))])
            accuracy += (predicted_labels == labels).sum().item()
        elif config['model_type'] == 'linear_regression':   # Supervised case: loss computed based on the labels
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            test_loss += sum([criterion(labels[i], outputs[i]).tolist() for i in range(len(inputs))])
        else:
            raise ValueError(f"Invalid value given for config_dict['model_type']: {config['model_type']}")
    test_loss /= len(trainloader.dataset)
    logger.debug(f"Gradient: For client {client_index} and model {model_index}, test loss = {test_loss}")
    logger.debug(f"Gradient: For client {client_index} and model {model_index}, test accuracy = {accuracy}")
    gradient = get_single_model_gradients(net)
    gradient = [layer_gradient/len(trainloader) for layer_gradient in gradient]        # Divide by the number of minibatches in the trainloader
    return gradient


def get_custom_optimizer(
    params: Iterator[torch.nn.Parameter],
    config: Union[ConfigDict, Dict[str, Union[bool, bytes, float, int, str]]],
    **kwargs
) -> torch.optim.Optimizer:
    """Returns a custom optimizer depending on the optimizer type."""

    optimizer_type = config['optimizer_type']
    lr = config['local_learning_rate']

    if optimizer_type == "SGD":
        return torch.optim.SGD(params=params, lr=lr, **kwargs)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(params=params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")


def set_up_LR_scheduler(config_dict):
    # Set up learning rate scheduler
    if 'latest_learning_rate' not in config_dict.keys():
        config_dict.latest_learning_rate = [config_dict['local_learning_rate']] * config_dict['number_of_clients']
    if 'latest_scheduler_state_dict' not in config_dict.keys():
        config_dict['latest_scheduler_state_dict'] = [None] * config_dict['number_of_clients']
    return config_dict
