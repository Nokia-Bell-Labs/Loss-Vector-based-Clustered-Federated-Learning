# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import re
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

from config import Status
from model import Autoencoder
from utils.learning_utils import set_single_model_parameters


def process_metrics_dicts(metrics_dicts, config_dict, metric_name):

    # values_per_client_model_round = np.zeros((config_dict['number_of_clients'], config_dict['number_of_models'], config_dict['number_of_rounds']))
    values_per_client_model_round = np.zeros((config_dict['number_of_clients'], config_dict['number_of_models']))

    for metrics_dict in metrics_dicts:  # one dict per client
        for key, value in metrics_dict.items():  # for each key-value pair
            if bool(re.search(metric_name, key)):   # filter only the key-value pairs that pertain to the metric `metric_name`
                # Pattern: 'test_client_{client_ID}_model_{model_index}_round_{current_round}_{metric_name}'
                numbers = re.findall('\\d+', key)
                client_ID = int(numbers[0])
                model_index = int(numbers[1])
                # current_round_index = int(numbers[2]) - 1     # subtracting 1 because of the round definition starting from 1 and because we now want to use current_round as an index
                value_to_log = value
                # print(f'clientID={client_ID}, model_index={model_index}, current_round={current_round}')

                # print(values_per_client_model_round)
                # values_per_client_model_round[client_ID][model_index][current_round_index] = value_to_log
                values_per_client_model_round[client_ID][model_index] = value_to_log

    return values_per_client_model_round


def plot_input_vs_output(net, input_images, title=None):
    images = input_images.view(input_images.size(0), -1)
    outputs = net(images)

    fig, ax = plt.subplots()

    images_to_plot = images.view(-1, 28, 28).permute(1,0,2).reshape(28, -1)
    outputs_to_plot = outputs.view(-1, 28, 28).permute(1,0,2).reshape(28, -1)
    images_to_plot = images_to_plot.detach().numpy()
    outputs_to_plot = outputs_to_plot.detach().numpy()

    combined = np.vstack([images_to_plot, outputs_to_plot])
    ax.imshow(combined, cmap="gray")
    ax.set_axis_off()
    plt.title(title)
    # plt.show()

    return fig


def plot_loss_vs_label(losses, labels, all_labels, loss_axis_limit=np.inf):
    # Convert lists to numpy arrays
    losses = np.array(losses)
    labels = np.array(labels)

    # Plot MSE reconstruction loss versus label
    fig, ax = plt.subplots()
    ax.scatter(labels, losses, s=5)
    ax.set_xlabel('Label')
    ax.set_ylabel('MSE Reconstruction Loss')
    plt.title('MNIST Autoencoder Reconstruction Loss vs Label')
    if loss_axis_limit < np.inf:
        ax.set_ylim([0, loss_axis_limit])
    ax.set_xticks(all_labels)
    # plt.show()

    return fig


def plot_loss_histograms_per_label(losses, labels, all_labels, loss_axis_limit=np.inf):

    n_bins = 100

    # Convert lists to numpy arrays
    losses = np.array(losses)
    labels = np.array(labels)

    # Plot loss histograms for each label
    fig, ax = plt.subplots(nrows=len(all_labels), ncols=1, sharex=True, squeeze=False)

    for digit_index, digit in enumerate(all_labels):
        indices_for_label = np.where(labels == digit)[0]
        losses_for_label = losses[indices_for_label]

        ax[digit_index][0].hist(losses_for_label, bins=n_bins, label=f'Ground truth label {digit}')
        # ax[digit_index].set_title(f'Digit {digit}')
        ax[digit_index][0].xaxis.set_tick_params(labelbottom=True)
        if loss_axis_limit < np.inf:
            ax[digit_index][0].set_xlim([0, loss_axis_limit])
        ax[digit_index][0].legend(prop={'size': 10})
        # plt.show()

    fig.supxlabel('MSE Reconstruction Loss')
    fig.supylabel('Count')
    fig.suptitle('MNIST Autoencoder Reconstruction Loss vs Label')
    # plt.show()

    return fig


def plot_roc_curve(ground_truth_labels_binary_list, per_image_loss_list):
    fig, ax = plt.subplots()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true=ground_truth_labels_binary_list,
        y_score=per_image_loss_list,
        pos_label=Status.ABNORMAL,
        drop_intermediate=False
    )
    roc_auc = roc_auc_score(
        y_true=ground_truth_labels_binary_list,
        y_score=per_image_loss_list
    )
    ax.set_xlabel('False positive rate (FP/FP+TN)')
    ax.set_ylabel('True positive rate (TP/TP+FN)')
    plt.title('ROC curve')
    plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = %0.4f' % roc_auc)
    plt.legend(loc='best')
    return fig


def plot_roc_curve_for_all_clients(ground_truth_labels_binary_list_per_client, per_image_loss_list_per_client):
    fig, ax = plt.subplots()

    num_clients = len(ground_truth_labels_binary_list_per_client)
    for client_ID in range(num_clients):
        ground_truth_labels_binary_list = ground_truth_labels_binary_list_per_client[client_ID]
        per_image_loss_list = per_image_loss_list_per_client[client_ID]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_true=ground_truth_labels_binary_list,
            y_score=per_image_loss_list,
            pos_label=Status.ABNORMAL,
            drop_intermediate=False
        )
        roc_auc = roc_auc_score(
            y_true=ground_truth_labels_binary_list,
            y_score=per_image_loss_list
        )
        ax.plot(false_positive_rate, true_positive_rate, label=f'Client {client_ID}, AUC = %0.4f' % roc_auc)

    ax.set_xlabel('False positive rate (FP/FP+TN)')
    ax.set_ylabel('True positive rate (TP/TP+FN)')
    plt.title('ROC curve')
    plt.legend(loc='best')
    return fig


def loss_matrix_to_df(loss_matrix):
    losses_df = pd.DataFrame(loss_matrix)
    losses_df.index = ["Loss of client " + str(model_index) for model_index in range(loss_matrix.shape[0])]
    losses_df.columns = ["On model " + str(client_index) for client_index in range(loss_matrix.shape[1])]
    return losses_df


# Visualization as a heatmap
def plot_heatmap(loss_matrix):
    heatmap = sns.heatmap(loss_matrix, annot=True, fmt=".4f", cmap='coolwarm', cbar_kws={'label': 'Reconstruction Loss'})
    fig = heatmap.figure
    plt.title('Heatmap of AE Reconstruction Losses')
    plt.xlabel('Model Index')
    plt.ylabel('Client Index')
    plt.close(fig)
    return fig


def plot_MNIST_digit_reconstruction(config_dict, testloaders, all_model_parameters):
    data_point_dimension = testloaders[0].dataset[0][0].shape[0]
    config_dict['autoencoder_input_size'] = data_point_dimension
    net = Autoencoder(
        input_size=config_dict['autoencoder_input_size'],
        middle_layer_size=config_dict['autoencoder_middle_layer_size'],
        output_activation_function=config_dict['autoencoder_output_activation_function'],
    ).to(config_dict[
             'device'])  # Only one model object is needed, as we feed the appropriate parameters into it each time and save them after each step. So the model object carries no history that we use.

    input_images = torch.stack([testloaders[client_index].dataset[0][0] for client_index in range(config_dict['number_of_clients'])])
    images = input_images.view(input_images.size(0), -1)

    reconstruction_figs = []
    actual_number_of_models = len(all_model_parameters.keys())
    for model_index in range(actual_number_of_models):
        fig, ax = plt.subplots()
        fig.suptitle(f"Model {model_index} evaluation")
        set_single_model_parameters(net, all_model_parameters[
            model_index])  # We set the parameters, and train() internally zeros out the gradients, so what was inside the net from previous evaluation/training steps does not matter.
        outputs = net(images)
        images_to_plot = images.view(-1, 28, 28).permute(1, 0, 2).reshape(28, -1)
        outputs_to_plot = outputs.view(-1, 28, 28).permute(1, 0, 2).reshape(28, -1)
        images_to_plot = images_to_plot.detach().numpy()
        outputs_to_plot = outputs_to_plot.detach().numpy()

        combined = np.vstack([images_to_plot, outputs_to_plot])
        ax.imshow(combined, cmap="gray")
        ax.set_axis_off()
        plt.close(fig)
        reconstruction_figs.append(fig)

    return reconstruction_figs


def plot_metrics_over_rounds(metrics_per_round):
    metrics_over_rounds_figs = []
    modes = list(metrics_per_round.keys())    # ['train', 'test']
    metric_names = list(metrics_per_round[modes[-1]].keys())
    number_of_rounds = len(metrics_per_round[modes[-1]][metric_names[-1]])

    for mode in modes:
        for metric_name in metric_names:
            fig, ax = plt.subplots()
            ax.plot(list(range(number_of_rounds)), metrics_per_round[mode][metric_name], marker='x')
            ax.set_xlabel('Rounds')
            ax.set_ylabel(metric_name)
            ax.set_xticks(range(number_of_rounds))
            plt.title(metric_name + '_over_time')
            plt.close(fig)
            metrics_over_rounds_figs.append(fig)

    return metrics_over_rounds_figs
