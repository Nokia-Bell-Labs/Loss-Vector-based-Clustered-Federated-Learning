# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.linear import Linear


class Autoencoder(nn.Module):
    def __init__(self, input_size, middle_layer_size, **kwargs):
        super(Autoencoder, self).__init__()
        output_activation_function = kwargs["output_activation_function"] if 'output_activation_function' in kwargs else nn.Sigmoid()       # default to Sigmoid unless defined otherwise

        if input_size < middle_layer_size * 4:
            raise ValueError(f"Input size must be strictly greater than middle layer size * 4.\nCurrent input size = {input_size}, current middle layer size = {middle_layer_size}.")

        self.encoder = nn.Sequential(
            nn.Linear(input_size, middle_layer_size * 4),
            nn.ReLU(),
            nn.Linear(middle_layer_size * 4, middle_layer_size * 2),
            nn.ReLU(),
            nn.Linear(middle_layer_size * 2, middle_layer_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(middle_layer_size, middle_layer_size * 2),
            nn.ReLU(),
            nn.Linear(middle_layer_size * 2, middle_layer_size * 4),
            nn.ReLU(),
            nn.Linear(middle_layer_size * 4, input_size),
            output_activation_function,         # torch.nn.Tanh() if config.dataset == 'SYNTHETIC' else torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CIFAR10Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (16x16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (8x8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (4x4)
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.latent_fc = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8x8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # (16x16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # (32x32)
            nn.Tanh()  # Output in [-1, 1] to match input normalization
        )

    def forward(self, x):
        # Input is flattened: [B, 3072] → reshape to [B, 3, 32, 32]
        x = x.view(-1, 3, 32, 32)

        # Encoder
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.latent_fc(x)

        # Decoder
        x = self.decoder_fc(z)
        x = self.decoder(x)

        # Flatten output to match input shape: [B, 3, 32, 32] → [B, 3072]
        return x.view(x.size(0), -1)
    

class Cifar10CNN(nn.Module):
    def __init__(self, number_of_clusters, **kwargs):
        super(Cifar10CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32x64 -> 16x16x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16x16x64 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16x128 -> 8x8x128

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 8x8x128 -> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8x256 -> 4x4x256

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 4x4x256 -> 4x4x512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 4x4x512 -> 2x2x512
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*2*512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, number_of_clusters)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class TinyImageNetCNN(nn.Module):
    def __init__(self, number_of_clusters, **kwargs):
        super(TinyImageNetCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, number_of_clusters)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Modified from: https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/src/model/AlexNet.py
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(x)
        return x


class GenConvNet(torch.nn.Module):
    def __init__(self, num_classes=10):  # num_classes will default to 10 (for MNIST)
        super(GenConvNet, self).__init__()
        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        # Fully connected layer (output size depends on the number of classes)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, num_classes)  # num_classes can be 10, 62, or any other value
        # Max pooling layer
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Works for MNIST, FEMNIST, EMNIST
        # Apply conv1 -> ReLU -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply conv2 -> ReLU -> pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for fully connected layer
        x = x.view(-1, 16 * 4 * 4)
        # Fully connected layer
        x = self.fc1(x)
        return x


class AmazonMLP(nn.Module):     # taken from https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            # nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU()
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out


class TextCNN(nn.Module):   # taken from https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/trainmodel/models.py
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3, 4, 5], max_len=200, dropout=0.8, padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out


def get_model(config_dict):
    """Returns a single model, depending on the model_type"""

    if config_dict['model_type'] == 'AE' and config_dict['dataset'] != 'CIFAR10':
        net = Autoencoder(
            input_size=config_dict['autoencoder_input_size'],
            middle_layer_size=config_dict['autoencoder_middle_layer_size'],
            output_activation_function=config_dict['autoencoder_output_activation_function'],
        ).to(config_dict['device'])
    elif config_dict['model_type'] == 'AE' and config_dict['dataset'] == 'CIFAR10':
        net = CIFAR10Autoencoder().to(config_dict['device'])
    elif config_dict['dataset'] == 'CIFAR10' and config_dict['model_type'] == 'CNN':
        net = Cifar10CNN(number_of_clusters=config_dict['number_of_data_classes']).to(config_dict['device'])
    elif config_dict['dataset'] == 'MNIST' and config_dict['model_type'] == 'CNN':
        net = GenConvNet(num_classes=config_dict['number_of_data_classes']).to(config_dict['device'])
    elif config_dict['dataset'] == 'FMNIST' and config_dict['model_type'] == 'CNN':
        net = GenConvNet(num_classes=config_dict['number_of_data_classes']).to(config_dict['device'])
    elif config_dict['dataset'] == 'FEMNIST' and config_dict['model_type'] == 'CNN':
        net = GenConvNet(num_classes=config_dict['FEMNIST_num_data_classes']).to(config_dict['device'])
    elif config_dict['dataset'] == 'AmazonReview' and config_dict['model_type'] == 'AmazonMLP':
        net = AmazonMLP().to(config_dict['device'])
    elif config_dict['dataset'] == 'AG_news' and config_dict['model_type'] == 'TextCNN':
        net = TextCNN(hidden_dim=100, vocab_size=32000, num_classes=config_dict['number_of_clusters']).to(config_dict['device'])     # Vocab size is 32000 for AG_News
    elif config_dict['dataset'] == 'TinyImageNet' and config_dict['model_type'] == 'CNN':
        net = TinyImageNetCNN(number_of_clusters=config_dict['number_of_data_classes']).to(config_dict['device'])
        # net = AlexNet(num_classes=config_dict['number_of_data_classes']).to(config_dict['device'])
    elif config_dict['model_type'] == 'linear_regression':
        if 'cur_linear_class_obj' not in config_dict:
            linear_class = Linear(config_dict)
            # Storing this object so it can be used by downstream functions
            config_dict['cur_linear_class_obj'] = linear_class
        # linear_singleton_class = Linear_singleton()
        # linear_singleton_class.set_model(linear_class)
        linear_class = config_dict['cur_linear_class_obj']
        net = linear_class.get_scratch_model().to(config_dict['device'])
    elif config_dict['dataset'] == 'AG_news' and config_dict['model_type'] == 'TextCNN':
        net = TextCNN(hidden_dim=100, vocab_size=32000, num_classes=config_dict['number_of_clusters']).to(config_dict['device'])     # Vocab size is 32000 for AG_News
    else:
        raise ValueError(f"Invalid dataset-model combination given: dataset: {config_dict['dataset']}, model: {config_dict['model_type']}")

    return net


def get_models(config_dict):
    """Returns a list of models, depending on the model_type"""

    if config_dict['model_type'] == 'AE' and config_dict['dataset'] != 'CIFAR10':
        # Create as many models as the actual number of models for this run
        nets = [
            Autoencoder(
                input_size=config_dict['autoencoder_input_size'],
                middle_layer_size=config_dict['autoencoder_middle_layer_size'],
                output_activation_function=config_dict['autoencoder_output_activation_function'],
            ).to(config_dict['device'])
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['model_type'] == 'AE' and config_dict['dataset'] == 'CIFAR10':
        nets = [
            CIFAR10Autoencoder().to(config_dict['device'])
            for _ in range(config_dict['number_of_models'])
        ]        
    elif config_dict['dataset'] == 'CIFAR10' and config_dict['model_type'] == 'CNN':
        nets = [
            Cifar10CNN(number_of_clusters=config_dict['number_of_data_classes']).to(config_dict['device'])
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['dataset'] == 'MNIST' and config_dict['model_type'] == 'CNN':
        nets = [
            GenConvNet(num_classes=config_dict['number_of_data_classes']).to(config_dict['device'])
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['dataset'] == 'FMNIST' and config_dict['model_type'] == 'CNN':
        nets = [
            GenConvNet(num_classes=config_dict['number_of_data_classes']).to(config_dict['device'])
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['dataset'] == 'FEMNIST' and config_dict['model_type'] == 'CNN':
        nets = [
            GenConvNet(num_classes=config_dict['FEMNIST_num_data_classes']).to(config_dict['device'])
            # for _ in range(config_dict['FEMNIST_number_of_models'])
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['dataset'] == 'AmazonReview' and config_dict['model_type'] == 'AmazonMLP':
        nets = [
            AmazonMLP().to(config_dict['device'])
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['dataset'] == 'AG_news' and config_dict['model_type'] == 'TextCNN':
        nets = [
            TextCNN(hidden_dim=100, vocab_size=32000, num_classes=config_dict['number_of_clusters']).to(config_dict['device'])     # Vocab size is 32000 for AG_News
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['dataset'] == 'TinyImageNet' and config_dict['model_type'] == 'CNN':
        nets = [
            TinyImageNetCNN(number_of_clusters=config_dict['number_of_data_classes']).to(config_dict['device'])
            # AlexNet(num_classes=config_dict['number_of_data_classes']).to(config_dict['device'])
            for _ in range(config_dict['number_of_models'])
        ]
    elif config_dict['model_type'] == 'linear_regression':
        if 'cur_linear_class_obj' not in config_dict:
            linear_class = Linear(config_dict)
            # Storing this object so it can be used by downstream functions
            config_dict['cur_linear_class_obj'] = linear_class
        # linear_singleton_class = Linear_singleton()
        # linear_singleton_class.set_model(linear_class)

        linear_class = config_dict['cur_linear_class_obj']
        nets = [net.to(config_dict['device']) for net in linear_class.get_initial_models()]
    else:
        raise ValueError(f"Invalid dataset-model combination given: dataset: {config_dict['dataset']}, model: {config_dict['model_type']}")

    return nets
