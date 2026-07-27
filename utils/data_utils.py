# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import json
import os
import random
import re
import shutil
from collections import defaultdict
import zipfile
import math
import numpy as np
import torch
from PIL import Image
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
from datasets import load_dataset
from collections import Counter
import gdown

from utils.synthetic_data_generation_utils import create_synthetic_datasets
from utils.linear import LinearDataGenerator, generate_thetas
from utils.dataset_modifier import split_dataset, convert_from_dict_to_list, split_dict_values, get_dominance_data_split
from utils.dataset_modifier import dict_overlap_swap_exps, get_label_skew_data_split, get_feature_skew_data_split
from utils.logging_utils import logging
logger = logging.getLogger(__name__)


def invert_dict(old_dict):
    """Inverts a dictionary by returning another dictionary with the original values as keys and lists of the original keys as values."""
    new_dict = {}
    for key, value in old_dict.items():
        new_dict.setdefault(value, []).append(key)
    return new_dict


def get_seeded_rng(config_dict):
    return np.random.default_rng(config_dict['seed'])


def unique_in_order(values):
    return list(dict.fromkeys(values))


# ------------------------- start MNIST -------------------------

def filter_MNIST_trainset(full_MNIST_trainset, digits_to_keep):
    per_digit_trainsets = [[mnist_val for mnist_val in full_MNIST_trainset if mnist_val[1] == normal_digit] for normal_digit in digits_to_keep]
    return per_digit_trainsets


def filter_MNIST_testset(full_MNIST_testset, digits_to_keep):
    testset = sum([[mnist_val for mnist_val in full_MNIST_testset if mnist_val[1] == normal_digit] for normal_digit in digits_to_keep], [])
    return testset


def load_MNIST_data(config_dict, angle=0):
    # Download the MNIST Dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle)),
        transforms.ToTensor(),       # Transforms images to a PyTorch Tensor
    ])
    full_MNIST_trainset = MNIST(root=config_dict['data_dir'][config_dict['dataset']], train=True, download=True, transform=transform)
    full_MNIST_testset = MNIST(root=config_dict['data_dir'][config_dict['dataset']], train=False, download=True, transform=transform)

    all_data_classes = list(config_dict['MNIST_all_digits'])
    digits_to_keep = unique_in_order([int(str(data_class).split("r")[0]) for data_class in all_data_classes])  # Only keep each digit once, while preserving the configured order.

    # Filter and keep only the specified digits
    train_data_raw, test_data_raw = {}, {}
    for digit in digits_to_keep:
        train_data_raw[digit] = [(mnist_val[0].view(-1), mnist_val[1]) for mnist_val in full_MNIST_trainset if mnist_val[1] == digit]
        test_data_raw[digit] = [(mnist_val[0].view(-1), mnist_val[1]) for mnist_val in full_MNIST_testset if mnist_val[1] == digit]

    return train_data_raw, test_data_raw


def rotate_MNIST_data_if_needed(train_data_raw, test_data_raw, config_dict):
    all_data_classes = list(config_dict['MNIST_all_digits'])
    all_data_classes = list(map(str, all_data_classes))   # Convert digits to string (if they are not already)
    train_data, test_data = {}, {}
    for data_class in all_data_classes:
        parts = data_class.split("r")
        if len(parts) > 2:
            raise ValueError("Invalid value contained in config_dict['MNIST_all_digits']")
        digit = int(parts[0])
        rotations = int(parts[1]) if len(parts) == 2 else 0

        train_data[data_class] = rotate_dataset(dataset=train_data_raw[digit], rotations=rotations)
        test_data[data_class] = rotate_dataset(dataset=test_data_raw[digit], rotations=rotations)

    return train_data, test_data


def rotate_dataset(dataset, rotations):
    rotated_dataset = [[] for _ in range(len(dataset))]

    for i in range(len(dataset)):
        rotated_dataset[i] = (
            torch.rot90(dataset[i][0].view(28, 28), k=rotations, dims=[0, 1]).contiguous().view(-1),
            dataset[i][1]
        )

    return rotated_dataset


# ------------------------- end MNIST -------------------------

# ------------------------- start FMNIST (FashionMNIST) -------------------------

def load_FMNIST_data(config_dict, angle=0):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle)),
        transforms.ToTensor(),  # Transforms images to a PyTorch Tensor
    ])

    # Load the full FMNIST dataset
    full_FMNIST_trainset = FashionMNIST(
        root=config_dict['data_dir'][config_dict['dataset']],
        train=True,
        download=True,
        transform=transform
    )

    full_FMNIST_testset = FashionMNIST(
        root=config_dict['data_dir'][config_dict['dataset']],
        train=False,
        download=True,
        transform=transform
    )

    # Get the filtered class labels
    all_data_classes = list(config_dict['MNIST_all_digits'])  # e.g., [0, 2, 4, 6]
    class_to_index = {orig_label: new_idx for new_idx, orig_label in enumerate(all_data_classes)}

    train_data_raw, test_data_raw = {}, {}

    # Filter and reindex labels
    for orig_label in all_data_classes:
        new_label = class_to_index[orig_label]
        train_data_raw[new_label] = [
            (x[0].view(-1), new_label)
            for x in full_FMNIST_trainset
            if x[1] == orig_label
        ]
        test_data_raw[new_label] = [
            (x[0].view(-1), new_label)
            for x in full_FMNIST_testset
            if x[1] == orig_label
        ]

    return train_data_raw, test_data_raw


# ------------------------- end FMNIST (FashionMNIST) -------------------------

# ------------------------- start CIFAR -------------------------

def load_CIFAR10_data(config_dict, angle=0):
    # Download and transform CIFAR-10 (train and test)

    # Define transforms for data augmentation
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle)),  # Rotate by angle
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle)),  # Rotate by angle
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = CIFAR10(root=config_dict['data_dir'][config_dict['dataset']], train=True, download=True, transform=transform_train)
    testset = CIFAR10(root=config_dict['data_dir'][config_dict['dataset']], train=False, download=True, transform=transform_test)

    selected_data_classes = random.sample(population=trainset.classes, k=config_dict['number_of_data_classes'])
    selected_data_class_targets = [trainset.class_to_idx[data_class] for data_class in selected_data_classes]

    # Filter and keep only the specified data classes
    train_data, test_data = {}, {}
    target_to_data_class_index_mapping = {}
    for data_class_index in range(len(selected_data_classes)):
        data_class_name = selected_data_classes[data_class_index]
        data_class_target = selected_data_class_targets[data_class_index]
        target_to_data_class_index_mapping[data_class_target] = data_class_index
        train_data[data_class_target] = [(np.transpose(trainset.data[i], axes=(2, 0, 1)), data_class_index) for i in range(len(trainset.targets)) if trainset.targets[i] == data_class_target]
        test_data[data_class_target] = [(np.transpose(testset.data[i], axes=(2, 0, 1)), data_class_index) for i in range(len(testset.targets)) if testset.targets[i] == data_class_target]

    return train_data, test_data


# ------------------------- end CIFAR -------------------------

# ------------------------- start TinyImageNet -------------------------

def load_TinyImageNet(config_dict):
    # We use the validation part as both validation and test in our code, as the test part of Tiny-ImageNet has no labels.

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_path = config_dict['data_dir'][config_dict['dataset']]
    trainset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=train_transform)
    # Preprocess validation data by moving images into class-specific subdirectories based on val_annotations.txt, so that validation is in the same format as training.
    preprocess_TinyImageNet_val(dataset_path)
    testset = ImageFolder(root=os.path.join(dataset_path, "val"), transform=test_transform)

    selected_data_classes = random.sample(population=trainset.classes, k=config_dict['number_of_data_classes'])
    selected_data_class_targets = [trainset.class_to_idx[data_class] for data_class in selected_data_classes]

    # Filter and keep only the specified data classes
    train_data, test_data = {}, {}
    for data_class_index in range(len(selected_data_classes)):
        data_class_target = selected_data_class_targets[data_class_index]
        train_data[data_class_target] = []
        test_data[data_class_target] = []

        for i in range(len(trainset.targets)):
            if trainset.targets[i] == data_class_target:
                image_path = trainset.imgs[i][0]
                if os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        img = img.convert('RGB')
                        train_data[data_class_target].append([np.transpose(img, axes=(2, 0, 1)), data_class_index])
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")

        for i in range(len(testset.targets)):
            if testset.targets[i] == data_class_target:
                image_path = testset.imgs[i][0]
                if os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        img = img.convert('RGB')
                        test_data[data_class_target].append([np.transpose(img, axes=(2, 0, 1)), data_class_index])
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")

    return train_data, test_data


def preprocess_TinyImageNet_val(data_dir):
    """
    Restructures the TinyImageNet validation directory to be compatible with torchvision.datasets.ImageFolder.
    """
    val_path = os.path.join(data_dir, 'val')
    val_images_path = os.path.join(val_path, 'images')
    val_annotations_path = os.path.join(val_path, 'val_annotations.txt')

    if not os.path.exists(val_images_path) or not os.path.exists(val_annotations_path):
        print(f"Validation directory or annotations not found at {val_path}. Skipping preprocessing.")
        return

    print(f"Preprocessing TinyImageNet validation set in {val_path}...")

    with open(val_annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_id = parts[1]

            # Create class directory if it doesn't exist
            class_dir = os.path.join(val_path, class_id)
            os.makedirs(class_dir, exist_ok=True)

            # Move image to its class directory
            src_path = os.path.join(val_images_path, img_name)
            dst_path = os.path.join(class_dir, img_name)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
            else:
                print(f"Warning: Image {img_name} not found at {src_path}")

    # Remove the original 'images' directory and 'val_annotations.txt' if empty
    if not os.listdir(val_images_path):
        os.rmdir(val_images_path)
    os.remove(val_annotations_path)
    print("Validation set preprocessing complete.")


# ------------------------- end TinyImageNet -------------------------

# ------------------------- start FEMNIST -------------------------

def read_dir(data_dir):
    # Taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py

    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def select_top_writers(data, N, top_k_factor=10):
    """
    Select N writers randomly from the top K% with most data points,
    where K is increased until at least top_k_factor * N writers are available.
    
    Args:
        data: dict of writer_id -> {'x': [...], 'y': [...]}
        N: number of writers to select
        top_k_factor: multiplier to ensure pool size is at least top_k_factor * N
    
    Returns:
        selected_writers: list of writer IDs
    """
    writer_sample_counts = {writer: len(info['y']) for writer, info in data.items()}
    sorted_writers = sorted(writer_sample_counts, key=writer_sample_counts.get, reverse=True)
    total_writers = len(sorted_writers)

    # Try increasing top_k until the pool is large enough
    for k_percent in range(10, 110, 10):  # 10%, 20%, ..., 100%
        top_k_count = math.ceil((k_percent / 100) * total_writers)
        top_writers = sorted_writers[:top_k_count]
        
        if len(top_writers) >= top_k_factor * N:
            break  # Sufficient pool size

    # Randomly select N writers from the top K%
    selected_writers = random.sample(top_writers, N)
    return selected_writers


def load_FEMNIST_data(config_dict):
    train_data_percentage = config_dict['train_data_percentage']

    all_writers, _, data = read_dir(config_dict['data_dir'][config_dict['dataset']])

    train_data = {}
    test_data = {}

    if config_dict['FEMNIST_data_mode'] == 'by_writer':
        number_of_writers = config_dict['number_of_data_classes']

        selected_writers = select_top_writers(data, number_of_writers)

        for i, writer in enumerate(selected_writers):
            number_of_data_points_of_writer = len(data[writer]['y'])
            train_data_length = int(train_data_percentage * number_of_data_points_of_writer)
            test_data_length = number_of_data_points_of_writer - train_data_length

            x_all = torch.tensor(data[writer]['x'], dtype=torch.float32)  # shape: [N, 784]
            y_all = data[writer]['y']

            # Flatten each image explicitly (though they may already be flat in FEMNIST)
            train_data[writer] = [
                (x.view(-1), y)
                for x, y in zip(x_all[:train_data_length], y_all[:train_data_length])
            ]

            test_data[writer] = [
                (x.view(-1), y)
                for x, y in zip(x_all[train_data_length:], y_all[train_data_length:])
            ]

    elif config_dict['FEMNIST_data_mode'] == 'by_character':
        number_of_characters = config_dict['number_of_data_classes']

        selected_characters = random.sample(population=list(range(62)), k=number_of_characters)

        images = {character: [] for character in selected_characters}
        # labels = {character: [] for character in selected_characters}
        for writer_data in data.values():
            for data_point_index in range(len(writer_data['y'])):
                image = writer_data['x'][data_point_index]
                character = writer_data['y'][data_point_index]
                if character in selected_characters:
                    # all_data[character].append((torch.Tensor(image), character))
                    images[character].append(torch.Tensor(image))
                    # labels[character].append()

        for character in selected_characters:
            number_of_data_points_for_character = len(images[character])
            train_data_length = int(train_data_percentage * number_of_data_points_for_character)
            character_data_points = list(
                zip(
                    images[character],
                    [character] * number_of_data_points_for_character
                )
            )
            random.shuffle(character_data_points)
            train_data[character] = character_data_points[:train_data_length]
            test_data[character] = character_data_points[train_data_length:]

    return train_data, test_data

# ------------------------- end FEMNIST -------------------------

# ------------------------- start synthetic -------------------------

def load_synthetic_data(config_dict):

    np.random.seed(config_dict['seed'])

    cluster_datasets = create_synthetic_datasets(
        number_of_datasets=config_dict['number_of_data_classes'],
        data_point_dimension=config_dict['SYNTHETIC_data_point_dimension'],
        number_of_overlap_basis_vectors=config_dict['SYNTHETIC_number_of_overlap_basis_vectors'],
        number_of_no_overlap_basis_vectors=config_dict['SYNTHETIC_number_of_no_overlap_basis_vectors'],
        overlap_dataset_size=config_dict['SYNTHETIC_overlap_dataset_size'],
        number_of_overlap_points_per_dataset=config_dict['SYNTHETIC_number_of_overlap_points_per_dataset'],
        number_of_no_overlap_points_per_dataset=config_dict['SYNTHETIC_number_of_no_overlap_points_per_dataset'],
    )
    # cluster_labels = list(range(config_dict['number_of_data_classes']))
    cluster_labels = [[cluster_index] * len(cluster_datasets[cluster_index]) for cluster_index in range(config_dict['number_of_data_classes'])]

    train_data, test_data = {}, {}
    for cluster_index in range(config_dict['number_of_data_classes']):
        number_of_data_points_for_cluster = len(cluster_datasets[cluster_index])
        train_data_length = int(config_dict['train_data_percentage'] * number_of_data_points_for_cluster)
        cluster_data_points = list(zip(torch.Tensor(cluster_datasets[cluster_index]), cluster_labels[cluster_index]))
        random.shuffle(cluster_data_points)
        train_data[cluster_index] = cluster_data_points[:train_data_length]
        test_data[cluster_index] = cluster_data_points[train_data_length:]

    return train_data, test_data


# ------------------------- end synthetic -------------------------

# ------------------------- begin linear -------------------------

def load_linear_data(config_dict):
    data_generator = LinearDataGenerator(config_dict)
    input_dim = config_dict['LINEAR_data_point_dimension']
    num_models = config_dict['number_of_data_classes']
    delta = config_dict['delta']
    theta_length = config_dict['theta_length']
    theta_stars = generate_thetas(input_dim, num_models, delta, theta_length)
    data_generator.set_theta_stars(theta_stars)
    train_data, test_data = data_generator.load_data()

    return train_data, test_data


# ------------------------- end linear -------------------------

# ------------------------- start AmazonReview -------------------------

def load_AmazonReview(config_dict):
    def load_amazon(base_path):  # modified from https://github.com/FengHZ/KD3A/blob/master/datasets/AmazonReview.py
        dimension = 5000
        amazon = np.load(os.path.join(base_path, "amazon.npz"))
        amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])), shape=amazon['xx_shape'][::-1]).tocsc()
        amazon_xx = amazon_xx[:, :dimension]
        amazon_yy = amazon['yy']
        amazon_yy = (amazon_yy + 1) / 2
        amazon_offset = amazon['offset'].flatten()
        # Partition the data into four categories and for each category partition the data set into training and test set.
        data_name = ["books", "dvd", "electronics", "kitchen"]
        num_data_sets = 4
        data_insts, data_labels, num_insts = [], [], []
        for i in range(num_data_sets):
            data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i + 1], :])
            data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :])
            num_insts.append(amazon_offset[i + 1] - amazon_offset[i])
            # Randomly shuffle.
            r_order = np.arange(num_insts[i])
            np.random.shuffle(r_order)
            data_insts[i] = data_insts[i][r_order, :]
            data_labels[i] = data_labels[i][r_order, :]
            data_insts[i] = data_insts[i].todense().astype(np.float32)
            data_labels[i] = data_labels[i].ravel().astype(np.int64)
        return data_insts, data_labels


    # Get AmazonReview data
    filename = "AmazonReview"
    rawdata_dir = config_dict['data_dir'][config_dict['dataset']]
    if not os.path.exists(rawdata_dir):
        os.makedirs(rawdata_dir)

    # Download zip file
    if not os.path.exists(f"{rawdata_dir}/{filename}.zip"):
        gdown.download(url="https://drive.google.com/u/0/uc?id=1QbXFENNyqor1IlCpRRFtOluI2_hMEd1W", output=f"{rawdata_dir}/{filename}.zip")

    rawdata_path = rawdata_dir + "/" + filename
    # Extract zip file
    if not os.path.exists(rawdata_path):
        # os.system(f'unzip {rawdata_dir}/{filename}.zip -d {rawdata_dir} + "/{filename}"')
        with zipfile.ZipFile(file=f"{rawdata_dir}/{filename}.zip", mode='r') as zip_ref:
            zip_ref.extractall(path=rawdata_dir)

        # Remove extracted zip file
        # os.remove(path=f"{rawdata_dir}/{filename}.zip")

    # Join data points with the corresponding labels
    all_data, all_labels = load_amazon(rawdata_path)
    number_of_labels = len(all_labels)
    all_data_flattened = {}
    for label in range(number_of_labels):
        all_data_flattened[label] = [np.ravel(x) for x in all_data[label]]
    all_data_with_labels = [list(zip(all_data_flattened[i], all_labels[i])) for i in range(len(all_labels))]
    # Partition by label
    train_data, test_data = {}, {}
    for data_class_index in range(number_of_labels):
        train_data[data_class_index], test_data[data_class_index] = train_test_split(all_data_with_labels[data_class_index], train_size=config_dict['train_data_percentage'], random_state=config_dict['seed'])

    return train_data, test_data


# ------------------------- end AmazonReview -------------------------

# ------------------------- start AG_news -------------------------

def load_AG_news(config_dict):
    dataset = load_dataset("ag_news")
    train_data_raw = dataset['train']
    test_data_raw = dataset['test']

    # Simple regex-based tokenizer
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    # Build vocabulary from training set
    counter = Counter()
    for example in train_data_raw:
        tokens = tokenize(example['text'])
        counter.update(tokens)

    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(20000))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    def numericalize(tokens):
        return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    class AGNewsDataset(Dataset):
        def __init__(self, split, label=None):
            data_split = dataset[split]
            if label is not None:
                data_split = data_split.filter(lambda example: example['label'] == label)
            self.data = data_split

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx]['text']
            label = self.data[idx]['label']
            tokens = tokenize(text)
            ids = numericalize(tokens)
            max_len = config_dict['AG_news_max_len']
            if len(ids) < max_len:
                ids += [vocab["<PAD>"]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def build_class_split(split):
        return {i: AGNewsDataset(split, label=i) for i in range(4)}

    train_data = build_class_split("train")
    test_data = build_class_split("test")

    return train_data, test_data


def load_AG_news_old(config_dict):
    dataset = load_dataset("ag_news")
    train_data_raw = dataset['train']
    test_data_raw = dataset['test']

    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    # Build vocab from training set
    counter = Counter()
    for example in train_data_raw:
        tokens = tokenize(example['text'])
        counter.update(tokens)

    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(20000))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    def numericalize(tokens):
        return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    class AGNewsDataset(Dataset):
        def __init__(self, split, label=None):
            self.data = dataset[split]
            if label is not None:
                self.data = self.data.filter(lambda example: example['label'] == label)  # Filter by label

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx]['text']
            label = self.data[idx]['label']
            tokens = tokenize(text)
            ids = numericalize(tokens)[:config_dict['AG_news_max_len']]
            padded = ids + [vocab["<PAD>"]] * (config_dict['AG_news_max_len'] - len(ids))
            return torch.tensor(padded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    # Partition by label
    train_data = {}
    for data_class_index in range(4):
        train_data[data_class_index] = AGNewsDataset("train", label=data_class_index)

    test_data = {}
    for data_class_index in range(4):
        test_data[data_class_index] = AGNewsDataset("test", label=data_class_index)

    return train_data, test_data


# ------------------------- end AG_news -------------------------

# ------------------------- Auxiliary functions -------------------------

def generate_server_testset(config_dict):
    # Create server testset and testloader
    if config_dict['dataset'] == 'MNIST':
        transform = transforms.ToTensor()
        full_MNIST_testset = MNIST(
            root=config_dict['data_dir'],
            train=False,
            download=True,
            transform=transform,
        )

        all_data_classes = list(config_dict['MNIST_all_digits'])
        digits_to_keep = unique_in_order([int(str(data_class).split("r")[0]) for data_class in all_data_classes])     # Only keep each digit once, while preserving the configured order

        # Filter and keep only the specified digits
        server_testset = filter_MNIST_testset(
            full_MNIST_testset=full_MNIST_testset,
            digits_to_keep=digits_to_keep,
        )
    else:       # if config_dict['dataset'] in ['FEMNIST', 'SYNTHETIC']
        server_testset = []

    return server_testset


def load_client_trainset_and_testset(config_dict, theta=0):
    if config_dict['dataset'] == 'MNIST':
        # per_digit_trainsets, testset = load_MNIST_data(digits_to_keep=config_dict['MNIST_all_digits'])
        # train_data, test_data = per_digit_trainsets, testset
        train_data_raw, test_data_raw = load_MNIST_data(config_dict=config_dict, angle=theta)
        train_data, test_data = rotate_MNIST_data_if_needed(train_data_raw, test_data_raw, config_dict)
    elif config_dict['dataset'] == 'CIFAR10':
        train_data, test_data = load_CIFAR10_data(config_dict, angle=theta)
    elif config_dict['dataset'] == 'FEMNIST':   # This should only be used for unsupervised FL
        train_data, test_data = load_FEMNIST_data(config_dict)
    elif config_dict['dataset'] == 'FMNIST':
        train_data, test_data = load_FMNIST_data(config_dict, angle=theta)
    elif config_dict['dataset'] == 'SYNTHETIC':
        train_data, test_data = load_synthetic_data(config_dict)
    elif config_dict['dataset'] == 'LINEAR':
        train_data, test_data = load_linear_data(config_dict)
    elif config_dict['dataset'] == 'AmazonReview':
        train_data, test_data = load_AmazonReview(config_dict)
    elif config_dict['dataset'] == 'AG_news':
        train_data, test_data = load_AG_news(config_dict)
    elif config_dict['dataset'] == 'TinyImageNet':
        train_data, test_data = load_TinyImageNet(config_dict)
    else:
        raise ValueError(f"Invalid value given for config_dict['dataset']: {config_dict['dataset']}")

    return train_data, test_data


def mix_data_classes(train_data_unmixed, test_data_unmixed, config_dict):
    original_labels = list(train_data_unmixed.keys())
    rng = get_seeded_rng(config_dict)

    client_distr_config = None

    logger.info(f"\nData mixture mode: {config_dict['data_mixture_mode']}")

    if config_dict['data_mixture_mode'] == 'no_mixture':
        train_data, test_data = train_data_unmixed, test_data_unmixed
        new_to_original_data_class_labels_mapping_dict = {label: [label] for label in original_labels}
        for cluster_label, data_class_labels in new_to_original_data_class_labels_mapping_dict.items():
            logger.info(f"Cluster {cluster_label} contains all data of data class {data_class_labels[0]}.")

    elif config_dict['data_mixture_mode'] == 'dominant':
        percentage_of_points_from_other_data_classes = config_dict['percentage_of_points_from_other_data_classes']

        train_data, test_data = train_data_unmixed, test_data_unmixed
        for data_class_label in original_labels:
            other_labels = [label for label in original_labels if label != data_class_label]

            number_of_train_points_from_other_data_classes = int(len(train_data_unmixed[data_class_label]) * percentage_of_points_from_other_data_classes)
            random_train_data_classes = random.choices(population=other_labels, k=number_of_train_points_from_other_data_classes)
            random_train_data_points = [
                random.sample(population=train_data_unmixed[random_train_data_classes[i]], k=1)[0]
                for i in range(number_of_train_points_from_other_data_classes)
            ]
            train_data[data_class_label].extend(random_train_data_points)
            rng.shuffle(train_data[data_class_label])

            number_of_test_points_from_other_data_classes = int(len(test_data_unmixed[data_class_label]) * percentage_of_points_from_other_data_classes)
            random_test_data_classes = random.choices(population=other_labels, k=number_of_test_points_from_other_data_classes)
            random_test_data_points = [
                random.sample(population=test_data_unmixed[random_test_data_classes[i]], k=1)[0]
                for i in range(number_of_test_points_from_other_data_classes)
            ]
            test_data[data_class_label].extend(random_test_data_points)
            rng.shuffle(test_data[data_class_label])

        new_to_original_data_class_labels_mapping_dict = {label: [label] for label in original_labels}
        for cluster_label, data_class_labels in new_to_original_data_class_labels_mapping_dict.items():
            logger.info(f"Cluster {cluster_label} contains {(1-percentage_of_points_from_other_data_classes)*100}% data from data class {data_class_labels[0]} and {percentage_of_points_from_other_data_classes*100}% from the other data classes.")

    elif config_dict['data_mixture_mode'] == 'zipped_no_overlap':
        train_data, test_data = {}, {}
        group_size = config_dict['data_class_merge_factor']
        new_to_original_data_class_labels_mapping_dict = {index: original_labels[i:i + group_size] for index, i in enumerate(range(0, len(original_labels), group_size))}
        new_labels = list(new_to_original_data_class_labels_mapping_dict.keys())

        for new_label in new_labels:
            train_data[new_label] = list(itertools.chain.from_iterable([train_data_unmixed[original_label] for original_label in new_to_original_data_class_labels_mapping_dict[new_label]]))
            rng.shuffle(train_data[new_label])
            test_data[new_label] = list(itertools.chain.from_iterable([test_data_unmixed[original_label] for original_label in new_to_original_data_class_labels_mapping_dict[new_label]]))
            rng.shuffle(test_data[new_label])

        for cluster_label, data_class_labels in new_to_original_data_class_labels_mapping_dict.items():
            logger.info(f"Cluster {cluster_label} contains all data from data classes {data_class_labels}.")

    elif config_dict['data_mixture_mode'] == 'zipped_with_overlap':
        exp_type = config_dict['exp_selector']
        if exp_type not in dict_overlap_swap_exps:
            raise ValueError(f"Invalid value given for overlap experiment: {exp_type}")

        exp_val_dict = dict_overlap_swap_exps[exp_type]
        if "label_skew" in exp_val_dict:  # Generating label skewed data
            train_data, group_indices_dict, test_data, _, swaps_lists = get_label_skew_data_split(
                exp_type,
                train_data_unmixed,
                test_data_unmixed
            )
        elif "feature_skew" in exp_val_dict:  # Generating feature skewed data
            train_data, group_indices_dict, test_data, _, swaps_lists = get_feature_skew_data_split(
                exp_type,
                train_data_unmixed,
                test_data_unmixed,
                config_dict,
                load_client_trainset_and_testset
            )
        elif 'dominance_label_skew' in exp_val_dict:  # Generating dominance based label skewed data
            train_data, group_indices_dict, test_data, _, swaps_lists = get_dominance_data_split(
                exp_type,
                train_data_unmixed,
                test_data_unmixed
            )
        else:
            N = exp_val_dict["N_train"]  # Number of train samples per label in each group
            label_groups = exp_val_dict["label_groups"]
            swaps_lists = exp_val_dict["swaps_lists"]
            is_swap = exp_val_dict["swaps_flag"] == 1

            num_points_for_all_classes = [N] * 10

            train_data_list, class_labels = convert_from_dict_to_list(train_data_unmixed)
            train_data, group_indices_dict = split_dataset(train_data_list, class_labels, label_groups, num_points_for_all_classes, swaps_lists, do_swap=is_swap)
            
            N = exp_val_dict["N_test"]  # Number of test samples per label in each group
            num_points_for_all_classes = [N] * 10
            test_data_list, class_labels = convert_from_dict_to_list(test_data_unmixed)
            test_data, group_indices_dict = split_dataset(test_data_list, class_labels, label_groups, num_points_for_all_classes, swaps_lists, do_swap=is_swap)

        new_to_original_data_class_labels_mapping_dict = {index: val for index, val in enumerate(swaps_lists)}

        for cluster_label, data_class_labels in new_to_original_data_class_labels_mapping_dict.items():
            logger.info(f"Cluster {cluster_label} contains all data from data classes {data_class_labels}.")

        client_distr_config = None
        if "client_config" in exp_val_dict:
            client_distr_config = exp_val_dict["client_config"]

    else:
        raise ValueError(f"Invalid value given for config_dict['data_mixture_mode']: {config_dict['data_mixture_mode']}")

    return train_data, test_data, new_to_original_data_class_labels_mapping_dict, client_distr_config


def assign_data_to_clients_RR(train_data, test_data, config_dict):
    """
    Works only for number_of_clients being an integer multiple of number_of_data_classes for now

    Example:
    4 clients, 2 data groups (writers or characters)
    DG1 -> C1, C2
    DG2 -> C3, C4
    """

    rng = get_seeded_rng(config_dict)

    # Convert test_data to a single list of datapoints. Currently, it is a dictionary with the data classes as keys and the data points as values, but this structure is not needed, as the label is already part of each data point (that is a tuple (tensor, label)).
    test_data_all_data_points_in_a_single_list = list(itertools.chain(*test_data.values()))
    rng.shuffle(test_data_all_data_points_in_a_single_list)

    number_of_data_classes = config_dict['number_of_data_classes']
    number_of_clients = config_dict['number_of_clients']
    selected_data_classes = train_data.keys()

    if number_of_clients % number_of_data_classes != 0:
        raise ValueError("Number of clients has to be an exact multiple of the number of data classes.")
    clients_per_data_class = number_of_clients // number_of_data_classes  # 5//2 = 2,     10 // 3 = 3

    trainloaders = []
    testloaders = []

    client_to_data_class_mapping = []
    next_client_index_to_assign_data_to = 0
    for data_class in selected_data_classes:
        total_data_points_for_data_group = len(train_data[data_class])
        chunk_size = total_data_points_for_data_group // clients_per_data_class
        shuffled_indices = list(range(total_data_points_for_data_group))
        rng.shuffle(shuffled_indices)
        w = 0
        for c_index in range(next_client_index_to_assign_data_to, next_client_index_to_assign_data_to + clients_per_data_class):
            chunk_start_index = w * chunk_size
            chunk_end_index = min((w + 1) * chunk_size, w * chunk_size + config_dict['number_of_datapoints_per_client'], total_data_points_for_data_group)
            # client_data = shuffled_train_data_of_data_class[chunk_start_index: chunk_end_index]
            client_data = [train_data[data_class][index] for index in shuffled_indices[chunk_start_index : chunk_end_index]]
            client_to_data_class_mapping.append(data_class)
            w += 1
            logger.info(f"Assigned {len(client_data)} out of {total_data_points_for_data_group} data points from data class {data_class} to client {c_index}.")

            trainloaders.append(DataLoader(dataset=client_data, batch_size=config_dict['local_batch_size'], shuffle=True))
            testloaders.append(DataLoader(dataset=test_data_all_data_points_in_a_single_list, batch_size=config_dict['local_batch_size'], shuffle=False))

        next_client_index_to_assign_data_to += clients_per_data_class

    return trainloaders, testloaders, client_to_data_class_mapping


def assign_data_to_clients_RR_without_replacement(data, config_dict):
    """
    Assigns data to clients WITHOUT replacement, i.e. by partitioning each cluster's data to the correponsing number of clients, using only the following parameters from the config_dict:
        config.number_of_clients
        config.number_of_clusters
    Works only for number_of_clients being an integer multiple of number_of_clusters for now

    Example: 4 clients, 2 clusters
        Cluster 1 -> Client 1, Client 2
        Cluster 2 -> Client 3, Client 4
    """

    rng = get_seeded_rng(config_dict)
    number_of_clusters = config_dict['number_of_clusters']
    number_of_clients = config_dict['number_of_clients']
    selected_clusters = data.keys()

    if number_of_clients % number_of_clusters != 0:
        raise ValueError("Number of clients has to be an exact multiple of the number of clusters.")
    clients_per_cluster = number_of_clients // number_of_clusters  # 5//2 = 2,     10 // 3 = 3

    client_to_cluster_mapping = []
    client_datasets = []
    next_client_index_to_assign_data_to = 0
    for cluster in selected_clusters:
        total_data_points_for_cluster = len(data[cluster])
        if config_dict['dataset'] == 'FEMNIST':
            config_dict['number_of_datapoints_per_client'] = total_data_points_for_cluster
            logger.info(f"FOR FEMNIST, changing number of datapoints for client to {total_data_points_for_cluster}")
        chunk_size = total_data_points_for_cluster // clients_per_cluster
        shuffled_indices = list(range(total_data_points_for_cluster))
        rng.shuffle(shuffled_indices)
        w = 0
        for c_index in range(next_client_index_to_assign_data_to, next_client_index_to_assign_data_to + clients_per_cluster):
            chunk_start_index = w * chunk_size
            chunk_end_index = min((w + 1) * chunk_size, w * chunk_size + config_dict['number_of_datapoints_per_client'], total_data_points_for_cluster)
            client_data = [data[cluster][index] for index in shuffled_indices[chunk_start_index: chunk_end_index]]
            client_to_cluster_mapping.append(cluster)
            w += 1
            logger.info(f"Assigned {len(client_data)}/{total_data_points_for_cluster} data points from data class {cluster} to client {c_index}.")
            client_datasets.append(client_data)

        next_client_index_to_assign_data_to += clients_per_cluster

    return client_datasets, client_to_cluster_mapping


def assign_data_to_clients_RR_with_replacement(data, config_dict):
    """
    Assigns data to clients WITH replacement, using only the following parameters from the config_dict:
        config.number_of_data_classes
        config.number_of_clients_per_cluster
        config.number_of_datapoints_per_client

    Example: 4 clients, 2 clusters
        Cluster 1 -> Client 1, Client 2
        Cluster 2 -> Client 3, Client 4
    """

    clients_per_cluster = config_dict['number_of_clients_per_cluster']
    datapoints_per_client = config_dict['number_of_datapoints_per_client']
    clusters = list(data.keys())

    client_to_cluster_mapping = []
    client_datasets = []

    next_client_index_to_assign_data_to = 0
    for cluster in clusters:
        for client_index in range(next_client_index_to_assign_data_to, next_client_index_to_assign_data_to + clients_per_cluster):
            client_data = random.choices(population=data[cluster], k=datapoints_per_client)
            client_datasets.append(client_data)
            client_to_cluster_mapping.append(cluster)
            logger.info(f"Assigned {datapoints_per_client} data points from cluster {cluster} to client {client_index}.")

        next_client_index_to_assign_data_to += clients_per_cluster

    return client_datasets, client_to_cluster_mapping
