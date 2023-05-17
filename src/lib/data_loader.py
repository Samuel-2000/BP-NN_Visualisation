"""
Dataset loader module

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import torch.utils.data
from . import paths
import os
from pathlib import Path
from torch import utils as utils
from torchvision import datasets, transforms


def data_load(args):
    """
    Function prepares and loads data
    :param args: dataset name enum arg
    :return: train loader, test loader
    """
    datasets_paths = ["MNIST\\raw\\t10k-images-idx3-ubyte", "cifar-10-batches-py\\data_batch_5",
                      "cifar-100-python\\train"]
    download_state = not Path(os.path.join(paths.dataset, datasets_paths[args.dataset])).exists()

    # prepare subsets
    if args.dataset == 0:
        # preprocess data
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])

        train_set = datasets.MNIST(paths.dataset, train=True, download=download_state, transform=transform)
        test_set = datasets.MNIST(paths.dataset, train=False, download=download_state, transform=transform)
    elif args.dataset == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_set = datasets.CIFAR10(paths.dataset, train=True, download=download_state, transform=transform)
        test_set = datasets.CIFAR10(paths.dataset, train=False, download=download_state, transform=transform)
    elif args.dataset == 2:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_set = datasets.CIFAR100(paths.dataset, train=True, download=download_state, transform=transform)
        test_set = datasets.CIFAR100(paths.dataset, train=False, download=download_state, transform=transform)
    else:
        print("error - zly argument dataset")
        exit(0)

    tr = range(len(train_set))
    te = range(len(test_set))
    pca = range(10000)

    train_set = torch.utils.data.Subset(train_set, tr)
    test_set = torch.utils.data.Subset(test_set, te)
    pca_set = torch.utils.data.Subset(train_set, pca)

    # get data loaders
    train_loader = utils.data.DataLoader(train_set, 64, shuffle=True)
    pca_train_loader = utils.data.DataLoader(pca_set, 1000, shuffle=True)
    test_loader = utils.data.DataLoader(test_set, 1000, shuffle=False)

    return train_loader, test_loader, pca_train_loader
