"""
Functions to prepare execution of the experiments

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch import optim as optim
from lib import data_loader, nets, paths


def run_prep(args, device):
    """
    Prepares NN
    """
    train_loader, test_loader, _ = data_loader.data_load(args)
    model = get_net(device, train_loader, test_loader, args)

    return train_loader, test_loader, model


def get_net(device, train_loader, test_loader, args):
    """
    Function prepares a neural network model for experiments

    :param device: device to use
    :param train_loader: training dataset loader
    :param test_loader: test dataset loader
    :param args: arguments
    :return: NN model
    """

    # Create instance of neural network
    model_classes = [nets.LeNet, nets.ModifiedLeNet, nets.VGG, nets.TinyNN, nets.TinyCNN]
    try:
        model = model_classes[args.NNmodel]().to(device)
    except IndexError:
        print("Error: model not specified")
        exit(0)

    loss_list = []
    acc_list = []

    # Save initial state of network if not saved yet
    if not paths.model_init_state.exists():
        torch.save(model.state_dict(), paths.model_init_state)

    # Initialize neural network model
    model.load_state_dict(torch.load(paths.model_init_state, map_location=torch.device(device)))
    torch.save(model.state_dict(), os.path.join(paths.checkpoints, f"checkpoint_0"))

    optimizer_class = optim.Adam if args.adam else optim.SGD

    optimizer = optimizer_class(model.parameters(), lr=0.01)  # set optimizer

    if args.step_experiment:
        train_loss_file = open(paths.experiment_train_loss_path, "a+")
    else:
        train_loss_file = open(paths.train_loss_path, "a+")

    if args.step_experiment:
        final_state = paths.final_experiment_state.exists()
    else:
        final_state = paths.final_state.exists()

    # Train model if not trained yet
    if not final_state:
        loss, acc = nets.test(model, test_loader, device)
        loss_list.append(loss)
        acc_list.append(acc)

        for epoch in tqdm(range(1, args.epochs), desc="Model training"):
            nets.train(model, args, train_loader, optimizer, device, epoch, train_loss_file)
            loss, acc = nets.test(model, test_loader, device)
            loss_list.append(loss)
            acc_list.append(acc)
            torch.save(model.state_dict(), os.path.join(paths.checkpoints, f"checkpoint_{epoch}"))

        if args.step_experiment:
            torch.save(model.state_dict(), paths.final_experiment_state)  # save final parameters of model
        else:
            torch.save(model.state_dict(), paths.final_state)  # save final parameters of model

        # Creating last checkpoint file (same as final state)
        filename = os.path.join(paths.checkpoints,
                                f"checkpoint_epoch_{args.epochs - 1}_step_{len(train_loader)}.pkl")
        optim_path = {"flat_w": model.get_flat_params(device), "loss": loss_list[-1]}
        with open(filename, "wb") as fd:
            pickle.dump(optim_path, fd)

        np.savetxt(paths.validation_loss_path, loss_list, fmt='%1.5f')
        np.savetxt(paths.validation_acc_path, acc_list, fmt='%1.2f')

    model.load_state_dict(torch.load(paths.final_state, map_location=torch.device(device)))

    train_loss_file.close()

    return model  # return neural network in final (trained) state


def train_net(args, device):
    """
        Trains Network in a deterministic manner.

        Warning: Works only for the provided model architecture specified in nets.py

        :param args: parameters
        :param device: device to be used
    """

    # Set Determinism
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # can slow down program

    run_prep(args, device)
