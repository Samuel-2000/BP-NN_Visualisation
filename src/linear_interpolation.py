"""
Functions to prepare execution of the experiments

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import numpy as np
from prep import run_prep
from lib import paths, examine_linear_interpolation, plots


def linear(args, device):
    """
        :param args: parameters
        :param device: device to be used
    """

    train_loader, test_loader, model = run_prep(args, device)
    alpha = np.linspace(-0.1, 1.1, args.alpha_steps)

    if args.NNmodel <= 2:
        interpolate = examine_linear_interpolation.Linear(model, device, alpha, paths.final_state, paths.model_init_state)  # Constructor
        interpolate.interpolate_all_linear(test_loader)  # interpolate model as whole (Linear)

    if args.NNmodel == 0 or args.NNmodel == 1:
        run_all_LeNet(args, device)
    elif args.NNmodel == 2:
        run_all_VGG(args, device)
    else:
        print("error, not specified model")
        exit(0)

    plot_available(args)


def run_all_LeNet(args, device):
    args.layer = "conv1"
    run_interpolation(args, device)

    args.layer = "conv2"
    run_interpolation(args, device)

    if args.NNmodel == 1:  # New-LeNet
        args.layer = "conv3"
        run_interpolation(args, device)

    args.layer = "fc1"
    run_interpolation(args, device)

    if args.NNmodel == 0:  # Old-LeNet
        args.layer = "fc2"
        run_interpolation(args, device)

        args.layer = "fc3"
        run_interpolation(args, device)


def run_all_VGG(args, device):
    for layer in ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "fc1"]:
        args.layer = layer
        run_interpolation(args, device)


def run_interpolation(args, device):
    """
    Runs the interpolation on layers
    :param args: experiment configuration
    :param device: device to be used
    """
    train_loader, test_loader, model = run_prep(args, device)
    alpha = np.linspace(-0.1, 1.1, args.alpha_steps)
    interpolate = examine_linear_interpolation.Linear(model, device, alpha, paths.final_state, paths.model_init_state)  # Constructor
    interpolate.layers_linear(test_loader, args.layer)  # interpolate layer (Linear)


def plot_available(args):
    x = np.linspace(-0.1, 1.1, args.alpha_steps)
    plots.plot_all_layers(x)
