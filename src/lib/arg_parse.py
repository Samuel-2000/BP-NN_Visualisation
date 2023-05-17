"""
argument parser module

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""

import argparse


def parse_arguments():
    """
    Function parses arguments from command line

    :return: program arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action="store_true", help="Disables CUDA training.")
    parser.add_argument("--train", action="store_true", help="train network")
    parser.add_argument("--NNmodel", type=int, action="store", default=1, nargs='?',
                        help="Set type of NN (default = 1 (new-LeNet)).\n"
                             "0 = old-LeNet"
                             "2 = VGG"
                             "3 = TinyNN"
                             "4 = TinyCNN")
    parser.add_argument("--dataset", type=int, action="store", default=0, nargs='?',
                        help="Set type of training set (default = 0 (MNIST)).\n"
                             "1 = Cifar-10\n"
                             "2 = Cifar-100")
    parser.add_argument("--dropout", type=int, action="store", default=0, help="Apply dropout regularization")

    parser.add_argument("--surface_all", action="store_true",  help="all surface visualizations")
    parser.add_argument("--surface_pca", action="store_true",  help="only pca visualization")
    parser.add_argument("--surface_avg", action="store_true",  help="only avg and avg_ndim visualization")
    parser.add_argument("--step", action="store_true",  help="step size visualization")
    parser.add_argument("--step_experiment", action="store_true", help="step size experiment")
    parser.add_argument("--adam", action="store_true",  help="Adam instead of SGD")
    parser.add_argument("--linear", action="store_true",  help="linear interpolation")
    parser.add_argument("--alpha-steps", type=int, action="store", default=13, nargs='?',
                        help="Set number of 1D interpolation steps (int, default = 20).")
    parser.add_argument("--epochs", type=int, action="store", default=14, nargs='?',
                        help="Set number of training epochs (default = 14).")

    args = parser.parse_args()

    return args
