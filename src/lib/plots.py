"""
Plot library for visualizing the neural network training progress

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.font_manager import FontProperties
from . import paths

color_trained = "dimgrey"

font = FontProperties()
font.set_size(20)


def plot_all_layers(x):
    """
    Function plots all performance of the model with modified layers in one figure

    :param x: data for x-axis (interpolation coefficient)
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Validation loss")
    ax.set_xlabel(r"$\alpha$")

    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel(r"$\alpha$")

    files = [file for file in os.listdir(paths.layers) if not re.search("distance", file) and not re.search("q", file)]
    for file in files:
        lab = file.split('_')[-1][0:-4]
        if re.search("loss", file):
            ax.plot(x, np.loadtxt(os.path.join(paths.layers, file)), label=lab, lw=1)
        if re.search("acc", file):
            ax2.plot(x, np.loadtxt(os.path.join(paths.layers, file)), lw=1)

    ax.plot(x, np.loadtxt(paths.loss_interpolated_model_path), label="all", color=color_trained, linewidth=1)
    ax.axvline(x=0, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.axvline(x=1, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.set_ylim(bottom=0)

    ax2.plot(x, np.loadtxt(paths.accuracy_interpolated_model_path), color=color_trained, linewidth=1)
    ax2.axvline(x=0, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax2.axvline(x=1, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax2.set_ylim(0, 100)

    fig.legend()
    fig.subplots_adjust(bottom=0.17)

    plt.savefig(os.path.join(paths.interpolation_img, f"all_({paths.name}).pdf"), format="pdf")

    plt.close("all")


def surface_compare(steps, file, coords, pcvariances, name):
    """
    Function plots a path on loss grid

    :param file: file to losses
    :param steps: path
    :param coords: coordinates for the grid
    :param pcvariances: chosen pca variances
    :param name: file name
    """
    if file.exists():
        with open(file, "rb") as fd:
            losses = pickle.load(fd)
    else:
        print("Error: losses doesnt exist")
        exit(0)

    loss_file = Path(os.path.join(paths.train_loss_path))
    if loss_file.exists():
        with open(loss_file, "r") as fd:
            # actual_losses = fd.read().split("\n")
            init_loss = fd.readline().strip("\n")
    else:
        print("Error (plots.py): train_loss_path doesnt exist")
        exit(0)

    threshold = float(init_loss) * 1.5
    losses = [[threshold if loss > threshold else loss for loss in row] for row in losses]

    fig, ax = plt.subplots(1, 1)

    plot_surface_window(ax, coords, losses, steps, pcvariances)

    # ax.title.set_text(name)
    plt.savefig(os.path.join(paths.Surface_img, f"{name}_({paths.name}).pdf"), format="pdf")

    plt.close("all")


def plot_surface_window(ax, coords, loss_grid, steps, pcvariances=None):
    """
        Function plots a path on loss grid

        :param ax: plot
        :param steps: steps
        :param loss_grid: validation loss surface
        :param coords: coordinates for the grid
        :param pcvariances: chosen pca variances
        """

    coords_x, coords_y = coords
    im = ax.contourf(coords_x, coords_y, loss_grid, levels=40, alpha=0.9)
    x_steps = [step[0] for step in steps]
    y_steps = [step[1] for step in steps]
    ax.plot(x_steps, y_steps, "r-", lw=3)
    ax.plot(x_steps[-1], y_steps[-1], "ro")

    samples_count = 150
    step = len(x_steps) / samples_count
    last_index = len(x_steps) - 1

    """  # grid visualisation
        for x in coords_x:
            for y in coords_y:
                ax.plot(x, y, "w*")
        """

    for a in range(samples_count):
        index = int(a * step)
        alpha = max(0.3, (0.8 - (a / samples_count)))
        ax.plot(x_steps[index], y_steps[index], ",", color="black", alpha=alpha)

    ax.plot(x_steps[last_index], y_steps[last_index], ",", color="black", alpha=0.3)

    plt.colorbar(im, ax=ax)
    if pcvariances is not None:
        ax.set_xlabel(f"PC1: {pcvariances[0]:.2%}")
        ax.set_ylabel(f"PC2: {pcvariances[1]:.2%}")


def plot_steps(args, steps):
    """
    Function plots steps of a model training path

    :param args: program arguments
    :param steps: steps sizes
    """
    if args.step_experiment:
        loss = open(paths.experiment_train_loss_path).readlines()
    else:
        loss = open(paths.train_loss_path).readlines()

    loss = [float(i) for i in loss]
    steps = get_averaged_steps(steps)
    loss_averaged = get_averaged_steps(loss)

    _, ax = plt.subplots(figsize=(8, 5))

    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Step size", color='tab:red')
    ax.set_xlabel("Averaged step index")
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, max(steps) * 1.1])
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.plot(steps, label="averaged steps", color='tab:red')

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel('Averaged Training Loss', color='tab:blue')  # we already handled the x-label with ax1
    ax2.set_ylim([0, max(loss_averaged) * 1.1])
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.plot(loss_averaged, label="validation loss")

    if args.step_experiment:
        plt.savefig(os.path.join(paths.step_img, f"Steps_experiment_({paths.name}).pdf"), format="pdf")
    else:
        plt.savefig(os.path.join(paths.step_img, f"Steps_({paths.name}).pdf"), format="pdf")

    plt.close("all")


def plot_all_steps():
    """
    finds steps, and combine them into 1 image
    """
    fig, ax = plt.subplots()

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Averaged step size")
    ax.set_xlabel("Averaged step index")

    all_steps = {}
    names_file = Path(os.path.join(paths.all_steps_dir, "models_select.txt"))
    if names_file.exists():
        with open(names_file, "r") as fd:
            models = fd.read().split("\n")
    else:
        models = os.listdir(paths.models_dir)

    for model_dir_name in models:
        step_size_file = Path(os.path.join(paths.models_dir, f"{model_dir_name}\\results\\Steps\\steps"))
        if step_size_file.exists():
            with open(step_size_file, "rb") as fd:
                all_steps[model_dir_name] = pickle.load(fd)

    for key, steps in all_steps.items():
        steps = get_averaged_steps(steps)
        ax.plot(steps, label=key)

    fig.legend(loc=8, ncol=2)
    ax.set_ylim(bottom=0)
    fig.subplots_adjust(bottom=0.16 + ((len(models) // 2 + 1) / 48))

    plt.savefig(os.path.join(paths.all_steps_dir, "all_steps.pdf"), format="pdf")

    plt.close("all")


def get_averaged_steps(steps):
    samples_count = 100
    step = len(steps) // samples_count
    average_steps = [sum(steps[i:i + step]) / step for i in range(0, len(steps), step)]
    if len(steps) % samples_count != 0:
        average_steps[-1] = sum(steps[-(len(steps) % samples_count):]) / (len(steps) % samples_count)
    average_steps[0] = average_steps[1]
    return average_steps
