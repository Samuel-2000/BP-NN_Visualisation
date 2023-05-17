"""
Neural Network Models

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import os
import pickle
import torch
from torch import nn as nn
from pathlib import Path
from . import paths
import torch.nn.functional as f
from .arg_parse import parse_arguments


class BaseNN(nn.Module):
    def __init__(self):
        super(BaseNN, self).__init__()
        args = parse_arguments()
        self.dropout = nn.Dropout(0.5)
        self.dropout_toggle = args.dropout
        self.input_channels_num = 1 if args.dataset == 0 else 3  # MNIST/CIFAR-10
        self.output_num = 100 if args.dataset == 2 else 10  # others/CIFAR-100

    def get_flat_params(self, device):
        params = [param.data for _, param in self.named_parameters()]
        flat_params = torch.cat([torch.flatten(param) for param in params]).to(device)
        return flat_params

    def load_from_flat_params(self, f_params):
        shapes = [(name, param.shape, param.numel()) for name, param in self.named_parameters()]
        state = {}
        c = 0
        for name, tsize, tnum in shapes:
            state[name] = torch.nn.Parameter(f_params[c: c + tnum].reshape(tsize))
            c += tnum
        self.load_state_dict(state, strict=True)
        return self


class LeNet(BaseNN):
    """
    Neural network class

    This net consists of 5 layers. Two are convolutional and
    the other three are fully connected.
    """

    def __init__(self):
        super(LeNet, self).__init__()
        # define layers of network
        self.conv1 = nn.Conv2d(self.input_channels_num, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_num)

    def forward(self, x, test_dropout=False):
        """
        Forward pass data

        :param x: Input data
        :param test_dropout: if we are testing, dropout is turned off.
        :return: Output data. Probability of a data sample belonging to one of the classes
        """
        x = f.max_pool2d(f.relu(self.conv1(x)), 2)
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        if self.dropout_toggle == 1 and not test_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return f.log_softmax(self.fc3(x), dim=1)


class ModifiedLeNet(BaseNN):
    def __init__(self):
        super(ModifiedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(self.input_channels_num, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 10, 3, 1)
        self.conv3 = nn.Conv2d(10, 10, 3, 1)
        self.fc1 = nn.Linear(10 * 2 * 2, self.output_num)

    def forward(self, x, test_dropout=False):
        x = f.max_pool2d(f.relu(self.conv1(x)), 2)
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = f.max_pool2d(f.relu(self.conv3(x)), 2)
        if self.dropout_toggle == 1 and not test_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        return f.log_softmax(self.fc1(x), dim=1)


class TinyCNN(BaseNN):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(self.input_channels_num, 3, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, 1)
        self.conv3 = nn.Conv2d(3, 3, 3, 1)
        self.fc1 = nn.Linear(3 * 2 * 2, self.output_num)

    def forward(self, x, test_dropout=False):
        x = f.leaky_relu(f.max_pool2d(self.conv1(x), 2), 0.1)
        x = f.leaky_relu(f.max_pool2d(self.conv2(x), 2), 0.1)
        x = f.leaky_relu(f.max_pool2d(self.conv3(x), 2), 0.1)
        if self.dropout_toggle == 1 and not test_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        return f.log_softmax(self.fc1(x), dim=1)


class TinyNN(BaseNN):
    def __init__(self):
        super(TinyNN, self).__init__()
        # define layers of network
        self.fc1 = nn.Linear(32 * 32 * self.input_channels_num, self.output_num)

    def forward(self, x, test_dropout=False):
        """
        Forward pass data

        :param x: Input data
        :param test_dropout: if we are testing, dropout is turned off.
        :return: Output data. Probability of a data sample belonging to one of the classes
        """
        if self.dropout_toggle == 1 and not test_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        output = f.log_softmax(self.fc1(x), dim=1)
        return output


class VGG(BaseNN):
    def __init__(self):
        super(VGG, self).__init__()

        # define layers of network
        self.conv1 = nn.Conv2d(self.input_channels_num, 16, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(16, 32, 3, 1, padding='same')
        self.conv4 = nn.Conv2d(32, 32, 3, 1, padding='same')
        self.conv5 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.conv6 = nn.Conv2d(64, 64, 3, 1, padding='same')
        self.fc1 = nn.Linear(4 * 4 * 64, self.output_num)

    def forward(self, x, test_dropout=False):
        x = f.max_pool2d(f.relu(self.conv2(f.relu(self.conv1(x)))), 2)
        x = f.max_pool2d(f.relu(self.conv4(f.relu(self.conv3(x)))), 2)
        x = f.max_pool2d(f.relu(self.conv6(f.relu(self.conv5(x)))), 2)
        if self.dropout_toggle == 1 and not test_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        return f.log_softmax(self.fc1(x), dim=1)


def train(model, args, train_loader, optimizer, device, epoch, train_loss_file):
    """
    Trains the network.

    :param model : Neural network model to be trained
    :param args : program arguments
    :param train_loader : Data loader
    :param optimizer : Optimizer
    :param device : Device on which will be the net trained
    :param epoch : Number of actual epoch
    :param train_loss_file : file for train loss output
    :return: training loss for according epoch
    """
    if args.step_experiment:
        # Load the step sizes from the file
        step_file = Path(os.path.join(paths.step_res, "steps"))
        if step_file.exists():
            with open(step_file, "rb") as fd:
                step_sizes = pickle.load(fd)
        else:
            print("Error: step-file does not exist")
            exit(0)

    model.train()
    train_loss = 0
    optim_path = {"flat_w": [], "loss": []}
    # i = 0  # sampled
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if args.step_experiment:
            current_step = (epoch - 1) * len(train_loader) + batch_idx
            learning_rate = step_sizes[current_step]
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

        if args.step_experiment:
            filename = Path(os.path.join(paths.experiment_checkpoints), f"checkpoint_epoch_{epoch}_step_{batch_idx}.pkl")
        else:
            filename = Path(os.path.join(paths.checkpoints), f"checkpoint_epoch_{epoch}_step_{batch_idx}.pkl")

        optim_path["flat_w"] = model.get_flat_params(device)
        optim_path["loss"] = loss

        train_loss_file.write(f"{str(float(loss.data.cpu().numpy()))}\n")
        # if (i + ((epoch - 1) * (60000 // 64))) % 100 == 0: # sampled
        with open(filename, "wb") as fd:
            pickle.dump(optim_path, fd)
        # i = i + 1 # sampled

    return train_loss / len(train_loader.dataset)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, test_dropout=True)
            test_loss += f.nll_loss(output, target, reduction="sum").item()
            correct += (output.argmax(dim=1) == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy
