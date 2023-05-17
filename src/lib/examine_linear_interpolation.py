"""
Neural Network Training Progress 1D examinator

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import copy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import nets, paths


class Examinator1D:
    def __init__(self, model, device, alpha, final_state_path, init_state_path):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.theta = copy.deepcopy(torch.load(final_state_path))
        self.theta_f = copy.deepcopy(torch.load(final_state_path))
        self.theta_i = copy.deepcopy(torch.load(init_state_path))


class Linear(Examinator1D):
    def __calc_theta_vec(self, layer, alpha):
        """
        Method calculates the value of parameters on the level of layer at an interpolation point alpha,
        using the linear interpolation.

        :param layer: layer
        :param alpha: interpolation coefficient
        """

        self.theta[layer] = torch.add(
            torch.mul(self.theta_i[layer], (1.0 - alpha)), torch.mul(self.theta_f[layer], alpha)
        )

    def interpolate_all_linear(self, test_loader):
        """
        Method interpolates all parameters of the model and after each interpolation step evaluates the
        performance of the model

        :param test_loader: test loader
        """
        if not paths.loss_interpolated_model_path.exists() or not paths.accuracy_interpolated_model_path.exists():
            v_loss_list = []
            acc_list = []
            layers = [name for name, _ in self.model.named_parameters()]

            self.model.load_state_dict(self.theta_f)
            for alpha_act in tqdm(self.alpha, desc="Model Level Linear", dynamic_ncols=True):
                for layer in layers:
                    self.__calc_theta_vec(layer, alpha_act)
                    self.model.load_state_dict(self.theta)

                loss, acc = nets.test(self.model, test_loader, self.device)
                v_loss_list.append(loss)
                acc_list.append(acc)

            np.savetxt(paths.loss_interpolated_model_path, v_loss_list, fmt='%1.5f')
            np.savetxt(paths.accuracy_interpolated_model_path, acc_list, fmt='%1.2f')
            self.model.load_state_dict(self.theta_f)

    def layers_linear(self, test_loader, layer):
        """
        Method interpolates parameters of selected layer of the model and evaluates the model after each interpolation
        step

        :param test_loader: test loader
        :param layer: layer to be interpolated
        """

        loss_res = Path(f"{paths.loss_interpolated_layer_path}_{layer}.txt")
        acc_res = Path(f"{paths.accuracy_interpolated_layer_path}_{layer}.txt")

        if not loss_res.exists() or not acc_res.exists():
            v_loss_list = np.array([], dtype=np.float64)
            acc_list = np.array([], dtype=np.float64)

            self.model.load_state_dict(self.theta_f)
            for alpha_act in tqdm(self.alpha, desc=f"Layer {layer} Level Linear", dynamic_ncols=True):
                self.__calc_theta_vec(layer + ".weight", alpha_act)
                self.__calc_theta_vec(layer + ".bias", alpha_act)

                self.model.load_state_dict(self.theta)

                v_loss, acc = nets.test(self.model, test_loader, self.device)
                v_loss_list = np.append(v_loss_list, v_loss)
                acc_list = np.append(acc_list, acc)

            np.savetxt(loss_res, v_loss_list, fmt='%1.5f')
            np.savetxt(acc_res, acc_list, fmt='%1.2f')

        self.model.load_state_dict(self.theta_f)
