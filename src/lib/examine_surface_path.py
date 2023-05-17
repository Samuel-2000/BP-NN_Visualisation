"""
Neural Network Training Progress 2D examinator

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""

import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA
from tqdm import tqdm

from . import functions, nets, paths, plots


def get_grid_distances(grid_surface, grid_pca):
    xy_steps = len(grid_surface)
    grid_distances = [[get_weight_distance(grid_surface[j][i], grid_pca[j][i]) for j in range(xy_steps)] for i in
                      range(xy_steps)]
    return grid_distances


def get_weight_distance(weights_a, weights_b):
    squared_diffs = [(weights_a[i] - weights_b[i]) ** 2 for i in range(len(weights_a))]
    return sum(squared_diffs) ** 0.5


def get_coords(step_size, resolution, start_point_2d, center_point_2d, end_point_2d):
    """
    Calculates coordinates of the grid, on which the parameters will be interpolated.

    :param step_size: step size
    :param resolution: resolution of the visualization
    :param start_point_2d: training start point
    :param center_point_2d: reference point
    :param end_point_2d: reference point
    :return: x coordinate, y coordinate
    """
    converted_x = []
    converted_y = []

    for i in range(-resolution, resolution + 1):
        x_pos = i * step_size[0] + center_point_2d[0]
        y_pos = i * step_size[1] + center_point_2d[1]

        if abs(x_pos - end_point_2d[0]) < step_size[0] / 2:
            converted_x.extend(sorted([x_pos, end_point_2d[0]]))
        elif abs(x_pos - start_point_2d[0]) < step_size[0] / 2:
            converted_x.extend(sorted([x_pos, start_point_2d[0]]))
        else:
            converted_x.append(x_pos)

        if abs(y_pos - end_point_2d[1]) < step_size[1] / 2:
            converted_y.extend(sorted([y_pos, end_point_2d[1]]))
        elif abs(y_pos - start_point_2d[1]) < step_size[1] / 2:
            converted_y.extend(sorted([y_pos, start_point_2d[1]]))
        else:
            converted_y.append(y_pos)

    return converted_x, converted_y


def weights_and_dist(path_2d, center_point_2d, x, y):
    point_distances_L2 = [math.sqrt(((x - p[0]) ** 2) + ((y - p[1]) ** 2)) for p in path_2d]
    point_index = point_distances_L2.index(min(point_distances_L2))
    ptoi_y = y - path_2d[point_index][1]

    if ptoi_y < 0:
        center_weight = 0  # ignore center
        point_weight = 1
    else:
        center_distance = math.sqrt(((x - center_point_2d[0]) ** 2) + ((y - center_point_2d[1]) ** 2))
        center_weight = point_distances_L2[point_index] / (center_distance + point_distances_L2[point_index])
        point_weight = center_distance / (center_distance + point_distances_L2[point_index])

    point_distance = ((x - path_2d[point_index][0]), (y - path_2d[point_index][1]))  # from L2 back to x and y

    return point_weight, center_weight, point_index, point_distance


def get_averaged_2d_point(x, y, x_dir, y_dir, path_2d, path):
    vec_dist = np.array([x, y] - path_2d)
    L2_dist = np.linalg.norm(vec_dist, axis=1)

    weights = softmax(-L2_dist)

    return np.sum(
        (path + x_dir * vec_dist[:, 0, np.newaxis] + y_dir * vec_dist[:, 1, np.newaxis]) * weights[:, np.newaxis],
        axis=0)


def get_averaged_ndim_point(x, y, grid_ndim_point, x_dir, y_dir, path_2d, path):
    vec_dist = np.array([(x - p[0], y - p[1]) for p in path_2d])
    distances = np.linalg.norm(grid_ndim_point - path, axis=1)

    weights = softmax(-distances)

    return np.sum(
        (path + x_dir * vec_dist[:, 0, np.newaxis] + y_dir * vec_dist[:, 1, np.newaxis]) * weights[:, np.newaxis],
        axis=0)


def dim_reduction(path):
    """
    Performs PCA dimension reduction of the path to get the 2D representation.

    :param path: parameters of path points of the model
    :return: path of the optimization algorithm, pca directions and pcvariances
    """
    pca = PCA(n_components=2)
    path_2d = pca.fit_transform(path)
    x_dir, y_dir = pca.components_
    return path_2d, x_dir, y_dir, pca.explained_variance_ratio_


class Examinator2D:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.directory = paths.checkpoints

    def __get_steps(self):
        """
        loads path from file
        """
        steps_numpy = []
        files = os.listdir(os.path.abspath(self.directory))
        files.sort(key=functions.natural_keys)

        for filename in files:
            if "step" in filename:
                with open(os.path.join(os.path.abspath(self.directory), filename), "rb") as fd:
                    try:
                        checkpoint = functions.CpuUnpickler(fd).load()
                        steps_numpy.append(checkpoint["flat_w"].cpu().numpy())
                    except pickle.UnpicklingError:
                        continue
        return steps_numpy

    def __manage_loss(self, pca_train_loader, path_2d, coords, pc_variances, loss, grid, name, file):
        if loss is None:
            loss = self.__compute_loss_2d(pca_train_loader, grid, name)
            with open(file, "wb") as fd:
                pickle.dump(loss, fd)
        plots.surface_compare(path_2d, file, coords, pc_variances, name)

    def __compute_loss_2d(self, pca_train_loader, params_grid, name):
        """
        Calculates the loss of the model on 2D grid.

        :param pca_train_loader: train data set loader
        :param params_grid: parameter grid
        :return: 2D array of validation loss, position of minimum, value of minimum
        """
        xy_steps = len(params_grid)

        loss_2d = [[nets.test(self.model.load_from_flat_params(torch.Tensor(params_grid[x][y]).to(self.device)),
                              pca_train_loader, self.device)[0] for y in range(xy_steps)] for x in
                   tqdm(range(xy_steps), desc=f"{name} surface visualization")]

        return np.array(loss_2d).T

    def get_loss_grid(self, pca_train_loader, resolution=8):
        """
        calculating the loss on the grid point surrounding the training path.

        :param pca_train_loader: train data set loader
        :param resolution: resolution of the visualization (8x2 + 3 = 19x19)
        :return: path of the optimizer in 2D, validation loss grid, coordinates, pcvariances
        """
        path_numpy = self.__get_steps()
        path_2d, x_dir, y_dir, pc_variances = dim_reduction(path_numpy)

        # Find the min and max values along the x and y axes
        min_x_val = np.min(path_2d[:, 0])
        max_x_val = np.max(path_2d[:, 0])
        min_y_val = np.min(path_2d[:, 1])
        max_y_val = np.max(path_2d[:, 1])

        # Find the positions of the min and max values along the x and y axes
        min_x_pos = np.argmin(path_2d[:, 0])
        max_x_pos = np.argmax(path_2d[:, 0])
        min_y_pos = np.argmin(path_2d[:, 1])
        max_y_pos = np.argmax(path_2d[:, 1])

        # Calculate dist_x and dist_y
        dist_x = (max_x_val - min_x_val) * 3 / (resolution * 5)
        dist_y = (max_y_val - min_y_val) * 3 / (resolution * 5)

        # Calculate center_point_2d
        center_point_2d = ((min_x_val + max_x_val) / 2, (min_y_val + max_y_val) / 2)

        # Calculate center_point
        mid_point_x = (path_numpy[min_x_pos] + path_numpy[max_x_pos]) / 2
        mid_point_y = (path_numpy[min_y_pos] + path_numpy[max_y_pos]) / 2
        center_point = (mid_point_x + mid_point_y) / 2

        alpha = (dist_x, dist_y)
        start_point_2d = path_2d[0]
        end_point_2d = path_2d[-1]

        coords = get_coords(alpha, resolution, start_point_2d, center_point_2d, end_point_2d)

        paths_surface = Path(paths.Surface)

        files = [
            paths_surface / "PCA_loss_grid",
            paths_surface / "Sampled_loss_grid",
            paths_surface / "Sampled_with_PCA_loss_grid",
            paths_surface / "Averaged_loss_grid",
            paths_surface / "Averaged_ndim_loss_grid",
            paths_surface / "Weights_distances_grid"
        ]

        losses = [None] * len(files)

        for i, file in enumerate(files):
            if file.exists():
                with open(file, "rb") as fd:
                    losses[i] = pickle.load(fd)

        # Prepare grids
        grids = [[] for _ in range(5)]

        # Iterate over the x and y coordinates
        for x in tqdm(coords[0], desc="parameters interpolation"):
            rows = [[] for _ in range(5)]

            for y in coords[1]:
                # PCA
                rows[0].append(center_point + x_dir * (x - center_point_2d[0]) + y_dir * (y - center_point_2d[1]))

                if not self.args.surface_pca:
                    if not self.args.surface_avg:
                        # sampled
                        pt_weight, center_weight, point_index, point_dist_2d = weights_and_dist(path_2d,
                                                                                                center_point_2d, x, y)
                        closest_point = path_numpy[point_index]
                        rows[1].append(closest_point + x_dir * point_dist_2d[0] + y_dir * point_dist_2d[1])

                        # sampled with pca
                        rows[2].append(pt_weight * rows[1][-1] + center_weight * (
                                center_point + x_dir * (x - center_point_2d[0]) + y_dir * (y - center_point_2d[1])))

                    # averaged
                    if losses[3] is None:
                        rows[3].append(get_averaged_2d_point(x, y, x_dir, y_dir, path_2d, path_numpy))

                    # averaged_ndim
                    if losses[4] is None:
                        rows[4].append(get_averaged_ndim_point(x, y, rows[0][-1], x_dir, y_dir, path_2d, path_numpy))

            # Append rows to grids
            for i in range(0, 5):
                grids[i].append(rows[i])

        names = ["PCA", "Sampled", "Sampled with PCA", "Averaged", "Averaged Multidimensional",
                 'Sampled and Sampled with PCA weights L2 distance']

        if self.args.surface_all:
            for i in range(0, 5):
                self.__manage_loss(pca_train_loader, path_2d, coords, pc_variances, losses[i], grids[i], names[i],
                                   files[i])

            with open(files[5], "wb") as fd:
                pickle.dump(get_grid_distances(grids[1], grids[2]), fd)

            plots.surface_compare(path_2d, files[5], coords, pc_variances, names[5])

        elif self.args.surface_pca:
            self.__manage_loss(pca_train_loader, path_2d, coords, pc_variances, losses[0], grids[0], names[0], files[0])

        else:  # surface_avg
            for i in range(3, 5):
                self.__manage_loss(pca_train_loader, path_2d, coords, pc_variances, losses[i], grids[i], names[i],
                                   files[i])

        return
