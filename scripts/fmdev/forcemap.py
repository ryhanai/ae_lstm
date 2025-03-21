# -*- coding: utf-8 -*-

import time
import numpy as np
from abc import ABC
from scipy import stats
from sklearn.neighbors import KernelDensity


from enum import Enum
SmoothingMethod = Enum('SmoothingMethod', [('GAFS', 0), ('IFS', 1), ('SDF', 2)])


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    kde = stats.gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


def kde_sklearn(x, x_grid, sample_weights, bandwidth=1.0, **kwargs):
    kde_skl = KernelDensity(kernel="gaussian", bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x, sample_weight=sample_weights)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid)
    return np.exp(log_pdf)


class ForceMap(ABC):
    pass


class GridForceMap(ForceMap):
    def __init__(self, name, bandwidth=0.010):
        assert name == "seria_basket" or "konbini_shelf" or "small_table"
        if name == "seria_basket":  # IROS2023, moonshot interim review
            self.grid = np.mgrid[-0.13:0.13:40j, -0.13:0.13:40j, 0.73:0.99:40j]
            # X, Y, Z = self.grid
            # self.dV = 0.26 * 0.26 * 0.26 / (40**3)
            # positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            # self.positions = positions.T  # [number of points, 3]
            # self.bandwidth = bandwidth
        elif name == "seria_basket_old":
            self.grid = np.mgrid[-0.095:0.095:40j, -0.13:0.13:40j, 0.73:0.92:40j]
            # X, Y, Z = self.grid
            # self.dV = 0.19 * 0.26 * 0.20 / (40**3)
            # positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            # self.positions = positions.T  # [number of points, 3]
            # self.bandwidth = bandwidth
        elif name == "konbini_shelf":
            self.grid = np.mgrid[-0.3:0.3:120j, -0.4:0.4:160j, 0.73:0.93:40j]
            # X, Y, Z = self.grid
            # positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            # self.positions = positions.T  # [number of points, 3]
            # self.bandwidth = bandwidth
        elif name == "small_table":
            # self.grid = np.mgrid[-0.2:0.2:80j, -0.2:0.2:80j, 0.73:0.93:40j]
            self.grid = np.mgrid[-0.2:0.2:80j, -0.2:0.2:80j, 0.71:0.91:40j]

        X, Y, Z = self.grid
        self._xmax = X.max()
        self._xmin = X.min()
        self._xrange = self._xmax - self._xmin
        self._ymax = Y.max()
        self._ymin = Y.min()
        self._yrange = self._ymax - self._ymin
        self._zmax = Z.max()
        self._zmin = Z.min()
        self._zrange = self._zmax - self._zmin

        self._nx, self._ny, self._nz = X.shape
        self.dV = self._xrange * self._yrange * self._zrange / (self._nx * self._ny * self._nz)
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        self.positions = positions.T  # [number of points, 3]
        self.bandwidth = bandwidth

        # 'sony_box'
        # self.grid = np.mgrid[-0.115:0.115:40j, -0.115:0.115:40j, 0.93:1.16:40j]
        # 'ipad box'

        self.V = np.zeros(self.positions.shape[0])
        self.alpha = 0.8

        self._title = "force map"
        self._scene_name = name

    def get_grid_shape(self):
        return self.grid[0].shape

    def getDensity(self, sample_positions, sample_weights, moving_average=False, return_3d=False):
        if len(sample_weights) > 0:
            # V = kde_scipy(sample_positions, self.positions, bandwidth=0.3)
            # V = stats.gaussian_kde(sample_positions, bw_method=0.3)
            start_kde = time.time()
            V = kde_sklearn(
                sample_positions, self.positions, sample_weights=sample_weights, bandwidth=self.bandwidth, atol=1e-2
            )
            V = V * np.sum(sample_weights)
            print(f"KDE took: {time.time() - start_kde} [sec]")
        else:
            V = np.zeros(self.positions.shape[0])

        if moving_average:
            self.V = self.alpha * self.V + (1 - self.alpha) * V
            result = self.V
        else:
            result = V

        if return_3d:
            return result.reshape(self.get_grid_shape())
        else:
            return result

    def set_values(self, values):
        if values.ndim == 3:
            v = np.zeros((self.grid[0].shape))
            v[:, :, : values.shape[-1]] = values
            self.V = v.reshape(self.V.shape)
        else:
            self.V = values

    def get_values(self):
        return self.V

    def get_positions(self):
        return self.positions

    def set_title(self, title):
        self._title = title

    # def visualize(self, max_channels=20, zaxis_first=False):
    #     V = np.reshape(self.V, self.grid[0].shape)
    #     f = V / np.max(V)
    #     fig = plt.figure(figsize=(16, 6))
    #     fig.subplots_adjust(hspace=0.1)
    #     fig.suptitle(self._title, fontsize=28)

    #     if zaxis_first:
    #         channels = f.shape[0]
    #     else:
    #         channels = f.shape[-1]
    #     for p in range(min(channels, max_channels)):
    #         ax = fig.add_subplot(channels // 10, 10, p + 1)
    #         ax.axis("off")
    #         if zaxis_first:
    #             ax.imshow(f[p], cmap="gray", vmin=0, vmax=1.0)
    #         else:
    #             ax.imshow(f[:, :, p], cmap="gray", vmin=0, vmax=1.0)

    def get_scene(self):
        return self._scene_name


def plot_force_map(force_map, env="seria_basket", title=""):
    fmap = GridForceMap(env)
    fmap.set_values(force_map)
    fmap.set_title(title)
    fmap.visualize()
