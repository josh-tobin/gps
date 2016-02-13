import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

class ThreeDPlotter:
    def __init__(self, fig, gs, num_plots, rows=None, cols=None):
        if cols is None:
            cols = int(np.floor(np.sqrt(num_plots)))
        if rows is None:
            rows = int(np.ceil(float(num_plots)/cols))
        assert(num_plots <= rows*cols, 'Too many plots to put into gridspec.')

        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs)
        self._axarr = [plt.subplot(self._gs[i], projection='3d') for i in range(num_plots)]
        self._lims = [None for i in range(num_plots)]
        self._plots = [[] for i in range(num_plots)]

    def plot(self, i, xs, ys, zs, linestyle='-', linewidth=1.0, marker=None, markersize=5.0, markeredgewidth=0.75, color='black', alpha=1.0, label=''):
        # Manually clip at xlim, ylim, zlim (MPL doesn't support axis limits for 3D plots)
        if self._lims[i]:
            xlim, ylim, zlim = self._lims[i]
            xs[np.any(np.c_[xs < xlim[0], xs > xlim[1]], axis=1)] = np.nan
            ys[np.any(np.c_[ys < ylim[0], ys > ylim[1]], axis=1)] = np.nan
            zs[np.any(np.c_[zs < zlim[0], zs > zlim[1]], axis=1)] = np.nan

        # Create and add plot
        plot = self._axarr[i].plot(xs, ys, zs=zs, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, markeredgewidth=markeredgewidth, color=color, alpha=alpha, label=label)[0]
        self._plots[i].append(plot)

    def plot_3d_points(self, i, points, linestyle='-', linewidth=1.0, marker=None, markersize=5.0, markeredgewidth=0.75, color='black', alpha=1.0, label=''):
        self.plot(i, points[:, 0], points[:, 1], points[:, 2], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, markeredgewidth=markeredgewidth, color=color, alpha=alpha, label=label)

    def plot_3d_gaussian(self, i, mu, sigma, edges=100, linestyle='-.', linewidth=1.0, color='black', alpha=0.1, label=''):
        """
        Plots ellipses in the xy plane representing the Gaussian distributions specified by mu and sigma. 
        Args:
            mu    - Tx3 mean vector for (x, y, z)
            sigma - Tx3x3 covariance matrix for (x, y, z)
            edges - the number of edges to use to construct each ellipse
        """
        p = np.linspace(0, 2*np.pi, edges)
        xy_ellipse = np.c_[np.cos(p), np.sin(p)]
        T = mu.shape[0]
        
        mu_xy, sigma_xy = mu[:,0:2], sigma[:,0:2,0:2]
        u, s, v = np.linalg.svd(sigma_xy)
        mu_xyz = np.repeat(mu.reshape((T, 1, 3)), edges, axis=1)

        for t in range(T):
            xyz = np.repeat(mu[t, :].reshape((1, 3)), edges, axis=0)
            xyz[:, 0:2] += np.dot(xy_ellipse, np.dot(np.diag(np.sqrt(s[t, :])), u[t, :, :].T))
            self.plot_3d_points(i, xyz, linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha, label=label)

    def set_lim(self, i, xlim, ylim, zlim):
        """
        Sets the xlim, ylim, and zlim for plot i
        WARNING: limits must be set before adding data to plots
        Args:
            xlim - a tuple of (x_start, x_end)
            ylim - a tuple of (y_start, y_end)
            zlim - a tuple of (z_start, z_end)
        """
        self._lims[i] = [xlim, ylim, zlim]

    def clear(self, i):
        for plot in self._plots[i]:
            plot.remove()
        self._plots[i] = []

    def draw(self):
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

if __name__ == "__main__":
    import time
    import matplotlib.gridspec as gridspec


    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    plotter = ThreeDPlotter(fig, gs, 5, 2, 3)

    # xyzs = np.zeros((3, 1))
    # while True:
    #     xyz = np.random.randint(-10, 10, size=3).reshape((3,1))
    #     xyzs = np.append(xyzs, xyz, axis=1)
    #     xs, ys, zs = xyzs
    #     plotter.update(xs, ys, zs)
    #     time.sleep(1)
