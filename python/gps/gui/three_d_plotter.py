import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

class ThreeDPlotter:
    def __init__(self, fig, gs, num_plots, rows=None, cols=None):
        if cols is None:
            cols = int(np.floor(np.sqrt(num_plots)))
        if rows is None:
            rows = int(np.ceil(float(num_plots)/cols))

        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs)
        assert(len(self.actions) <= rows*cols, 'Too many actions to put into gridspec.')
        self._axarr = [plt.subplot(self._gs[i], projection='3d') for i in range(num_plots)]
        self._plots = [[] for i in range(num_plots)]

        for ax in self._axarr:
            ax.legend()

    def plot(self, i, xs, ys, zs, label, color):
        plot = self._axarr[i].plot(xs, ys, zs=zs, label=label, color=color)[0]
        self._plots[i].append(plot)

    def clear(self, i):
        for plot in self._plots[i]:
            plot.remove()

    def draw(self):
        self._fig.canvas.draw()
        self._ax.figure.canvas.flush_events()   # Fixes bug with Qt4Agg backend

if __name__ == "__main__":
    import time
    from mpl_toolkits.mplot3d import Axes3D

    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plotter = Plotter3D(ax, label='red', color='red')

    xyzs = np.zeros((3, 1))
    while True:
        xyz = np.random.randint(-10, 10, size=3).reshape((3,1))
        xyzs = np.append(xyzs, xyz, axis=1)
        xs, ys, zs = xyzs
        plotter.update(xs, ys, zs)
        time.sleep(1)
