import numpy as np
import matplotlib.pylab as plt

class Plotter3D:

    def __init__(self, axis, label='Plotter3D', color='black'):
        self._ax = axis
        self._label = label
        self._color = color

        self._plot = self._ax.plot([], [], [], color=self._color, label=self._label)[0]
        self._ax.legend()

    def update(self, xs, ys, zs):
        self._plot = self._ax.plot(xs, ys, zs=zs, color=self._color, label=self._label)[0]
        self._ax.figure.canvas.draw()
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
