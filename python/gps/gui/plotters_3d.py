import numpy as np
import matplotlib.pylab as plt

class Plotters3D:
    """
    Plot multiple 3D plots in a single axis.
    """

    def __init__(self, axis):
        self._ax = axis

    def clear():
        self._ax.clear()
        self._ax.legend()

    def plot(self, xs, ys, zs, color, label):
        self._ax.plot(xs, ys, zs=zs, color=color, label=label)
        
    def draw():
        self._ax.figure.canvas.draw()
        self._ax.figure.canvas.flush_events()   # Fixes bug with Qt4Agg backend

if __name__ == "__main__":
    import time
    from mpl_toolkits.mplot3d import Axes3D

    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plotter = Plotters3D(ax)

    xyzs = np.zeros((3, 1))
    while True:
        xyz = np.random.randint(-10, 10, size=3).reshape((3,1))
        xyzs = np.append(xyzs, xyz, axis=1)
        xs, ys, zs = xyzs
        plotter.update(xs, ys, zs)
        time.sleep(1)
