import numpy as np
import matplotlib.pylab as plt

class MeanPlotter:

    def __init__(self, axis, label='mean', color='black', alpha=0.15):
        self._ax = axis
        self._label = label
        self._color = color
        self._alpha = alpha

        self._init = False

    def init(self, data_len):
        self._t = 0
        self._data_len = data_len
        self._data = np.empty((0, data_len))
        self._data_mean = np.empty((0, 1))

        self._plots = [self._ax.plot([], [], '.', color=self._color, alpha=self._alpha)[0] for _ in range(data_len)]
        self._plots_mean = self._ax.plot([], [], '-', color=self._color, alpha=1.0, label=self._label)[0]
        self._ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15))

        self._init = True

    def update(self, x):
        x = np.ravel([x])

        if not self._init:
            self.init(x.shape[0])

        assert x.shape[0] == self._data_len
        x = x.reshape((1, self._data_len))
        mean = np.mean(x).reshape((1, 1))

        self._t += 1
        self._data = np.append(self._data, x, axis=0)
        self._data_mean = np.append(self._data_mean, mean, axis=0)

        [self._plots[i].set_data(np.arange(0, tf), self._data) for i in range(self._data_len)]
        self._plots_mean.set_data(np.arange(0, tf), self._data_mean)

        self._ax.figure.canvas.draw()

if __name__ == "__main__":
    import time
    import random

    plt.ion()
    fig, ax = plt.subplots()
    plotter = MeanPlotter(ax)

    i, j = 0, 0
    while True:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)
        data = [i, j, i+j, i-j]
        plotter.update(data)
        time.sleep(5e-3)
