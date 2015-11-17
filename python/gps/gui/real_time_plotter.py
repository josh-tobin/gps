import numpy as np
import matplotlib.pylab as plt

class RealTimePlotter:

    def __init__(self, axis, time_window=500, labels=None):
        self._ax = axis
        self._time_window = time_window
        self._labels = labels
        self._init = False

        if self._labels:
            self.init(len(self._labels))

    def init(self, data_len):
        self._t = 0
        self._data_len = data_len
        self._data = np.empty((0, data_len))

        cm = plt.get_cmap('spectral')
        self._plots = []
        for i in range(data_len):
            color = cm(1.0*i/data_len)
            label = self._labels[i] if self._labels is not None else str(i)
            self._plots.append(self._ax.plot([], [], color=color, label=label)[0])
        self._ax.set_xlim(0, self._time_window)
        self._ax.set_ylim(0, 1)
        self._ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15))

        self._init = True

    def update(self, x):
        x = np.ravel([x])

        if not self._init:
            self.init(x.shape[0])

        assert x.shape[0] == self._data_len
        x = x.reshape((1, self._data_len))

        self._t += 1
        self._data = np.append(self._data, x, axis=0)

        t, tw = self._t, self._time_window
        t0, tf = (0, t)  if t < tw else (t-tw, t)
        [self._plots[i].set_data(np.arange(t0, tf), self._data[t0:tf, i]) for i in range(self._data_len)]
        
        x_range = (0, tw) if t < tw else (t-tw, t)
        self._ax.set_xlim(x_range)

        y_min, y_max = np.amin(self._data[t0:tf,:]), np.amax(self._data[t0:tf,:])
        y_mid, y_dif = (y_min + y_max)/2.0, (y_max-y_min)/2.0
        y_range = y_mid - 1.25*y_dif, y_mid + 1.25*y_dif
        y_range_rounded = np.around(y_range, -int(np.floor(np.log10(np.amax(np.abs(y_range)+1e-100)))) + 1)
        self._ax.set_ylim(y_range_rounded)

        self._ax.figure.canvas.draw()

if __name__ == "__main__":
    import time
    import random

    plt.ion()
    fig, ax = plt.subplots()
    plotter = RealTimePlotter(ax, labels=['i', 'j', 'i+j', 'i-j'])

    i, j = 0, 0
    while True:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)
        plotter.update([i, j, i+j, i-j])
        time.sleep(5e-3)
