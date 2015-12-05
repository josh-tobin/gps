import numpy as np
import matplotlib.pylab as plt

class ImageVisualizer:

    def __init__(self, axis, imagesize=None, cropsize=None, rgb_channels=3):
        self._ax = axis
        self._image_size = imagesize
        self._crop_size = cropsize
        self._rgb_channels = rgb_channels
        self._init = False

        if self._crop_size:
            self.init(self._crop_size)
        elif self._image_size:
            self.init(self._image_size)

    def init(self, display_size):
        self._t = 0
        self._display_size = display_size
        display_x, display_y = self._display_size
        self._data = np.empty((0, display_x, display_y, self._rgb_channels))

        self._ax.set_axis_off()
        self._ax.set_xlim(0, display_x)
        self._ax.set_ylim(0, display_y)
        self._plot = self._ax.imshow(np.zeros((display_x, display_y, self._rgb_channels)))

        self._init = True

    def update(self, image):
        image = np.array(image)
        if self._crop_size:
            h, w, ch, cw = image.shape[0], image.shape[1], self._crop_size[0], self._crop_size[1]
            image = image[h/2-ch/2:h/2-ch/2+ch, w/2-cw/2:w/2-cw/2+cw, :]

        if not self._init:
            self.init((image.shape[0], image.shape[1]))

        assert image.shape == (self._display_size[0], self._display_size[1], self._rgb_channels)
        image = image.reshape((1, self._display_size[0], self._display_size[1], self._rgb_channels))

        self._t += 1
        self._data = np.append(self._data, image, axis=0)

        self._plot.set_array(self._data[self._t-1])

        self._ax.figure.canvas.draw()

if __name__ == "__main__":
    import time
    import random

    plt.ion()
    fig, ax = plt.subplots()
    visualizer = ImageVisualizer(ax, cropsize=(3,3))

    im = np.zeros((5, 5, 3))
    while True:
        i = random.randint(0, im.shape[0]-1)
        j = random.randint(0, im.shape[1]-1)
        k = random.randint(0, im.shape[2]-1)
        im[i, j, k] = (im[i, j, k] + random.randint(0, 255)) % 256
        visualizer.update(im)
        time.sleep(5e-3)
