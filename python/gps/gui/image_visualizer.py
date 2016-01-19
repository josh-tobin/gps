import numpy as np
import matplotlib.pyplot as plt


class ImageVisualizer:
    """
    If rostopic is given to constructor, then this will automatically update with rostopic image.
    Else, the update method must be manually called.
    """
    def __init__(self, axis, imagesize=None, cropsize=None, rgb_channels=3, rostopic=None):
        self._ax = axis
        self._image_size = imagesize
        self._crop_size = cropsize
        self._rgb_channels = rgb_channels
        self._init = False

        if self._crop_size:
            self.init(self._crop_size)
        elif self._image_size:
            self.init(self._image_size)

        if rostopic is not None:
            try:
                import rospy
                import roslib; roslib.load_manifest('gps_agent_pkg')
                from sensor_msgs.msg import Image

                rospy.Subscriber(rostopic, Image, self.update_ros, queue_size=1, buff_size=2**24)
            except ImportError as e:
                print 'rostopic image visualization not enabled', e

        self._overlay_plot = self._ax.imshow(np.zeros((1,1,3)), alpha=0.0)

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
            image = image[h/2-ch/2:h/2-ch/2+ch,w/2-cw/2:w/2-cw/2+cw,:]

        if not self._init:
            self.init((image.shape[0], image.shape[1]))

        assert image.shape == (self._display_size[0], self._display_size[1], self._rgb_channels)
        image = image.reshape((1, self._display_size[0], self._display_size[1], self._rgb_channels))

        self._t += 1
        self._data = np.append(self._data, image, axis=0)

        self._plot.set_array(self._data[self._t-1])

        self._ax.figure.canvas.draw()

    def update_ros(image_msg):
        # Extract image.
        image = np.fromstring(image_msg.data, np.uint8)
        # Convert from ros image format to matplotlib image format.
        image = image.reshape(curr_im.height, curr_im.width, 3)[::-1,:,::-1]
        # Update visualizer.
        self.update(image)

    def get_current_image():
        if not self._init or self._t == 0:
            return None
        return self._data[self._t-1]

    def overlay_image(self, image):
        if image is None:
            self._overlay_plot.set_array(np.zeros(1,1,3))
            self._overlay_plot.set_alpha(0.0)
        else:
            self._overlay_plot.set_array(image)
            self._overlay_plot.set_alpha(0.2)
        self._ax.figure.canvas.draw()

if __name__ == "__main__":
    import time
    import random

    plt.ion()
    fig, ax = plt.subplots()
    visualizer = ImageVisualizer(ax, cropsize=(3, 3))

    im = np.zeros((5, 5, 3))
    while True:
        i = random.randint(0, im.shape[0] - 1)
        j = random.randint(0, im.shape[1] - 1)
        k = random.randint(0, im.shape[2] - 1)
        im[i,j,k] = (im[i,j,k] + random.randint(0, 255)) % 256
        visualizer.update(im)
        time.sleep(5e-3)
