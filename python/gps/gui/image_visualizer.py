""" This file defines the image visualizer class. """
import logging
import random
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.action_axis import Action, ActionAxis

LOGGER = logging.getLogger(__name__)


class ImageVisualizer(object):
    """
    If rostopic is given to constructor, then this will automatically
    update with rostopic image. Else, the update method must be manually
    called.
    """
    def __init__(self, hyperparams, fig, gs, imagesize=None, cropsize=None, rgb_channels=3, rostopic=None):
        self._hyperparams = hyperparams

        # Real-time image
        self._image_size = imagesize
        self._crop_size = cropsize
        self._rgb_channels = rgb_channels
        self._init = False

        if self._crop_size:
            self.init(self._crop_size)
        elif self._image_size:
            self.init(self._image_size)

        # Image overlay
        self._initial_image = np.zeros((1,1,3))
        self._initial_alpha = 0.0
        self._target_image = np.zeros((1,1,3))
        self._target_alpha = 0.0
        self._initial_image_overlay_on = False
        self._target_image_overlay_on = False

        # Actions
        actions_arr = [
            Action('oii', 'overlay_initial_image', self.toggle_initial_image_overlay, axis_pos=0),
            Action('oti', 'overlay_target_image',  self.toggle_target_image_overlay,  axis_pos=1),
        ]
        self._actions = {action._key: action for action in actions_arr}
        for key, action in self._actions.iteritems():
            if key in self._hyperparams['keyboard_bindings']:
                action._kb = self._hyperparams['keyboard_bindings'][key]
            if key in self._hyperparams['ps3_bindings']:
                action._pb = self._hyperparams['ps3_bindings'][key]

        # GUI Components
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=gs)
        self._gs_action_axis = self._gs[0:1, 0]
        self._gs_image_axis  = self._gs[1:8, 0]

        self._action_axis = ActionAxis(self._fig, self._gs_action_axis, 1, 2, self._actions,
                ps3_process_rate=self._hyperparams['ps3_process_rate'],
                ps3_topic=self._hyperparams['ps3_topic'],
                ps3_button=self._hyperparams['ps3_button'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])
        
        self._ax_image = plt.subplot(self._gs_image_axis)
        self._ax_image.set_axis_off()
        self._plot = self._ax_image.imshow(np.zeros((1,1,3)))
        self._overlay_plot_initial = self._ax_image.imshow(self._initial_image, alpha=self._initial_alpha)
        self._overlay_plot_target  = self._ax_image.imshow(self._target_image , alpha=self._target_alpha)

        self._fig.canvas.draw()

        # ROS subscriber for PS3 controller
        if rostopic is not None:
            try:
                import rospy
                import roslib
                from sensor_msgs.msg import Image

                roslib.load_manifest('gps_agent_pkg')
                rospy.Subscriber(rostopic, Image, self.update_ros, queue_size=1,
                                 buff_size=2**24)
            except ImportError as e:
                LOGGER.debug('rostopic image visualization not enabled: %s', e)

    def init(self, display_size):
        """ Initialize images. """
        self._t = 0
        self._display_size = display_size
        display_x, display_y = self._display_size
        self._data = np.empty((0, display_x, display_y, self._rgb_channels))
        self._init = True

    def update(self, image):
        """ Update images. """
        image = np.array(image)
        if self._crop_size:
            h, w = image.shape[0], image.shape[1]
            ch, cw = self._crop_size[0], self._crop_size[1]
            image = image[(h/2-ch/2):(h/2-ch/2+ch), (w/2-cw/2):(w/2-cw/2+cw), :]

        if not self._init:
            self.init((image.shape[0], image.shape[1]))

        assert image.shape == (self._display_size[0], self._display_size[1],
                               self._rgb_channels)
        image = image.reshape((1, self._display_size[0], self._display_size[1],
                               self._rgb_channels))

        self._t += 1
        self._data = np.append(self._data, image, axis=0)

        self._plot.set_array(self._data[self._t-1])
        self.draw()

    def update_ros(self, image_msg):
        # Extract image.
        image = np.fromstring(image_msg.data, np.uint8)
        # Convert from ros image format to matplotlib image format.
        image = image.reshape(image_msg.height, image_msg.width, 3)[:,:,::-1]
        image = 255 - image
        # Update visualizer.
        self.update(image)

    def get_current_image(self):
        if not self._init or self._t == 0:
            return None
        return self._data[-1]

    def set_initial_image(self, image, alpha=0.2):
        self._initial_image = image
        self._initial_alpha = alpha

    def set_target_image(self, image, alpha=0.2):
        self._target_image = image
        self._target_alpha = 0.2

    def toggle_initial_image_overlay(self, event=None):
        self._initial_image_overlay_on = not self._initial_image_overlay_on
        if (self._initial_image_overlay_on):
            self._overlay_plot_initial.set_array(self._initial_image)
            self._overlay_plot_initial.set_alpha(self._initial_alpha)
        else:
            self._overlay_plot_initial.set_array(np.zeros((1,1,3)))
            self._overlay_plot_initial.set_alpha(0.0)
        self.draw()

    def toggle_target_image_overlay(self, event=None):
        self._target_image_overlay_on = not self._target_image_overlay_on
        if (self._target_image_overlay_on):
            self._overlay_plot_target.set_array(self._target_image)
            self._overlay_plot_target.set_alpha(self._target_alpha)
        else:
            self._overlay_plot_target.set_array(np.zeros((1,1,3)))
            self._overlay_plot_target.set_alpha(0.0)
        self.draw()

    def draw(self):
        self._ax_image.draw_artist(self._ax_image.patch)
        self._ax_image.draw_artist(self._plot)
        self._ax_image.draw_artist(self._overlay_plot_initial)
        self._ax_image.draw_artist(self._overlay_plot_target)
        self._fig.canvas.update()
        # self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots()
    visualizer = ImageVisualizer(ax, cropsize=(3, 3))

    im = np.zeros((5, 5, 3))
    while True:
        i = random.randint(0, im.shape[0] - 1)
        j = random.randint(0, im.shape[1] - 1)
        k = random.randint(0, im.shape[2] - 1)
        im[i, j, k] = (im[i, j, k] + random.randint(0, 255)) % 256
        visualizer.update(im)
        time.sleep(5e-3)
