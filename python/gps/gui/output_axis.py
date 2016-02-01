""" This file defines the output axis. """
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ColorConverter


class OutputAxis:
    def __init__(self, fig, gs, log_filename=None, max_display_size=10, border_on=False,
            bgcolor='white', bgalpha=0.0, font_family='sans-serif'):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])
        self._log_filename = log_filename

        self._text_box = self._ax.text(0.02, 0.95, '', color='black', fontsize=12, va='top',
                ha='left', transform=self._ax.transAxes, family=font_family)
        self._text_arr = []
        self._max_display_size = max_display_size
        self._bgcolor = bgcolor
        self._bgalpha = bgalpha

        self.cc = ColorConverter()
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        if not border_on:
            self._ax.spines['top'].set_visible(False)
            self._ax.spines['right'].set_visible(False)
            self._ax.spines['bottom'].set_visible(False)
            self._ax.spines['left'].set_visible(False)

        self.draw()

    #TODO: Add docstrings here.
    def set_text(self, text):
        self._text_arr = [text]
        self.log_text(text)
        self.draw()

    def append_text(self, text):
        self._text_arr.append(text)
        if len(self._text_arr) > self._max_display_size:
            self._text_arr = self._text_arr[-self._max_display_size:]
        self.log_text(text)
        self.draw()

    def log_text(self, text):
        if self._log_filename is not None:
            with open(self._log_filename, 'a') as f:
                f.write(text + '\n')

    def set_bgcolor(self, color, alpha=1.0):
        self._bgcolor = color
        self._bgalpha = alpha
        self.draw()

    def draw(self):
        self._text_box.set_text('\n'.join(self._text_arr))
        self._ax.set_axis_bgcolor(self.cc.to_rgba(self._bgcolor, self._bgalpha))
        self._fig.canvas.draw()


if __name__ == "__main__":
    plt.ion()

    axis = plt.gca()
    output_axis = OutputAxis(axis, max_display_size=5, log_filename=None)

    for i in range(10):
        output_axis.append_text(str(i))
        time.sleep(1)

    plt.ioff()
    plt.show()
