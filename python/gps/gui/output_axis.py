import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter

class OutputAxis:

    def __init__(self, axis, max_display_size=5, log_filename=None):
        self._axis = axis
        self._fig = axis.get_figure()
        self._text_arr = []
        self._max_display_size = max_display_size
        self._log_filename = log_filename

        self.cc = ColorConverter()
        self.draw()

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

    def draw(self):
        all_text = '\n'.join(self._text_arr)

        self._axis.clear()
        self._axis.set_axis_off()
        self._axis.text(0, 1, all_text, color='black', fontsize=12,
            va='top', ha='left', transform=self._axis.transAxes)
        self._fig.canvas.draw()

    def set_bgcolor(self, color, alpha=1.0):
        self._axis.set_axis_on()
        self._axis.set_xticks([])
        self._axis.set_yticks([])
        self._axis.set_axis_bgcolor(self.cc.to_rgba(color, alpha))
        self._fig.canvas.draw()

if __name__ == "__main__":
    import time

    plt.ion()

    axis = plt.gca()
    output_axis = OutputAxis(axis, max_display_size=5, log_filename=None)

    for i in range(10):
        output_axis.append_text(str(i))
        time.sleep(1)

    plt.ioff()
    plt.show()
