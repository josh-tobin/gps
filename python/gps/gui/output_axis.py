import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter


class OutputAxis:
    def __init__(self, axis, log_filename=None, max_display_size=10, border_on=False,
            bgcolor='white', bgalpha=0.0, font_family='sans-serif'):
        self._axis = axis
        self._fig = axis.get_figure()
        self._log_filename = log_filename

        self._text_box = self._axis.text(0.02, 0.95, '', color='black', fontsize=12, va='top',
                ha='left', transform=self._axis.transAxes, family=font_family)
        self._text_arr = []
        self._max_display_size = max_display_size
        self._bgcolor = bgcolor
        self._bgalpha = bgalpha
        
        self.cc = ColorConverter()
        self._axis.set_xticks([])
        self._axis.set_yticks([])
        if not border_on:
            self._axis.spines['top'].set_visible(False)
            self._axis.spines['right'].set_visible(False)
            self._axis.spines['bottom'].set_visible(False)
            self._axis.spines['left'].set_visible(False)

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

    def set_bgcolor(self, color, alpha=1.0):
        self._bgcolor = color
        self._bgalpha = alpha
        self.draw()

    def draw(self):
        self._text_box.set_text('\n'.join(self._text_arr))
        self._axis.set_axis_bgcolor(self.cc.to_rgba(self._bgcolor, self._bgalpha))
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
