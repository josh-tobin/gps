import matplotlib.pyplot as plt

class OutputAxis:

    def __init__(self, axis, max_display_size=5, log_filename=None):
        self._axis = axis
        self._text_arr = []
        self._max_display_size = max_display_size
        self._log_filename = log_filename

    def set_text(self, text):
        self.text_arr = [text]
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

        self._ax_output.clear()
        self._ax_output.set_axis_off()
        self._ax_output.text(0, 1, all_text, color='black', fontsize=12,
            va='top', ha='left', transform=self._ax_output.transAxes)
        self._fig.canvas.draw()

if __name__ == "__main__":
    plt.ion()
    axis = plt.gca()
    output_axis = OutputAxis(axis, max_display_size=5, log_filename=None)
