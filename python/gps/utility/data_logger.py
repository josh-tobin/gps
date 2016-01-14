import pickle
import sys


class DataLogger:
    """
    This class pickles data into files and unpickles data from files.
    TODO: Handle logging text to terminal, gui text, and/or log file at DEBUG, INFO, WARN, ERROR,
        FATAL levels.
    TODO: Handle logging data to terminal, gui text/plots, and/or data files.
    """
    def __init__(self):
        pass

    def pickle(self, filename, data):
        pickle.dump(data, open(filename, 'wb'))

    def unpickle(self, filename):
        try:
            return pickle.load(open(filename, 'rb'))
        except IOError:
            print('Unpickle error. Cannot find file: ' + filename)
        sys.exit()
