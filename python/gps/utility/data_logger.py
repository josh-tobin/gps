import copy
import pickle

from gps.utility.data_logger_config import config

class DataLogger:
    """
    This class handles all of the logging from the algorithm.
    Logs text to terminal, gui text, and/or log file at DEBUG, INFO, WARN, ERROR, FATAL levels.
    Logs data to terminal, gui text/plots, and/or data files.
    """

    def __init__(self, hyperparams):
        # Hyperparameters
        self._hyperparams = copy.deepcopy(config)
        self._hyperparams.update(hyperparams)
        self._data_files_dir = self._hyperparams['data_files_dir']

    def pickle(self, data, data_type, itr):
        """
        data: either sample data or algorithm object
        type: string either 'sample' or 'algorithm'
        itr: integer represent iteteration number
        data_dir: the directory in which to look for the data 
        """
        filename = self._data_files_dir + data_type + '_itr_' + str(itr) + '.p'
        pickle.dump(data, open(filename, 'wb'))

    def unpickle(self, data_type, itr):
        """
        type: string either 'sample' or 'algorithm'
        itr: integer represent iteteration number
        data_dir: the directory in which to look for the data 
        """
        filename = self._data_files_dir + data_type + '_itr_' + str(itr) + '.p'
        return pickle.load(open(filename, 'rb'))

    # def load(type, itr, dir=self._data_files_dir):
    #   pass

    # def log_text(self, text, level='INFO'):
    #   """
    #   Args:
 #            text: string of text
 #        """
    #   pass

    # def log_key_value(self, key, value):
    #   pass

    # def log_data_dict(self, data_dict):
    #   for k, v in data_dict.iteritems():
    #       self.log_data(k, v)
