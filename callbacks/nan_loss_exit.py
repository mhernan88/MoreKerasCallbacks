from keras.callbacks import Callback
import numpy as np


###############################################################################

# AUTHOR: Michael Hernandez

# PURPOSE: This callback can be included in a callbacks list in a Keras model.
# This callback has Keras stop training new epochs of a model if loss becomes NaN.

###############################################################################

class NaNExit(Callback):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.end_time = end_time

    def on_epoch_end(self, epoch, logs=None):
        """
        :param epoch: The epoch that is ending
        :param logs: Log variable passed to this method
        :return: None
        """
        current = logs.get(self.monitor)
        if np.isnan(current):
            print("Encountered a nan loss. Stopping training.")
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return
