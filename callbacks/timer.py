from keras.callbacks import Callback
from datetime import datetime as dt


###############################################################################

# AUTHOR: Michael Hernandez

# PURPOSE: This callback can be included in a callbacks list in a Keras model.
# This callback has Keras stop training new epochs of a model after a given
# datetime.

# NOTES: The "logs" variable in on_epoch_end has no use in the actual function.
# It is there to match the arguments of the inherited "Callback" class.

###############################################################################

class Timer(Callback):
    def __init__(self, end_time, *args, **kwargs):
        """
        :param end_time: The datetime at which you wish to stop training your
        model. Should be a python datetime object.
        """
        super().__init__(*args, **kwargs)
        self.end_time = end_time

    def on_epoch_end(self, epoch, logs=None):
        """
        :param epoch: The epoch that is ending
        :param logs: Log variable passed to this method
        :return: None
        """
        if dt.now() > self.end_time:
            print("Reached ending time. Stopping training.")
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return