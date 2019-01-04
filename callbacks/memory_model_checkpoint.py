from tensorflow.train import summary_iterator
from keras.callbacks import ModelCheckpoint
import numpy as np

###############################################################################

# AUTHOR: Michael Hernandez

# PURPOSE: This callback can be included in a callbacks list in a Keras model.
# This callback is nearly identical to the ModelCheckpoint callback, with the
# exception that it takes into account the best score of a model in prior
# training sessions (either through directly providing it or through reading
# a tensorboard logfile).

# Primarily, this can be used when training a model in several batches. You
# may want for Keras to only save the best model among all training batches.

# NOTES: The 'tensorboard_logfile' should point directly to the actual file
# that is produced by tensorboard.

###############################################################################


class MemoryModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, score=None, tensorboard_logfile=None,
                 monitor='val_loss', verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto', period=1):
        """
        :param filepath: Keras' filepath argument. "filepath: string, path to save the model file."
        :param score: float, the score of the existing model you wish to use (either score or
            tensorboard_logfile must be provided).
        :param tensorboard_logfile: string, path to the tensorboard log file you wish to pull
        score metrics from (either score or tensorboard_logfile must be provided).
        :param monitor: Keras' monitor argument. "monitor: quantity to monitor."
        :param verbose: Keras' verbose argument. "verbose: verbosity mode, 0 or 1."
        :param save_best_only: Keras' save_best_only argument, NOTE: This is defaulted to True: "save_best_only:
            if `save_best_only=True`, the latest best model according to the quantity monitored will not be
            overwritten".
        :param save_weights_only: Keras' save_weights_only argument: "save_weights_only: if True,
            then only the model's weights will be saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`)."
        :param mode: Keras' mode argument: "mode: one of {auto, min, max}. If `save_best_only=True`, the decision
            to overwrite the current save file is made based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`, this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is automatically inferred from the name of the monitored
            quantity."
        :param period: Keras' period argument: "Interval (number of epochs) between checkpoints".
        """
        super(MemoryModelCheckpoint, self).__init__(filepath, monitor, verbose,
                 save_best_only, save_weights_only, mode, period)

        if (score is None and tensorboard_logfile is None) or (score is not None and tensorboard_logfile is not None):
            raise Exception("Either (XOR) scroe or logfile must be provided.")
        elif score is not None:
            self.best = score
        elif tensorboard_logfile is not None:
            scores = []
            for x in summary_iterator(tensorboard_logfile):
                for val in x.summary.value:
                    if val.tag == 'loss':
                        scores.append(val.simple_value)

            if self.monitor_op == np.greater:
                self.best = np.max(scores)
            else:
                self.best = np.min(scores)