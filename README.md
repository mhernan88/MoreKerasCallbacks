# MoreKerasCallbacks [![Build Status](https://travis-ci.org/mhernan88/more_keras_callbacks.svg?branch=master)](https://travis-ci.org/mhernan88/more_keras_callbacks) [![codecov](https://codecov.io/gh/mhernan88/more_keras_callbacks/branch/master/graph/badge.svg)](https://codecov.io/gh/mhernan88/more_keras_callbacks)
Additional useful callbacks for Keras neural network models.

<hr>

Acknowledgements: The entire [Keras team](https://github.com/keras-team/keras).

How To Use:
1. (Optionally) install using (e.g. ```python setup.py install ```)
2. Import the selected callback (e.g. ```from callbacks.timer import Timer```)
3. Configure the callback (e.g. ```stop_time = Timer(datetime.strptime("201801010001","%Y%m%d%H%M%S"))```)
4. Add the callback to your callbacks_list (e.g. ```cb_list = list(Timer(stop_time))```)
5. Add your callbacks list to your training session (e.g. ```model.fit(x, y, callbacks=cb_list)```)

Files:
- memory_model_checkpoint.py - This file contains the MemoryModelCheckpoint callback class. It is very similar to the
ModelCheckpoint callback class. The main exception is that MemoryModelCheckpoint takes into account the scores of a 
prior model training when deciding whether to save. This allows the ModelCheckpoint to find the best model amongst
**all** training sessions rather than just a single training session. MemoryModelCheckpoint can either be provided a
best score or can read the tensorboard logfile of a prior training session and find the best score. Otherwise, this  
class is the same as ModelCheckpoint.
- timer.py - This file contains the Timer callback class. This class takes a datetime object as an input. All this 
class does is check, at the end of each epoch, whether the current system datetime is later than the input datetime.
If the system time is later than the input datetime, training ends.
