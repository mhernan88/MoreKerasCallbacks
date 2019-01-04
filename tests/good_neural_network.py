from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard

from numpy.random import rand, normal
from numpy import exp

import os, glob


def wipe_folders():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r_files = glob.glob(os.path.join(base_dir,"tests","test_log_files","regression","events.out.tfevents.*"))
    c_files = glob.glob(os.path.join(base_dir, "tests", "test_log_files", "classification", "events.out.tfevents.*"))

    for f in r_files:
        os.remove(f)
    for f in c_files:
        os.remove(f)


def good_neural_network_regression(nodes, activation):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    x = rand(1000, 4)
    y = x[:,0] * 1.2 + x[:,1] * -0.4 + + x[:,2] * -2.1 + x[:,3] * 1.1 + normal()

    input_layer = Input(shape=(x.shape[1],))
    dense_layer1 = Dense(units=nodes, activation=activation)(input_layer)
    dense_layer2 = Dense(units=nodes, activation=activation)(dense_layer1)
    output_layer = Dense(units=1, activation="linear")(dense_layer2)

    m = Model(inputs=[input_layer], output=[output_layer])
    m.compile("adam", "mse", ["mse"])

    tb = TensorBoard(os.path.join(base_dir,"tests","test_log_files","regression"))
    cb = [tb]

    m.fit(x, y, 32, epochs=100, callbacks=cb)


def good_neural_network_classification(nodes, activation):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    x = rand(1000, 4)
    y = 1/(1+exp(-1 * (x[:, 0] * 1.2 + x[:, 1] * -0.4 + + x[:, 2] * -2.1 + x[:, 3] * 1.1 + normal())))

    input_layer = Input(shape=(x.shape[1],))
    dense_layer1 = Dense(units=nodes, activation=activation)(input_layer)
    dense_layer2 = Dense(units=nodes, activation=activation)(dense_layer1)
    output_layer = Dense(units=1, activation="sigmoid")(dense_layer2)

    m = Model(inputs=[input_layer], output=[output_layer])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    tb = TensorBoard(os.path.join(base_dir, "tests", "test_log_files", "classification"))
    cb = [tb]

    m.fit(x, y, 32, epochs=100, callbacks=cb)


if __name__ == "__main__":
    wipe_folders()
    good_neural_network_regression(10,"relu")
    good_neural_network_classification(10, "relu")