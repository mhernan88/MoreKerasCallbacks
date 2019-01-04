import unittest
from numpy.random import rand, randint
from datetime import datetime as dt, timedelta as td
from keras.models import Model
from keras.layers import Dense, Input
from callbacks.timer import Timer


class TestTimer(unittest.TestCase):
    def setUp(self):
        self.y_continuous = rand(1000)
        self.y_binary = randint(0,1, 1000)
        self.x = rand(1000, 10)

    def tearDown(self):
        self.y_continuous, self.y_binary, self.x = None, None, None

    def time_neural_network(self, y, final_activation, loss):
        input_layer = Input((self.x.shape[1],))
        dense_layer = Dense(5, activation="relu")(input_layer)
        output_layer = Dense(1, activation=final_activation)(dense_layer)

        m = Model(inputs=[input_layer], output=output_layer)
        m.compile("adam", loss)

        stop_time = dt.now() + td(seconds=30)
        timer_callback = Timer(stop_time)
        cb_list = [timer_callback]

        t1 = dt.now()
        m.fit(x=self.x, y=y, batch_size=32, epochs=1000000000, callbacks=cb_list)
        t2 = dt.now()

        delta = t2 - t1

        return delta

    def test_regression(self):
        time_to_run = self.time_neural_network(self.y_continuous, "linear", "mse")
        self.assertLess(time_to_run.seconds, 120)

    def test_classification(self):
        time_to_run = self.time_neural_network(self.y_binary, "sigmoid", "binary_crossentropy")
        self.assertLess(time_to_run.seconds, 120)
