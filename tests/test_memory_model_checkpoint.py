import unittest, glob, os
from numpy.random import rand, randint
from keras.models import Model
from keras.layers import Dense, Input
from callbacks.memory_model_checkpoint import MemoryModelCheckpoint


class TestBase(unittest.TestCase):
    def get_models(self):
        return glob.glob(os.path.join(self.base_dir, "tests", "test_checkpoints/*.h5"))

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.y_continuous = rand(1000) * 10000
        self.y_binary = randint(0,1, 1000)
        self.x = rand(1000, 4)

    def tearDown(self):
        self.y_continuous, self.y_binary, self.x = None, None, None
        files = self.get_models()
        for f in files:
            os.remove(f)


class TestMemoryModelCheckpointWithScore(TestBase):
    def run_neural_network(self, y, final_activation, loss, metrics, score, mode, monitor):
        input_layer = Input((self.x.shape[1],))
        dense_layer = Dense(5, activation="relu")(input_layer)
        output_layer = Dense(1, activation=final_activation)(dense_layer)

        m = Model(inputs=[input_layer], output=output_layer)
        m.compile("adam", loss, metrics=[metrics])

        memory_callback = MemoryModelCheckpoint(filepath=os.path.join(self.base_dir,"tests","test_checkpoints",
                                                                      "test.h5"), mode=mode, score=score,
                                                monitor=monitor, verbose=1)
        cb_list = [memory_callback]

        m.fit(x=self.x, y=y, batch_size=32, epochs=10, callbacks=cb_list, verbose=1, validation_split=0.2)
        return

    def test_regression_neg(self):
        self.run_neural_network(self.y_continuous, "linear", "mse", "mse", 0.00, "min", "val_mean_squared_error")
        files = self.get_models()
        self.assertLess(len(files),1)

    def test_classification_neg(self):
        self.run_neural_network(self.y_binary, "sigmoid", "binary_crossentropy", "acc", 1.01, "max", "val_acc")
        files = self.get_models()
        self.assertLess(len(files),1)

    def test_regression_pos(self):
        self.run_neural_network(self.y_binary, "linear", "mse", "mse", 100000000000, "min", "val_mean_squared_error")
        files = self.get_models()
        self.assertGreater(len(files),0)

    def test_classification_pos(self):
        self.run_neural_network(self.y_binary, "sigmoid", "binary_crossentropy", "acc", 0.00, "max", "val_acc")
        files = self.get_models()
        self.assertGreater(len(files),0)


class TestMemoryModelCheckpointWithTensorbaordLog(TestBase):
    def run_neural_network(self, y, final_activation, loss, metrics, tensorboard_logfile, mode, monitor):
        input_layer = Input((self.x.shape[1],))
        dense_layer = Dense(5, activation="relu")(input_layer)
        output_layer = Dense(1, activation=final_activation)(dense_layer)

        m = Model(inputs=[input_layer], output=output_layer)
        m.compile("adam", loss, metrics=[metrics])

        memory_callback = MemoryModelCheckpoint(filepath=os.path.join(self.base_dir,"tests","test_checkpoints",
                                                                      "test.h5"), mode=mode,
                                                tensorboard_logfile=tensorboard_logfile,
                                                monitor=monitor, verbose=1)
        cb_list = [memory_callback]

        m.fit(x=self.x, y=y, batch_size=32, epochs=1, callbacks=cb_list, verbose=1, validation_split=0.2)
        return

    def test_regression_neg(self):
        try:
            f = glob.glob(os.path.join(self.base_dir,"tests","test_log_files","regression",
                                       "events.out.tfevents.*"))[0]
            self.run_neural_network(self.y_continuous, "linear", "mse", "mse", f, "min", "val_mean_squared_error")
            files = self.get_models()
            print("Was able to successfully train model.")
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_classification_neg(self):
        try:
            f = glob.glob(os.path.join(self.base_dir, "tests", "test_log_files", "classification",
                                       "events.out.tfevents.*"))[0]
            self.run_neural_network(self.y_binary, "sigmoid", "binary_crossentropy", "acc", f, "max", "val_acc")
            files = self.get_models()
            print("Was able to successfully train model.")
            self.assertTrue(True)
        except:
            self.assertTrue(False)