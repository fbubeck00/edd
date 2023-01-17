import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from time import time
from keras.utils.np_utils import to_categorical


class ArcusSenilisCNN:
    def __init__(self, img):
        self.img = img
        self.model = None
        self.history = None
        self.learning_rate = 0.001
        self.n_epochs = 40

    def train(self):
        x_train = None
        y_train = None

        # create validation dataset
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=8)

        # define model architecture
        self.model = Sequential()
        self.model.add(Conv2D(115, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(115, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        # define Optimizer
        opt = tf.keras.optimizers.SGD(lr=self.learning_rate)

        # define loss and optimizer
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Modeling
        start_training = time()
        self.history = self.model.fit(x_train, y_train, epochs=self.n_epochs, validation_data=(x_val, y_val),
                                      batch_size=128, verbose=1)
        end_training = time()

        # Time
        duration_training = end_training - start_training
        duration_training = round(duration_training, 4)

        # Number of Parameter
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        n_params = trainableParams + nonTrainableParams

        # Prediction for Training mse
        loss, error = self.model.evaluate(x_train, y_train, verbose=0)
        error *= 100
        error = round(error, 4)

        # Summary
        print('------ TensorFlow - CNN ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Accuracy Training: ', error)
        print("Number of Parameter: ", n_params)

    def test(self):
        x_test = None
        y_test = None

        # Predict Data
        start_test = time()
        loss, error = self.model.evaluate(x_test, y_test, verbose=0)
        error *= 100
        error = round(error, 4)
        end_test = time()

        # Time
        duration_test = end_test - start_test
        duration_test = round(duration_test, 4)

        print(f'Duration Inference: {duration_test} seconds')

        print("Accuracy Testing: %.2f" % error)
        print("")