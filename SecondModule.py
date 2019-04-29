import pandas as pd
import time
import warnings
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import keras

from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from API import Initable

warnings.filterwarnings("ignore")

class NeuralNetworkModel(Initable):

    def initialize_filenames(self):
        self.filename = 'model_saves/second_module/model_structure.h5'
        self.jsonfile = 'model_saves/second_module/model_weights.json'
        self.data_file_name = 'data/minutes.csv'

    def initialize_data(self, filename):
        self.raw_data = pd.read_csv(filename)
        self.data = self.raw_data['last'].tolist()
        self.base_timestamp = self.raw_data['time'].tolist()[0]

    def load_data(self, filename, seq_len, normalise_window):
        #f = open(filename, 'r').read()
        #data = f.split('\n')
        self.initialize_data(filename)
        sequence_length = seq_len + 1
        result = []
        for index in range(len(self.data) - sequence_length):
            result.append(self.data[index: index + sequence_length])

        if normalise_window:
            result = self.normalise_windows(result)

        result = np.array(result)

        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return [x_train, y_train, x_test, y_test]

    def normalise_windows(self, window_data):
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data

    def build_model(self, layers):
        self.model = Sequential()
        self.model.add(LSTM(
            input_dim=layers[0],
            output_dim=layers[1],
            return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(
            layers[2],
            return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(
            output_dim=layers[3]))
        self.model.add(Activation("linear"))

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop")
        print("Compilation Time : %s" % str(time.time() - start))

    def predict_point_by_point(self, model, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def save_model(self, model, filename, jsonfile):
        model_json = model.to_json()
        with open(jsonfile, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(filename)
        print("Save model")

    def load_model(self, filename, jsonfile):
        json_file = open(jsonfile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(filename)
        print('End model loading')

    def denormalise_point(self, point):
        return float(self.data[0]) * (point + 1)

    def get_cost(self, input_time):
        input_timestamp = time.mktime(datetime.datetime.strptime(input_time, "%d/%m/%Y %H %M %S").timetuple())
        print('Base timestamp is: ' + str(self.base_timestamp))
        main_input = np.array([[[input_timestamp - self.base_timestamp]]])
        print('Main input: ' + str(main_input))
        return str(self.denormalise_point(float(self.predict_point_by_point(self.model, main_input))))

    def initialize_and_save(self):
        self.initialize_filenames()
        print('LSTM model init...')
        self.build_model([1, 50, 100, 1])
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(self.data_file_name, 50, True)
        self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=512,
            nb_epoch=1,
            validation_split=0.05)
        self.save_model(self.model, self.filename, self.jsonfile)

    def initialize(self):
        self.initialize_filenames()
        print('LSTM model init...')
        keras.backend.clear_session()
        self.load_model(self.filename, self.jsonfile)
        self.initialize_data(self.data_file_name)
        print('LSTM model initialized...')

if __name__ == '__main__':
    a = NeuralNetworkModel()
    a.initialize_and_save()
