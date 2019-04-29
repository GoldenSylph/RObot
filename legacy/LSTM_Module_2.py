import pandas as pd
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

warnings.filterwarnings("ignore")

class NeuralNetworkModel(Initable):

    def load_data(self, filename, seq_len, normalise_window):
        self.myDataFromCSV = pd.read_csv(filename)
        self.timestamps = []
        self.prices = []
        self.dates = []

        self.timestamps = self.myDataFromCSV['time'].tolist()
        self.prices = self.myDataFromCSV['last'].tolist()

        for j in self.timestamps:
            self.dates.append(time.ctime(j))

        self.data = self.prices

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
        self.x_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.x_test = result[int(row):, :-1]
        self.y_test = result[int(row):, -1]

        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

        input_length = self.x_train.shape[1]
        input_dim = self.x_train.shape[2]
        output_dim = self.y_train.shape[0]

        print(input_length)#12000
        print(input_dim)#1
        print(output_dim)#2993
        
        return [self.x_train, self.y_train, self.x_test, self.y_test]

    def normalise_windows(self, window_data):
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data

    def build_model(self):
        self.model = Sequential()

        self.model.add(LSTM(
            input_dim=1,
            output_dim=1,
            return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(
            100,
            return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(
            output_dim=1))
        self.model.add(Activation("linear"))

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop")

        print('Compilation Time : %.5f', time.time() - start)
        print(self.model)
        
        return self.model

    """def predict_point_by_point(self, model, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        print('Done predict')
        return predicted

    def predict_sequence_full(self, model, data, window_size):
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        print('Done predict')
        return predicted

    def predict_sequences_multiple(self, model, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        print('Done predict')
        return prediction_seqs"""

    def save_model(self, model, filename, jsonfile):
        model_json = model.to_json()
        with open(jsonfile, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(filename)
        print("Save model")


    def load_model(self, jsonfile):
        json_file = open(jsonfile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        print('End model loading')
        return loaded_model

    def get_cost(self, time):
        return self.model.predict(time)
        """myList = time.split('/')
        secondaryList = myList[2].split(' ')
        myList = myList[:-1]
        l = myList + secondaryList
        d = datetime.datetime(int(l[2]), int(l[1]), int(l[0]), int(l[3]), int(l[4]), int(l[5]))
        
        self.value = []
        self.value = self.prices[:len(self.x_train)]

        for i in range (len(self.dates)):
            if self.dates[i] == d:
                self.result = self.value[i]
            else:
                self.result = 0.000199001731675508"""

        #return self.result

    def initialize(self):
        print('LSTM model init...')
        X_train, y_train, X_test, y_test = self.load_data('data\minutes.csv', 50, True)

        self.model = self.build_model()
        self.model.fit(
            X_train,
            y_train,
            batch_size=512,
            nb_epoch=1,
            validation_split=0.05)
        predictions = self.predict_sequences_multiple(self.model, X_test, 50, 50)

if __name__ == '__main__':
    a = NeuralNetworkModel()
    model = a.initialize()
        
    #save
    a.save_model(model, "model.h5", "model.json")

    #load
##    loaded_model = a.load_model('model.json')
                        
    
##    a.save_model(model)
##    loaded_model = a.load_model('my_model.h5')
    
##    print(a.get_cost('29/03/2019 11 30 00'))
