import pandas as pd
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf

warnings.filterwarnings("ignore")

class NeuralNetworkModel():
    
    def load_data(self, filename, seq_len, normalise_window):
        self.myDataFromCSV = pd.read_csv(filename)
        self.timestamps_c = []
        self.prices_c = []
        self.dates_c = []

        self.timestamps_c = self.myDataFromCSV['time'].tolist()
        self.prices_c = self.myDataFromCSV['last'].tolist()

        for j in self.timestamps_c:
            self.dates_c.append(time.ctime(j))

        self.data = self.prices_c

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

        layers = [input_dim, 5, input_dim, 5]

        self.build_model(layers)
        
        return [self.x_train, self.y_train, self.x_test, self.y_test]

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

        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.model.fit(self.x_train, self.x_test, batch_size=32, epochs=5)

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop")
        print('Compilation Time : %.5f', time.time() - start)
        print(self.model)

##        self.predict_point_by_point(self.model, self.x_train)
        self.predict_sequence_full(self.model, self.x_train, 100)
        self.save_model()
        self.predict_sequences_multiple(self.model, self.x_train, 100, (len(self.data) - 12000))
        
        return model

    def predict_point_by_point(self, model, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        print('Done predict')
        return property

    def predict_sequence_full(self, model, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
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
        for i in range(len(data)/prediction_len):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in xrange(prediction_len):
                predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        print('Done predict')
        return prediction_seqs

    def save_model(self):
        self.model.save('my_model.h5')

    def load_model(self):
        self.loaded_model = tf.keras.models.load_model('my_model.h5')

if __name__ == '__main__':
    a = NeuralNetworkModel()
    a.load_data('data/minutes.csv', 12000, True)
##    a.load_model()
