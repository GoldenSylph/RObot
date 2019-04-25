import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from functools import reduce
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import requests
import datetime as dt
from datetime import timedelta
import calendar
import time
from scipy.special import binom
import urllib.request, json
import os
import http.client
import tensorflow as tf

class DataGeneratorSeq(object):

    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data,batch_labels

    def unroll_batches(self):

        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()    

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))

class NeuralNetworkModel():

    def read_csv_file(self):
        self.myDataFromCSV = pd.read_csv('minutes.csv')
        print(self.myDataFromCSV)

        self.timestamps_c = []
        self.prices_c = []
        
        self.timestamps_c = self.myDataFromCSV.loc[:,'time'].as_matrix()
        self.prices_c = self.myDataFromCSV.loc[:,'last'].as_matrix()
        
        self.prices_c = self.prices_c[-12000:]
        
        self.timestamps_c = self.timestamps_c[-12000:]
        
        self.dates_c = []

        for j in self.timestamps_c:
            self.dates_c.append(time.ctime(j))

        self.MSE_errors = []
        
        self.df_comp = {'dates': self.dates_c, 'prices': self.prices_c}
        self.df_c = pd.DataFrame(data=self.df_comp)
        self.df_c = self.df_c.sort_values('dates')
        self.df_c = self.df_c.reset_index(drop=True)
        self.df_c.head()
        
        self.vizialize_initial_data()

    
    def vizialize_initial_data(self):
        plt.figure(figsize = (18,9))
        plt.plot(range(self.df_c.shape[0]),(self.df_c['prices']))
        plt.xticks(range(0,self.df_c.shape[0],500),self.df_c['dates'].loc[::500],rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.show()
        
        self.divide_by_train_test_data()


    def divide_by_train_test_data(self):
        self.prices = self.df_c.loc[:,'prices'].as_matrix()

        self.train_data = self.prices[:11000]
        self.test_data = self.prices[11000:]
        
        self.scale_data()


    def scale_data(self):
        self.scaler = MinMaxScaler()
        self.train_data = self.train_data.reshape(-1,1)
        self.test_data = self.test_data.reshape(-1,1)

        self.train_scaler()


    def train_scaler(self):
        self.smoothing_window_size = 2500
        for di in range(0,10000,self.smoothing_window_size):
            self.scaler.fit(self.train_data[di:di+self.smoothing_window_size,:])
            self.train_data[di:di+self.smoothing_window_size,:] = self.scaler.transform(self.train_data[di:di+self.smoothing_window_size,:])

        # You normalize the last bit of remaining data
        self.scaler.fit(self.train_data[di+self.smoothing_window_size:,:])
        self.train_data[di+self.smoothing_window_size:,:] = self.scaler.transform(self.train_data[di+self.smoothing_window_size:,:])

        # Reshape both train and test data
        self.train_data = self.train_data.reshape(-1)

        # Normalize test data
        self.test_data = self.scaler.transform(self.test_data).reshape(-1)

        self.perform_exponential_moving_average()


    def perform_exponential_moving_average(self):
        self.EMA = 0.0
        self.gamma = 0.1
        for ti in range(11000):
          self.EMA = self.gamma*self.train_data[ti] + (1-self.gamma)*self.EMA
          self.train_data[ti] = self.EMA

        self.all_mid_data = np.concatenate([self.train_data,self.test_data],axis=0)

        self.standart_average_calc()


    def standart_average_calc(self):
        self.window_size = 100
        self.N = self.train_data.size
        self.std_avg_predictions = []
        self.std_avg_x = []
        self.mse_errors = []

        for pred_idx in range(self.window_size,self.N):

            if pred_idx >= self.N:
                self.date = self.dt_c.datetime.strptime(k, '%Y-%m-%d %H:%M:%S').date() + dt.timedelta(days=1)
            else:
                self.date = self.df_c.loc[pred_idx,'dates']

            self.std_avg_predictions.append(np.mean(self.train_data[pred_idx-self.window_size:pred_idx]))
            self.mse_errors.append((self.std_avg_predictions[-1]-self.train_data[pred_idx])**2)
            self.std_avg_x.append(self.date)

        print('MSE error for standard averaging: %.5f'%(0.5*np.mean(self.mse_errors)))

        self.standart_average_vizualize()


    def standart_average_vizualize(self):
        plt.figure(figsize = (18,9))
        plt.plot(range(self.df_c.shape[0]),self.all_mid_data,color='b',label='True')
        plt.plot(range(self.window_size,self.N),self.std_avg_predictions,color='orange',label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(fontsize=18)
        plt.show()

        self.exponential_moving_average_calc()


    def exponential_moving_average_calc(self):
        self.window_size = 100
        self.N = self.train_data.size

        self.run_avg_predictions = []
        self.run_avg_x = []

        self.mse_errors = []

        self.running_mean = 0.0
        self.run_avg_predictions.append(self.running_mean)

        self.decay = 0.5

        for pred_idx in range(1,self.N):

            self.running_mean = self.running_mean*self.decay + (1.0-self.decay)*self.train_data[pred_idx-1]
            self.run_avg_predictions.append(self.running_mean)
            self.mse_errors.append((self.run_avg_predictions[-1]-self.train_data[pred_idx])**2)
            self.run_avg_x.append(self.date)

        print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(self.mse_errors)))

        self.exponential_moving_average_vizualize()


    def exponential_moving_average_vizualize(self):
        plt.figure(figsize = (18,9))
        plt.plot(range(self.df_c.shape[0]),self.all_mid_data,color='b',label='True')
        plt.plot(range(0,self.N),self.run_avg_predictions,color='orange', label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(fontsize=18)
        plt.show()

        self.start_lstm()


    def start_lstm(self):
        self.dg = DataGeneratorSeq(self.train_data,5,5)
        self.u_data, self.u_labels = self.dg.unroll_batches()

        for ui,(dat,lbl) in enumerate(zip(self.u_data,self.u_labels)):   
            print('\n\nUnrolled index %d'%ui)
            dat_ind = dat
            lbl_ind = lbl
            print('\tInputs: ',dat )
            print('\n\tOutput:',lbl)

        self.defining_hyperparameters()


    def defining_hyperparameters(self):
        self.D = 1
        self.num_unrollings = 50
        self.batch_size = 500
        self.num_nodes = [200,200,150]
        self.n_layers = len(self.num_nodes)
        self.dropout = 0.2

        tf.reset_default_graph()

        self.unroll_input()


    def unroll_input(self):
        self.train_inputs, self.train_outputs = [],[]

        # unroll the input over time defining placeholders for each time step
        for ui in range(self.num_unrollings):
            self.train_inputs.append(tf.placeholder(tf.float32, shape=[self.batch_size,self.D],name='train_inputs_%d'%ui))
            self.train_outputs.append(tf.placeholder(tf.float32, shape=[self.batch_size,1], name = 'train_outputs_%d'%ui))

        self.defining_lstm_parameters()


    def defining_lstm_parameters(self):
        self.lstm_cells = [
            tf.contrib.rnn.LSTMCell(num_units=self.num_nodes[li],
                                    state_is_tuple=True,
                                    initializer= tf.contrib.layers.xavier_initializer()
                                   )
        for li in range(self.n_layers)]

        self.drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
            lstm, input_keep_prob=1.0,output_keep_prob=1.0-self.dropout, state_keep_prob=1.0-self.dropout
        ) for lstm in self.lstm_cells]
        self.drop_multi_cell = tf.contrib.rnn.MultiRNNCell(self.drop_lstm_cells)
        self.multi_cell = tf.contrib.rnn.MultiRNNCell(self.lstm_cells)

        self.w = tf.get_variable('w',shape=[self.num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))

        self.calculating_lstm_output()


    def calculating_lstm_output(self):
        self.c, self.h = [],[]
        self.initial_state = []
        for li in range(self.n_layers):
          self.c.append(tf.Variable(tf.zeros([self.batch_size, self.num_nodes[li]]), trainable=False))
          self.h.append(tf.Variable(tf.zeros([self.batch_size, self.num_nodes[li]]), trainable=False))
          self.initial_state.append(tf.contrib.rnn.LSTMStateTuple(self.c[li], self.h[li]))

        # Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
        # a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
        self.all_inputs = tf.concat([tf.expand_dims(t,0) for t in self.train_inputs],axis=0)

        # all_outputs is [seq_length, batch_size, num_nodes]
        self.all_lstm_outputs, self.state = tf.nn.dynamic_rnn(
            self.drop_multi_cell, self.all_inputs, initial_state=tuple(self.initial_state),
            time_major = True, dtype=tf.float32)

        self.all_lstm_outputs = tf.reshape(self.all_lstm_outputs, [self.batch_size*self.num_unrollings,self.num_nodes[-1]])

        self.all_outputs = tf.nn.xw_plus_b(self.all_lstm_outputs,self.w,self.b)

        self.split_outputs = tf.split(self.all_outputs,self.num_unrollings,axis=0)

        self.defining_training_loss()


    def defining_training_loss(self):
        print('Defining training Loss')
        self.loss = 0.0
        with tf.control_dependencies([tf.assign(self.c[li], self.state[li][0]) for li in range(self.n_layers)]+
                                     [tf.assign(self.h[li], self.state[li][1]) for li in range(self.n_layers)]):
          for ui in range(self.num_unrollings):
            self.loss += tf.reduce_mean(0.5*(self.split_outputs[ui]-self.train_outputs[ui])**2)

        print('Learning rate decay operations')
        self.global_step = tf.Variable(0, trainable=False)
        self.inc_gstep = tf.assign(self.global_step,self.global_step + 1)
        self.tf_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)
        self.tf_min_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)

        self.learning_rate = tf.maximum(
            tf.train.exponential_decay(self.tf_learning_rate, self.global_step, decay_steps=1, decay_rate=0.5, staircase=True),
            self.tf_min_learning_rate)

        self.optimization()


    def optimization(self):
        print('TF Optimization operations')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients, self.v = zip(*self.optimizer.compute_gradients(self.loss))
        self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)
        self.optimizer = self.optimizer.apply_gradients(
            zip(self.gradients, self.v))

        print('\tAll done')

        self.maintaning_lstm()


    def maintaning_lstm(self):
        #Prediction Related Calculations
        print('Defining prediction related TF functions')

        self.sample_inputs = tf.placeholder(tf.float32, shape=[1,self.D])

        # Maintaining LSTM state for prediction stage
        self.sample_c, self.sample_h, self.initial_sample_state = [],[],[]
        for li in range(self.n_layers):
          self.sample_c.append(tf.Variable(tf.zeros([1, self.num_nodes[li]]), trainable=False))
          self.sample_h.append(tf.Variable(tf.zeros([1, self.num_nodes[li]]), trainable=False))
          self.initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(self.sample_c[li],self.sample_h[li]))

        self.reset_sample_states = tf.group(*[tf.assign(self.sample_c[li],tf.zeros([1, self.num_nodes[li]])) for li in range(self.n_layers)],
                                       *[tf.assign(self.sample_h[li],tf.zeros([1, self.num_nodes[li]])) for li in range(self.n_layers)])

        self.sample_outputs, self.sample_state = tf.nn.dynamic_rnn(self.multi_cell, tf.expand_dims(self.sample_inputs,0),
                                           initial_state=tuple(self.initial_sample_state),
                                           time_major = True,
                                           dtype=tf.float32)

        with tf.control_dependencies([tf.assign(self.sample_c[li],self.sample_state[li][0]) for li in range(self.n_layers)]+
                                      [tf.assign(self.sample_h[li],self.sample_state[li][1]) for li in range(self.n_layers)]):  
          self.sample_prediction = tf.nn.xw_plus_b(tf.reshape(self.sample_outputs,[1,-1]), self.w, self.b)

        print('\tAll done')

        self.running_lstm()


    def running_lstm(self):
        self.epochs = 30
        self.valid_summary = 1 # Interval you make test predictions

        self.n_predict_once = 50 # Number of steps you continously predict for

        self.train_seq_length = self.train_data.size # Full length of the training data

        self.train_mse_ot = [] # Accumulate Train losses
        self.test_mse_ot = [] # Accumulate Test loss
        self.predictions_over_time = [] # Accumulate predictions

        self.session = tf.InteractiveSession()

        tf.global_variables_initializer().run()

        # Used for decaying learning rate
        self.loss_nondecrease_count = 0
        self.loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate

        print('Initialized')
        self.average_loss = 0

        # Define data generator
        self.data_gen = DataGeneratorSeq(self.train_data,self.batch_size,self.num_unrollings)

        self.x_axis_seq = []

        # Points you start your test predictions from
        self.test_points_seq = np.arange(11000,12000,50).tolist()

        self.training()

    def training(self):
        for ep in range(self.epochs):       

            # ========================= Training =====================================
            for step in range(self.train_seq_length//self.batch_size):

                self.u_data, self.u_labels = self.data_gen.unroll_batches()

                self.feed_dict = {}
                for ui,(dat,lbl) in enumerate(zip(self.u_data,self.u_labels)):            
                    self.feed_dict[self.train_inputs[ui]] = dat.reshape(-1,1)
                    self.feed_dict[self.train_outputs[ui]] = lbl.reshape(-1,1)

                self.feed_dict.update({self.tf_learning_rate: 0.0001, self.tf_min_learning_rate:0.000001})

                _, l = self.session.run([self.optimizer, self.loss], feed_dict=self.feed_dict)

                self.average_loss += l

            # ============================ Validation ==============================
            if (ep+1) % self.valid_summary == 0:

              self.average_loss = self.average_loss/(self.valid_summary*(self.train_seq_length//self.batch_size))

              # The average loss
              if (ep+1)%self.valid_summary==0:
                print('Average loss at step %d: %f' % (ep+1, self.average_loss))

              self.train_mse_ot.append(self.average_loss)

              self.average_loss = 0 # reset loss

              self.predictions_seq = []

              self.mse_test_loss_seq = []

              # ===================== Updating State and Making Predicitons ========================
              for w_i in self.test_points_seq:
                self.mse_test_loss = 0.0
                self.our_predictions = []

                if (ep+1)-self.valid_summary==0:
                  # Only calculate x_axis values in the first validation epoch
                  self.x_axis=[]

                # Feed in the recent past behavior of stock prices
                # to make predictions from that point onwards
                for tr_i in range(w_i-self.num_unrollings+1,w_i-1):
                  self.current_price = self.all_mid_data[tr_i]
                  self.feed_dict[self.sample_inputs] = np.array(self.current_price).reshape(1,1)    
                  _ = self.session.run(self.sample_prediction,feed_dict=self.feed_dict)

                self.feed_dict = {}

                self.current_price = self.all_mid_data[w_i-1]

                self.feed_dict[self.sample_inputs] = np.array(self.current_price).reshape(1,1)

                # Make predictions for this many steps
                # Each prediction uses previous prediciton as it's current input
                for pred_i in range(self.n_predict_once):

                  self.pred = self.session.run(self.sample_prediction,feed_dict=self.feed_dict)

                  self.our_predictions.append(np.asscalar(self.pred))

                  self.feed_dict[self.sample_inputs] = np.asarray(self.pred).reshape(-1,1)

                  if (ep+1)-self.valid_summary==0:
                    # Only calculate x_axis values in the first validation epoch
                    self.x_axis.append(w_i+pred_i)

                  self.mse_test_loss += 0.5*(self.pred-self.all_mid_data[w_i+pred_i])**2

                self.session.run(self.reset_sample_states)

                self.predictions_seq.append(np.array(self.our_predictions))

                self.mse_test_loss /= self.n_predict_once
                self.mse_test_loss_seq.append(self.mse_test_loss)

                if (ep+1)-self.valid_summary==0:
                  self.x_axis_seq.append(self.x_axis)

              self.current_test_mse = np.mean(self.mse_test_loss_seq)

              # Learning rate decay logic
              if len(self.test_mse_ot)>0 and self.current_test_mse > min(self.test_mse_ot):
                  self.loss_nondecrease_count += 1
              else:
                  self.loss_nondecrease_count = 0

              if self.loss_nondecrease_count > self.loss_nondecrease_threshold :
                    self.session.run(self.inc_gstep)
                    self.loss_nondecrease_count = 0
                    print('\tDecreasing learning rate by 0.5')

              self.test_mse_ot.append(self.current_test_mse)
              
              self.MSE_errors.append(np.mean(self.mse_test_loss_seq))
              
              print('\tTest MSE: %.5f'%np.mean(self.mse_test_loss_seq))
              self.predictions_over_time.append(self.predictions_seq)
              print('\tFinished Predictions')

        self.predictions_vizializing()

    def predictions_vizializing(self):
        self.best_prediction_epoch = 28 # replace this with the epoch that you got the best results when running the plotting code

        plt.figure(figsize = (18,18))
        plt.subplot(2,1,1)
        plt.plot(range(self.df_c.shape[0]),self.all_mid_data,color='b')

        # Plotting how the predictions change over time
        # Plot older predictions with low alpha and newer predictions with high alpha
        self.start_alpha = 0.25
        self.alpha  = np.arange(self.start_alpha,1.1,(1.0-self.start_alpha)/len(self.predictions_over_time[::3]))
        for p_i,p in enumerate(self.predictions_over_time[::3]):
            for xval,yval in zip(self.x_axis_seq,p):
                plt.plot(xval,yval,color='r',alpha=self.alpha[p_i])

        plt.title('Evolution of Test Predictions Over Time',fontsize=18)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.xlim(11000,12500)

        plt.subplot(2,1,2)

        # Predicting the best test prediction you got
        plt.plot(range(self.df_c.shape[0]),self.all_mid_data,color='b')
        for xval,yval in zip(self.x_axis_seq,self.predictions_over_time[self.best_prediction_epoch]):
            plt.plot(xval,yval,color='r')

        plt.title('Best Test Predictions Over Time',fontsize=18)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.xlim(11000,12500)
        plt.show()

        self.mse_error_calc()

    def mse_error_calc(self):
        self.sum_errors = 0
        self.count = 0
        for error in self.MSE_errors:
            self.sum_errors += error
            self.count += 1

        self.MSE_error_result = self.sum_errors/self.count
        print('MSE_ERROR: %.5f'%self.MSE_error_result)

        self.get_price()

    def get_price(self, time):

        self.value = []
        self.value = prices[:len(self.train_data)]
        self.value.extend(self.predictions_over_time[self.best_prediction_epoch])
            
        for i in range (len(self.dates)):
            if self.dates[i] == time:
                result = self.value[i]

        return result

if __name__ == '__main__':
    a = NeuralNetworkModel()
    a.read_csv_file()
