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

sns.set(font_scale=1.2)

raw_data = None
years_to_substract = 3

sns.set(font_scale=1.2)
df = pd.DataFrame(data = None)
raw_data = None

try:
    i = 0 
    frame = [None] * 120
    cursor = ['', '36124'
                ,'36922'
                ,'37669'
                ,'38975'
                ,'39691'
                ,'40475'
                ,'40830'
                ,'41591'
                ,'42178'
                ,'42838'
                ,'43508'
                ,'43924'
                ,'44580'
                ,'45295'
                ,'45995'
                ,'46794'
                ,'47098'
                ,'47333'
                ,'47520'
                ,'48063'
                ,'49003'
                ,'50174'
                ,'50979'
                ,'52780'
                ,'54123'
                ,'55646'
                ,'57153'
                ,'61797'
                ,'65267'
                ,'68161'
                ,'73014'
                ,'77034'
                ,'80610'
                ,'85639'
                ,'89092'
                ,'95736'
                ,'100394'
                ,'104947'
                ,'110674'
                ,'112970'
                ,'118768'
                ,'124681'
                ,'134533'
                ,'140909'
                ,'146740'
                ,'152164'
                ,'155883'
                ,'160398'
                ,'164852'
                ,'173021'
                ,'182121'
                ,'185748'
                ,'193739'
                ,'197430'
                ,'200594'
                ,'207210'
                ,'211540'
                ,'218889'
                ,'228924'
                ,'237026'
                ,'254964'
                ,'264283'
                ,'271798'
                ,'287230'
                ,'300424'
                ,'311978'
                ,'320307'
                ,'327216'
                ,'335203'
                ,'352789'
                ,'362040'
                ,'372393'
                ,'381395'
                ,'398395'
                ,'422571'
                ,'448588'
                ,'462869'
                ,'474464'
                ,'475312'
                ,'476696'
                ,'480280'
                ,'483913'
                ,'489561'
                ,'496075'
                ,'501169'
                ,'507398'
                ,'514153'
                ,'524222'
                ,'535626'
                ,'549273'
                ,'555018'
                ,'567147'
                ,'577458'
                ,'582183'
                ,'593661'
                ,'605828'
                ,'621776'
                ,'634019'
                ,'637877'
                ,'644137'
                ,'654666'
                ,'668562'
                ,'677149'
                ,'681195'
                ,'687529'
                ,'694800'
                ,'703314'
                ,'711586'
                ,'720483'
                ,'732830'
                ,'744826'
                ,'752897'
                ,'761273'
                ,'774350'
                ,'786971'
                ,'797786'
                ,'812313'
                ,'836155'
                ,'847252']

    for i in range(len(cursor)):
        if (cursor[i] == ''):
            request_raw = 'https://api.idex.market/returnTradeHistory?market=ETH_AURA&start=%d&end=%d&count=100' % (time.mktime((dt.datetime.now() - timedelta(days=365*years_to_substract)).timetuple()), time.mktime(dt.datetime.now().timetuple()))
            response = requests.get(request_raw)
            raw_data = response.json()
            dates = []
            prices = []
            seller_fees = []
            buyer_fees = []
            gas_fees = []
            usd_values = []
            for entry in raw_data:
                dates.append(entry['date'])
                prices.append(float(entry['price']))
                seller_fees.append(float(entry['sellerFee']))
                buyer_fees.append(float(entry['buyerFee']))
                gas_fees.append(float(entry['gasFee']))
                usd_values.append(float(entry['usdValue']))

            composed_data = {'dates': dates, 'prices': prices,
                             'seller_fees': seller_fees, 'buyer_fees': buyer_fees,
                             'gas_fees': gas_fees, 'usd_values': usd_values}
            df = pd.DataFrame(data=composed_data)
            df['dates'] = pd.to_datetime(df['dates'], format='%Y-%m-%d %H:%M:%S', yearfirst=True, errors='ignore')
            frame[i] = df
            print(i)
        else:
            cursor_string = '&cursor=' + cursor[i]
            request_raw = 'https://api.idex.market/returnTradeHistory?market=ETH_AURA&start=%d&end=%d&sort=asc&count=100%s' % (time.mktime(((2017, 11, 22, 0, 0, 0, 0, 0, 0))), time.mktime(dt.datetime.now().timetuple()), cursor_string)
            response = requests.get(request_raw)
            raw_data = response.json()
            dates = []
            prices = []
            seller_fees = []
            buyer_fees = []
            gas_fees = []
            usd_values = []
            for entry in raw_data:
                dates.append(entry['date'])
                prices.append(float(entry['price']))
                seller_fees.append(float(entry['sellerFee']))
                buyer_fees.append(float(entry['buyerFee']))
                gas_fees.append(float(entry['gasFee']))
                usd_values.append(float(entry['usdValue']))
            composed_data = {'dates': dates, 'prices': prices,
                             'seller_fees': seller_fees, 'buyer_fees': buyer_fees,
                             'gas_fees': gas_fees, 'usd_values': usd_values}
            df = pd.DataFrame(data=composed_data)
            df['dates'] = pd.to_datetime(df['dates'], format='%Y-%m-%d %H:%M:%S', yearfirst=True, errors='ignore')
            frame[i] = df
            print(i)

    df = pd.concat(frame, ignore_index=True)

    #Sort data by date
    df = df.sort_values('dates')

    df = df.reset_index(drop=True)

    #Double check the result
    df.head()

    print(df)
        

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['prices']))
    plt.xticks(range(0,df.shape[0],500),df['dates'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Price',fontsize=18)
    plt.show()


    prices = df.loc[:,'prices'].as_matrix()

    train_data = prices[:11000]
    test_data = prices[11000:]

    # Scale the data to be between 0 and 1
    # When scaling remember! You normalize both test and train data with respect to training data
    # Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    # Train the Scaler with training data and smooth data
    smoothing_window_size = 2500
    for di in range(0,10000,smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)

    # Now perform exponential moving average smoothing
    # So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1
    for ti in range(11000):
      EMA = gamma*train_data[ti] + (1-gamma)*EMA
      train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data,test_data],axis=0)



    ##STANDART AVERAGE
    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size,N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d %H:%M:%S').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx,'dates']

        std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

    ##vizualization
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(fontsize=18)
    plt.show()


    ##EXPONENTIAL MOVING AVERAGE
    window_size = 100
    N = train_data.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1,N):

        running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
        run_avg_x.append(date)

    print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(fontsize=18)
    plt.show()



    #LSTM
    class DataGeneratorSeq(object):

        def __init__(self,prices,batch_size,num_unroll):
            self._prices = prices
            self._prices_length = len(self._prices) - num_unroll
            self._batch_size = batch_size
            self._num_unroll = num_unroll
            self._segments = self._prices_length //self._batch_size
            self._cursor = [offset * self._segments for offset in range(self._batch_size)]

        def next_batch(self):

            batch_data = np.zeros((self._batch_size),dtype=np.float32)
            batch_labels = np.zeros((self._batch_size),dtype=np.float32)

            for b in range(self._batch_size):
                if self._cursor[b]+1>=self._prices_length:
                    #self._cursor[b] = b * self._segments
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



    dg = DataGeneratorSeq(train_data,5,5)
    u_data, u_labels = dg.unroll_batches()

    for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):   
        print('\n\nUnrolled index %d'%ui)
        dat_ind = dat
        lbl_ind = lbl
        print('\tInputs: ',dat )
        print('\n\tOutput:',lbl)


    #Defining Hyperparameters
    D = 1 # Dimensionality of the data. Since your data is 1-D this would be 1
    num_unrollings = 50 # Number of time steps you look into the future.
    batch_size = 500 # Number of samples in a batch
    num_nodes = [200,200,150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
    n_layers = len(num_nodes) # number of layers
    dropout = 0.2 # dropout amount

    tf.reset_default_graph() # This is important in case you run this multiple times


    #Defining Inputs and Outputs
    # Input data.
    train_inputs, train_outputs = [],[]

    # You unroll the input over time defining placeholders for each time step
    for ui in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
        train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))


    #Defining Parameters of the LSTM and Regression layer
    lstm_cells = [
        tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
                                state_is_tuple=True,
                                initializer= tf.contrib.layers.xavier_initializer()
                               )
     for li in range(n_layers)]

    drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
    ) for lstm in lstm_cells]
    drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
    multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

    w = tf.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))


    ##Calculating LSTM output and Feeding it to the regression layer to get final prediction
    # Create cell state and hidden state variables to maintain the state of the LSTM
    c, h = [],[]
    initial_state = []
    for li in range(n_layers):
      c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
      h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
      initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

    # Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
    # a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)

    # all_outputs is [seq_length, batch_size, num_nodes]
    all_lstm_outputs, state = tf.nn.dynamic_rnn(
        drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
        time_major = True, dtype=tf.float32)

    all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

    all_outputs = tf.nn.xw_plus_b(all_lstm_outputs,w,b)

    split_outputs = tf.split(all_outputs,num_unrollings,axis=0)


    ##Loss Calculation and Optimizer
    # When calculating the loss you need to be careful about the exact form, because you calculate
    # loss of all the unrolled steps at the same time
    # Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps

    print('Defining training Loss')
    loss = 0.0
    with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)]+
                                 [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
      for ui in range(num_unrollings):
        loss += tf.reduce_mean(0.5*(split_outputs[ui]-train_outputs[ui])**2)

    print('Learning rate decay operations')
    global_step = tf.Variable(0, trainable=False)
    inc_gstep = tf.assign(global_step,global_step + 1)
    tf_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)
    tf_min_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)

    learning_rate = tf.maximum(
        tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
        tf_min_learning_rate)

    # Optimizer.
    print('TF Optimization operations')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v))

    print('\tAll done')


    #Prediction Related Calculations
    print('Defining prediction related TF functions')

    sample_inputs = tf.placeholder(tf.float32, shape=[1,D])

    # Maintaining LSTM state for prediction stage
    sample_c, sample_h, initial_sample_state = [],[],[]
    for li in range(n_layers):
      sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
      sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
      initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li],sample_h[li]))

    reset_sample_states = tf.group(*[tf.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                                   *[tf.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

    sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0),
                                       initial_state=tuple(initial_sample_state),
                                       time_major = True,
                                       dtype=tf.float32)

    with tf.control_dependencies([tf.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
                                  [tf.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):  
      sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)

    print('\tAll done')



    ##Running the LSTM
    epochs = 30
    valid_summary = 1 # Interval you make test predictions

    n_predict_once = 50 # Number of steps you continously predict for

    train_seq_length = train_data.size # Full length of the training data

    train_mse_ot = [] # Accumulate Train losses
    test_mse_ot = [] # Accumulate Test loss
    predictions_over_time = [] # Accumulate predictions

    session = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # Used for decaying learning rate
    loss_nondecrease_count = 0
    loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate

    print('Initialized')
    average_loss = 0

    # Define data generator
    data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings)

    x_axis_seq = []

    # Points you start your test predictions from
    test_points_seq = np.arange(11000,12000,50).tolist()

    for ep in range(epochs):       

        # ========================= Training =====================================
        for step in range(train_seq_length//batch_size):

            u_data, u_labels = data_gen.unroll_batches()

            feed_dict = {}
            for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
                feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
                feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)

            feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})

            _, l = session.run([optimizer, loss], feed_dict=feed_dict)

            average_loss += l

        # ============================ Validation ==============================
        if (ep+1) % valid_summary == 0:

          average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))

          # The average loss
          if (ep+1)%valid_summary==0:
            print('Average loss at step %d: %f' % (ep+1, average_loss))

          train_mse_ot.append(average_loss)

          average_loss = 0 # reset loss

          predictions_seq = []

          mse_test_loss_seq = []

          # ===================== Updating State and Making Predicitons ========================
          for w_i in test_points_seq:
            mse_test_loss = 0.0
            our_predictions = []

            if (ep+1)-valid_summary==0:
              # Only calculate x_axis values in the first validation epoch
              x_axis=[]

            # Feed in the recent past behavior of stock prices
            # to make predictions from that point onwards
            for tr_i in range(w_i-num_unrollings+1,w_i-1):
              current_price = all_mid_data[tr_i]
              feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)    
              _ = session.run(sample_prediction,feed_dict=feed_dict)

            feed_dict = {}

            current_price = all_mid_data[w_i-1]

            feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)

            # Make predictions for this many steps
            # Each prediction uses previous prediciton as it's current input
            for pred_i in range(n_predict_once):

              pred = session.run(sample_prediction,feed_dict=feed_dict)

              our_predictions.append(np.asscalar(pred))

              feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)

              if (ep+1)-valid_summary==0:
                # Only calculate x_axis values in the first validation epoch
                x_axis.append(w_i+pred_i)

              mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2

            session.run(reset_sample_states)

            predictions_seq.append(np.array(our_predictions))

            mse_test_loss /= n_predict_once
            mse_test_loss_seq.append(mse_test_loss)

            if (ep+1)-valid_summary==0:
              x_axis_seq.append(x_axis)

          current_test_mse = np.mean(mse_test_loss_seq)

          # Learning rate decay logic
          if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
              loss_nondecrease_count += 1
          else:
              loss_nondecrease_count = 0

          if loss_nondecrease_count > loss_nondecrease_threshold :
                session.run(inc_gstep)
                loss_nondecrease_count = 0
                print('\tDecreasing learning rate by 0.5')

          test_mse_ot.append(current_test_mse)
          print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
          predictions_over_time.append(predictions_seq)
          print('\tFinished Predictions')


    ##Visualizing the Predictions
    best_prediction_epoch = 28 # replace this with the epoch that you got the best results when running the plotting code

    plt.figure(figsize = (18,18))
    plt.subplot(2,1,1)
    plt.plot(range(df.shape[0]),all_mid_data,color='b')

    # Plotting how the predictions change over time
    # Plot older predictions with low alpha and newer predictions with high alpha
    start_alpha = 0.25
    alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
    for p_i,p in enumerate(predictions_over_time[::3]):
        for xval,yval in zip(x_axis_seq,p):
            plt.plot(xval,yval,color='r',alpha=alpha[p_i])

    plt.title('Evolution of Test Predictions Over Time',fontsize=18)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Price',fontsize=18)
    plt.xlim(11000,12500)

    plt.subplot(2,1,2)

    # Predicting the best test prediction you got
    plt.plot(range(df.shape[0]),all_mid_data,color='b')
    for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
        plt.plot(xval,yval,color='r')

    plt.title('Best Test Predictions Over Time',fontsize=18)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Price',fontsize=18)
    plt.xlim(11000,12500)
    plt.show()

    class GetInfo:

        def GetPrice(time):

            value = []
            value = prices[:len(train_data)]
            value.extend(predictions_over_time[best_prediction_epoch])
            
            for i in range (len(dates)):
                if dates[i] == time:
                    result = value[i]

            return result

except Exception:
    print("Unidentified exception")
