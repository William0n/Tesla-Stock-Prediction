### Libraries 

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
plt.style.use('fivethirtyeight')

data = pd.read_csv('stock prices tsla.csv')

## Initial plotting of tesla closing prices 
plt.figure(figsize = (11,8))
plt.plot( data['Close'])
plt.ylabel('Closing Price')
plt.xlabel('Date')
plt.xticks(range(0, data.shape[0] , 75), data['Date'].loc[::75], rotation = 45)
plt.show()

## Extracting the closing price values from the full stock data set
close_p = data.filter(['Close'])
values = np.float32(close_p.values)

## splitting closing price values into a training and test set where training 
## set consist of 80% of the total data 

train_index = math.floor(len(values) * 0.80)
train_set = values[:train_index, :]
test_set = values[train_index:, :]

## Scaling the data 

scaler = MinMaxScaler(feature_range = (0,1))

train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)

#scaler.inverse_transform(train_set) 

## Creating x training values with the corresponding y training values 
x_train =[]
y_train=[]

for i in range(30 , len(train_set) - 1):
    x_train.append(train_set[(i - 30):i, 0])
    y_train.append(train_set[i + 1, 0 ])
    
## creating x test and y test values 

x_test =[]
y_test=[]

for i in range(30 , len(test_set) - 1):
    x_test.append(test_set[(i - 30):i, 0])
    y_test.append(test_set[i + 1, 0 ])


## Reshaping TRAINING set into numpy array 
x_train = np.array(x_train, dtype = 'float32')
x_train = np.reshape(x_train, (len(x_train), 30 , 1))

y_train = np.array(y_train, dtype= 'float32')
y_train = np.reshape(y_train, (len(y_train), 1))

## Reshaping TEST set into numpy array 
x_test = np.array(x_test, dtype = 'float32')
x_test = np.reshape(x_test, (len(x_test), 30 , 1))

y_test = np.array(y_test, dtype= 'float32')
y_test = np.reshape(y_test, (len(y_test), 1))

## Simple RNN Layer Model 

modelRnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences = True, input_shape = [None, 1]),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.SimpleRNN(20, return_sequences = True),
    tf.keras.layers.Dropout(0.2), 
    
    tf.keras.layers.SimpleRNN(1)
    ])

modelRnn.compile(loss = "MSE",
              optimizer = 'adam',
              metrics = ['MSE'])

history_rnn = modelRnn.fit(x_train, y_train, epochs = 50, batch_size = 50)
results_rnn = modelRnn.evaluate(x_test, y_test)

### LSTM Model

modelLSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(20, return_sequences = True, input_shape = [None, 1]),
     tf.keras.layers.Dropout(0.2),
     
    tf.keras.layers.LSTM(20, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
      
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2), 
    
    tf.keras.layers.Dense(1)
    ])

modelLSTM.compile(loss = "MSE",
              optimizer = "adam",
              metrics = ['MSE'])

history_lstm = modelLSTM.fit(x_train, y_train, epochs = 50, batch_size = 50)
results_lstm = modelLSTM.evaluate(x_test, y_test)


### Using the models to predict x_train and x_test values
train_predRnn = modelRnn.predict(x_train)
train_predRnn = scaler.inverse_transform(train_predRnn)
test_predRnn = modelRnn.predict(x_test)
test_predRnn = scaler.inverse_transform(test_predRnn)

## Prediction for LSTM model
train_predlstm = modelLSTM.predict(x_train)
train_predlstm = scaler.inverse_transform(train_predlstm)
test_predlstm = modelLSTM.predict(x_test)
test_predlstm = scaler.inverse_transform(test_predlstm)

y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

## RMSE values 
rmse_trainR = math.sqrt(mean_squared_error( y_train, train_predRnn))
rmse_test_R = math.sqrt(mean_squared_error( y_test, test_predRnn))

rmse_trainL = math.sqrt(mean_squared_error( y_train, train_predlstm))
rmse_test_L = math.sqrt(mean_squared_error( y_test, test_predlstm))

### Graphs of the losses on training set 
plt.plot(history_rnn.history['loss'])
plt.plot(history_lstm.history['loss'])

plt.title('Model Training Accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['RNN', 'LSTM'])


### Graphs comparing predicted values vs actual data (blue)
plt.figure(figsize = (14,8))
plt.subplot(1,2,1)


plt.plot(data['Close'])
plt.plot(range(30, 603, 1) ,train_predRnn)
plt.plot(range(635, 755, 1),test_predRnn)
plt.legend(['Real','training', 'test'])
plt.ylabel('Closing Price (USD)')
plt.xlabel('Date (in days)')


plt.subplot(1,2,2)
plt.plot(data['Close'])
plt.plot(range(30, 603, 1) ,train_predlstm)
plt.plot(range(635, 755, 1),test_predlstm)
plt.xlabel('Date (in days)')

plt.subplots_adjust(wspace = 0.25)
plt.show()

