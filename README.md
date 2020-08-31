# Let's Get Rich: Using RNNs To Predict Tesla Stock Prices

## Introduction 

For this particular project, my goal is to apply Reccurent Neural Networks (RNN) on Tesla stock closing prices; more specifically, I will create 2 models which utilize different types of hidden layers in the tensorflow/keras package in python. Hopefully one of these models are able to accurately predict the stock data on the test set and allows me to become the next Wolf on Wallstreet. 


## Packages and Resources Used 
**Packages:**
  * Tensorflow/keras
  * Numpy
  * Matplotlib
  * Sklearn
  * Pandas
  * Math
  
**Data Used:** https://github.com/William0n/Tesla-Stock-Prediction/blob/master/stock%20prices%20tsla.csv </br>
**Stock Quote Source:** https://ca.finance.yahoo.com/quote/TSLA?p=TSLA
 
## Data Preprocessing 

After collecting the 3 year stock prices of Tesla (20/07/2017 to 20/07/2020), the data needed some cleaning in order to work properly in the models. The following 
changes carried out: 

  * Created a `Values` vector which contains the closing prices 
  * Splitted the data into training and test set using a 80/20 split. As this is sequential data the training set contains the first 80% of the full data
  * Training data was scaled between values 0 and 1 using `MinMaxScaler` 
  * Training and Test sets were reshaped into a 3-d array to fit the expected input shape of the RNN layer 
## Modeling 

As this was my first real attempt at using nerual networks on sequential data, I wanted to experiment with 2 slightly different layers offered in tensorflow. 
The first model which was used as a baseline model included `simpleRnn` layers, and the second model used `LSTM` layers. For each model, I included a 20% drop out layer after each RNN layer in hopes of reducing the model
overfitting the data. 

For both models, I chose to use the RMSE metricc as I felt that significant outliers needed to be penalized harder in order to properly evaluate the models' accuracy

**Model with simple RNN layers:** </br>
```
modelRnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences = True, input_shape = [None, 1]),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.SimpleRNN(20, return_sequences = True),
    tf.keras.layers.Dropout(0.2), 
    
    tf.keras.layers.SimpleRNN(1)
    ])

```

**Model with LSTM layers:**  
```
modelLSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(20, return_sequences = True, input_shape = [None, 1]),
     tf.keras.layers.Dropout(0.2),
     
    tf.keras.layers.LSTM(20, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
      
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2), 
    
    tf.keras.layers.Dense(1)
    ])
 ```
## Model Performance 
**LSTM RMSE Train:** 19.133 </br>
**LSTM RMSE Test:** 343.070 </br>

**Simple RNN RMSE Train:** 15.870 </br>
**Simple RNN RMSE Test:** 517.240

<img src="imgs/Model Loss Plot fix.png"  width = 400/>

## Conclusion 
After plotting some predictions using the 2 model, it became quite clear that perhaps the 2 created models suffered from the classic problem of over fitting to the training set; this can be seen by comparing the `yellow lines` with the `blue lines`. The simple RNN model was able to predict training set data better than the LSTM model, but it could not generalize as well as the LSTM model on the test set. As seen in the graphs below, the `simple RNN model (left graph)` had predictions which appeared rather flat, whereas, the `LSTM model (right graph)` showed predictions which had slight resemblances to the pattern found in the actual data. Although I was already expecting this result on the test set, it was quite surprising for me to see the model with simple RNN layers to outperform the LSTM model on the training set. 

Nonetheless, this was a very fun project and although my dreams of becoming the next Wolf on Wallstreet using RNNs on stock prices are looking rather dim, I did manage to learn quite a few things about LSTM/RNN modeling.


<img src="imgs/Prediction plot.png"  width = 600/>
