# Let's Get Rich: Using RNNs To Predict Tesla Stock Prices

## Introduction 

What better way to get rich than learn
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
## Results
**LSTM RMSE Train:** 19.133 </br>
**LSTM RMSE Test:** 343.070 </br>

**Simple RNN RMSE Train:** 15.870 </br>
**Simple RNN RMSE Test:** 517.240

<img src="imgs/Model Loss Plot fix.png"  width = 500/>

<img src="imgs/Prediction plot.png"  width = 500/>
