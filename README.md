# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Develop a Recurrent Neural Network model for stock price prediction using the stock prediction dataset




## DESIGN STEPS

### STEP 1:
Import the necessary tensorflow modules 

### STEP 2:
load the stock dataset

### STEP 3:
Fit the model and train the data
### Step 4:
predict the values

## PROGRAM


```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
import tensorflow as tf

dataset_train = pd.read_csv('rnn-stock-price-prediction/trainset.csv')


print(dataset_train.columns)

print(dataset_train.head())

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

print(train_set.shape)

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

print(training_set_scaled.shape)

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))


print(X_train.shape)

length = 60
n_features = 1

model = Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(50,input_shape=(60,1), activation = 'relu')),
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse')


model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('rnn-stock-price-prediction/testset.csv')

print(dataset_total.shape)

test_set = dataset_test.iloc[:,1:2].values

print(test_set.shape)

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
print(dataset_test.shape)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

print(X_test.shape)

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


m=tf.keras.losses.MeanSquaredError()

m(dataset_test["Close"],predicted_stock_price)
```
## OUTPUT

### True Stock Price, Predicted Stock Price vs time

Include your plot here

![image](https://user-images.githubusercontent.com/75235209/194799077-0bb3cb31-ba7b-4da6-8204-0e8258cfdefd.png)

Include the mean square error
![image](https://user-images.githubusercontent.com/75235209/194799123-372a5b0f-936e-4f22-a6bf-5e25d3414168.png)

## RESULT
Thus a Recurrent Neural Network model for stock price prediction is created and executed successfully.
