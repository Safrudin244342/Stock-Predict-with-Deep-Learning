import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import pickle

# load model
model = load_model("model/AUDJPY.keras")

# get stock price
with open("dataframe/dataframeAUDJPY=X.pickel", mode="rb") as file:
    df = pickle.load(file)

# make new dataframe
data = df.filter(['Close'])

# convert to numpy array
dataset = data.values

# get number for training the model
training_data_len = math.ceil(len(dataset) * .8)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(dataset)

# create data set for test ai
dataTest = scaledData[training_data_len - 100: , :]
xTest = []
yTest = dataset[training_data_len:, :]
for i in range(100, len(dataTest)):
    xTest.append(dataTest[i-100:i, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

# predict stock price
predicts = model.predict(xTest)
predicts = scaler.inverse_transform(predicts)

# get the root mean squared error (RMSE)
RMSE = np.sqrt(np.mean(predicts - yTest) **2)

# show data in graphic
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predicts

plt.figure(figsize=(16, 8))
plt.title("Model")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Stock Price USD $", fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
