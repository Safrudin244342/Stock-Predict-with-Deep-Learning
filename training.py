import matplotlib.pyplot as plt
import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import pickle

plt.style.use("fivethirtyeight")

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

# create data set for training ai
trainData = scaledData[0:training_data_len, :]
xTrain = []
yTrain = []

for i in range(100, len(trainData)):
    xTrain.append(trainData[i-100:i, 0])
    yTrain.append(trainData[i, 0])

# convert data set to numpy array
xTrain, yTrain = np.array(xTrain), np.array(yTrain)

# reshape array, because LSTM need array 3D not 2D
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

# create model ai
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, batch_size=1, epochs=3)
model.save("model/AUDJPY.keras", include_optimizer=False)
