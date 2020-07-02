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
df = web.DataReader("AUDJPY=X", data_source="yahoo", start="2000-01-01", end="2020-06-30")
with open("dataframe/dataframeAUDJPY=X.pickel", mode="wb") as file:
    pickle.dump(df, file)

print(df)
