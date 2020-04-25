import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

# Getting the DataFrame with the stock quote
df = web.DataReader("NFLX", data_source="yahoo", start="2012-01-03", end="2020-04-01")

# This print shows the head and the tail of the DataFrame created
# print(df)

# This print shows the number of rows and columns in the data set
# print(df.shape)

# Plots the closing price history
# plt.figure(figsize=(16,8))
# plt.title("Close Price History - Netflix")
# plt.plot(df["Close"])
# plt.xlabel("Date", fontsize=18)
# plt.ylabel("Close Price USD ($)", fontsize=18)
# plt.show()

# A new DataFrame with only the 'Close' column
data = df.filter(["Close"])

# Converting to numpy
dataset = data.values

# Getting the number of rows to train our model, setting the size of our training model to 80%
# and rounding up the values
training_data_len = math.ceil(len(dataset) * 0.8)

# Scaling the data to use in the neural network and help the model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
# print(scaled_data)

# Crating the Training DataSet and the scaled Training DataSet
train_data = scaled_data[0:training_data_len, :]

# Splitting the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Converting yo numpy
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping the data (from 2D to 3D)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compiling the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Training the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Creating the testing DataSet and a new array containing the scaled values
test_data = scaled_data[training_data_len - 60:, :]

# Creating the DataSets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# Convert to numpy
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Getting the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Plotting the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions

plt.figure(figsize=(16, 8))
plt.title("Netflix Close Prices")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Prices USD ($)", fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Val", "Predictions"], loc="lower right")
plt.show()
