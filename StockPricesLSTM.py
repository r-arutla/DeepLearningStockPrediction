#https://medium.com/@computersgeek/deep-learning-for-predicting-stock-prices-4cf95b08b23b
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#table of data
training_set = pd.read_csv('GOOG_data_train.csv')

#getting only open price values
training_set = training_set.iloc[:,1:2].values

#setting all of the values to a scale from 0 to 1
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#determining input(x) and output(y) values
x_train = training_set[0:1257]
y_train = training_set[1:1258]

#adding to the input another dimension for time
x_train = np.reshape(x_train, (1257, 1, 1))

#add arguments to lstm layer
regressor = Sequential()
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#compile all layers into one system
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fit regressor to the training dataset
regressor.fit(x_train, y_train, batch_size = 32, epochs = 200)

#make predictions and visualize
test_set = pd.read_csv('GOOG_data_test.csv')
real_stock_price = test_set.iloc[:,1:2].values
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))

#transforming test data set
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.cla()
plt.clf()
#plot the data
plt.plot(real_stock_price, color = 'red', label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = 'blue', label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#print(training_set)
