import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
from collections.abc import Iterable
from keras.backend import mean as tf_mean, sum as tf_sum
from typing import Union


def load_data(file_path: str):
    stock_df = pd.read_csv(file_path)
    stock_df.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis=1, inplace=True)
    stock_df.set_index("Date", inplace=True)
    return stock_df


def plot_time_series(*time_series: np.array):
    for ts in time_series:
        pyplot.plot(ts)
    pyplot.show()


def arrange_features_and_labels(time_series, n_time_steps=40, lag=1):
    X_train = []
    y_train = []
    lag -= 1
    for i in range(n_time_steps, len(time_series) - lag):
        X_train.append(time_series[i - n_time_steps:i, :])
        y_train.append(time_series[i + lag, :])
    return np.array(X_train), np.array(y_train)


def build_model(input_shape: tuple):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


if __name__ == '__main__':
    stock_price = load_data("data/TSLA.csv")
    stock_price = stock_price.to_numpy()
    # plot_time_series(stock_price)

    lag = 3
    test_size = 50 + lag
    time_steps = 50
    scaler = MinMaxScaler()
    stock_price_scaled = scaler.fit_transform(stock_price)
    train_set = stock_price_scaled[:-test_size]
    test_set = stock_price_scaled[-test_size:]

    x_train, y_train = arrange_features_and_labels(train_set, n_time_steps=time_steps, lag=lag)
    model = build_model(x_train.shape[1:])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.0001)
    model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])

    x_test, y_test = arrange_features_and_labels(test_set, n_time_steps=time_steps, lag=lag)
    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    plot_time_series(y_test, y_pred)

    print(f"test score with {lag} lag: {tf.math.sqrt(model.evaluate(x_test, y_test))}")