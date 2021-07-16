import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sys as system
import datetime as dt
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os

MODEL_NAME = "aksjer.model"
TICKER_ID = "ATEA.XOSL"

df = pd.read_csv("dataset.csv", delimiter=";", parse_dates=['date'])
df = df.drop(columns=['date', 'id'])
# df['date'] = (df.date - df.date.min()).dt.days

train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
# train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
train = df[df['symbol'] != TICKER_ID]
test = df[df['symbol'] == TICKER_ID].iloc[2300:]
print(train.shape, test.shape)

f_columns = ['high', 'low', 'open', 'volume']
f_transformer = RobustScaler()
cnt_transformer = RobustScaler()
l_encoder = LabelEncoder()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['close']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['close'] = cnt_transformer.transform(train[['close']])
train['symbol'] = l_encoder.fit_transform(train[['symbol']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['close'] = cnt_transformer.transform(test[['close']])
test['symbol'] = l_encoder.fit_transform(test[['symbol']])


def create_dataset(X, y, time_steps=1, future_day=1):
    Xs, ys = [], []
    for i in range(time_steps, len(X) - future_day):
        v = X.iloc[i - time_steps:i].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + future_day])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 60
FUTURE_DAY = 30
X_train, y_train = create_dataset(train, train.close, time_steps=TIME_STEPS)
X_test, y_test = create_dataset(test, test.close, time_steps=TIME_STEPS)

# [samples, time_steps, n_features]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

if os.path.exists(MODEL_NAME):
    model = keras.models.load_model(MODEL_NAME)
else:
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=128,
                input_shape=(X_train.shape[1], X_train.shape[2])
            )
        )
    )
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=256,
        validation_split=0.1,
        shuffle=False
    )
    model.save(MODEL_NAME)

# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.legend()
# plt.show()

y_pred = model.predict(X_test)
y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

plt.plot(y_test_inv.flatten(), label='actual')
plt.plot(y_pred_inv.flatten(), 'r', label='predicted')
plt.legend()
plt.show()
