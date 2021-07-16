import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import datetime as dt
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import RobustScaler
import os

MODEL_NAME = "aksje.model"

df = pd.read_csv("dataset.csv", delimiter=";")
df = df.drop(columns=['symbol', 'date', 'id'])

train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

f_columns = ['high', 'low', 'open', 'volume']
f_transformer = RobustScaler()
cnt_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['close']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['close'] = cnt_transformer.transform(train[['close']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['close'] = cnt_transformer.transform(test[['close']])


def create_dataset(X, y, time_steps=1, future_day=1):
    Xs, ys = [], []
    for i in range(time_steps, len(X) - future_day):
        v = X.iloc[i - time_steps:i].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + future_day])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 60
FUTURE_DAY = 10
X_train, y_train = create_dataset(train, train.close, time_steps=TIME_STEPS, future_day=FUTURE_DAY)
X_test, y_test = create_dataset(test, test.close, time_steps=TIME_STEPS, future_day=FUTURE_DAY)

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
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model_checkpoint = f"{MODEL_NAME}/checkpoint"
    history = model.fit(
        X_train, y_train,
        epochs=256,
        batch_size=2560,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=128, mode="min"),
            tf.keras.callbacks.ModelCheckpoint(
                model_checkpoint, monitor='val_loss', save_best_only=True,
                save_weights_only=True, mode='min'
            )
        ],
        shuffle=False
    )
    model.load_weights(model_checkpoint)
    model.save(MODEL_NAME)

# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.legend()
# plt.show()

y_pred = model.predict(X_test)
y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

plt.plot(y_test_inv.flatten(), marker='.', label='actual')
plt.plot(y_pred_inv.flatten(), 'r', marker='.', label='predicted')
plt.legend()
plt.show()
