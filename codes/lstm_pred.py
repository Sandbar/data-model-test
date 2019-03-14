

from sklearn.metrics import mean_absolute_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np


def build_model(layers):  #layers [1,72,100,1]
    model = Sequential()
    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))
    model.compile(loss="mae", optimizer="rmsprop")
    return model


#直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:' ,np.array(predicted).shape)  #(412L,1L)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def lstm_evaluation(ytest, ypred):
    print('eva...')
    mae = mean_absolute_error(ytest, ypred)
    print('mae: ', mae, ' mape: ', mae * len(ytest) / sum(ytest))
    return mae * len(ytest) / sum(ytest)



