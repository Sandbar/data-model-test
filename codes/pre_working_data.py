

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost_pred
# import lstm_pred as lstm
import warnings


def load_data():
    data = pd.read_csv('C.test_data201803.csv')
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data.sort(columns=['time'], inplace=True, ascending=True)
    print(data.head())
    print(max(data['time']), min(data['time']))
    print(max(data['KWH']), min(data['KWH']))
    print(len(data))
    print(data.dtypes)
    return data


def plt_origin_data(data):
    plt.plot(data['time'], data['KWH'], 'r-')
    plt.ylabel('KWH')
    plt.xlabel('time')
    plt.show()


def barplot_data(data):
    plt.boxplot(data['KWH'])
    plt.show()


def pie_data(data):
    count_data = [0, 0]
    plt.subplot(211)
    for index in range(len(data)):
        row = data.iloc[index]
        if row['KWH'] == 0:
            count_data[0] += 1
        else:
            count_data[1] += 1
    print(count_data)
    plt.pie(count_data, labels=['KWH=0', 'KWH>0'], colors=['red', 'blue'], autopct='%3.1f%%')

    count_data = [0, 0]
    plt.subplot(212)
    for index in range(len(data)):
        row = data.iloc[index]
        if row['KWH'] <= 2500:
            count_data[0] += 1
        else:
            count_data[1] += 1
    print(count_data)
    plt.pie(count_data, labels=['KWH<2500', 'KWH≥2500'], colors=['red', 'blue'], autopct='%3.1f%%')
    plt.show()


def calc_month_data(data):
    data['month'] = data['time'].apply(lambda x: x.month)
    month = [0 for _ in range(12)]
    plt.subplot(211)
    for index in range(len(data)):
        month[data['month'][index]-1] += data['KWH'][index]
    plt.ylabel('month total KWH')
    plt.xlabel('month')
    plt.plot(np.arange(len(month)), month, 'r-')

    plt.subplot(212)
    min_date = min(data['time'])
    data['month_day'] = data['time'].apply(lambda x: (x-min_date).days)
    month_day = [0 for _ in range(max(data['month_day']))]
    for index in range(len(data)):
        month_day[data['month_day'][index]-1] += data['KWH'][index]
    plt.ylabel('day total KWH')
    plt.xlabel('day')
    plt.plot(np.arange(len(month_day)), month_day, 'b-')
    plt.show()


def pre_working(data):
    pre_data = dict()
    for index in range(len(data)):
        pre_data[data.iloc[index]['id']] = [0 for _ in range(73)]
    for key in pre_data.keys():
        tdata = data[(0 <= key - data['id']) & (key - data['id'] < 73)]
        for index in range(len(tdata)):
            row = tdata.iloc[index]
            pre_data[key][key - row['id']] += row['KWH']

    df_data = pd.DataFrame.from_dict(pre_data, orient='index')
    sp_size = len(df_data) - 31 * 24
    train, test = df_data.iloc[73:sp_size, :], df_data.iloc[sp_size:, :]
    print(df_data.shape, train.shape, test.shape)
    xgb_predict(train, test)
    return df_data


def xgb_predict(train, test):
    xgb = xgboost_pred.xgb_train(train.iloc[:, 1:], train.iloc[:, 0])
    ypred = xgboost_pred.xgb_pred(test.iloc[:, 1:], xgb)
    mape = xgboost_pred.evaluate(test.iloc[:, 0], ypred)
    print(mape)


# def lstm_predict(train, test):
#     epochs = 1
#     model = lstm.build_model([1, 72, 100, 1])
#     x_train = np.reshape(train, (train.iloc[:, 1:].shape[0], train.iloc[:, 1:].shape[1], 1))
#     x_test = np.reshape(test, (test.iloc[:, 1:].shape[0], test.iloc[:, 1:].shape[1], 1))
#     model.fit(x_train, train.iloc[:, 0], batch_size=512, nb_epoch=epochs, validation_split=0.05)
#     ypred = lstm.predict_point_by_point(model, x_test)
#     mape = lstm.lstm_evaluation(test.iloc[:, 0], ypred)
#     print(mape)

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    data = load_data()
    plt_origin_data(data)
    barplot_data(data)
    pie_data(data)
    calc_month_data(data)
    df_data = pre_working(data)

    df_data = pd.read_csv('tmp_data.csv')
    sp_size = len(df_data) - 31 * 24
    train, test = df_data.iloc[73:sp_size, :], df_data.iloc[sp_size:, :]
    print(df_data.shape, train.shape, test.shape)
    xgb_predict(train, test)
    lstm_predict(train, test)

