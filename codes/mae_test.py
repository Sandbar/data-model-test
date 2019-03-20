
import pandas as pd
import xgboost_pred
import numpy as np


def xgb_predict(train, test):
    xgb = xgboost_pred.xgb_train(train.iloc[:, 1:], train.iloc[:, 0])
    ypred = xgboost_pred.xgb_pred(test.iloc[:, 1:], xgb)
    mape = xgboost_pred.evaluate(test.iloc[:, 0], ypred)
    print(mape)


data = pd.read_csv('tmp_data.csv')

''' 原始的72维数据，即预测的前三天的每小时的数据 '''
sp_size = len(data) - 31 * 24
train, test = data.iloc[73:sp_size, :], data.iloc[sp_size:, :]
print('before:', data.shape, train.shape, test.shape)
xgb_predict(train, test)

# ''' 加入前后三天的差值，即前三天的每个小数数据和相邻小时差的数据 '''
names = data.columns
for index in range(1, len(names)-1):
    data[names[index]+'_'+names[index+1]] = data[names[index]] - data[names[index+1]]

sp_size = len(data) - 31 * 24
train, test = data.iloc[73:sp_size, :], data.iloc[sp_size:, :]
print('after:', data.shape, train.shape, test.shape)
xgb_predict(train, test)


''' 相邻两个小时差的数据,前三天的每小时数据做一个log'''
def logx(dt):
    if dt == 0:
        return 0
    elif dt < 0:
        return - np.log(-dt)
    else:
        return np.log(dt)
sp_size = len(data) - 31 * 24

for name in names:
    if name != '0' and '_' not in name:
        data[name+'_log'] = data[name].apply(logx)
names = [name for name in data.columns if '_' in name or name=='0']
print(names)
train, test = data.iloc[73:sp_size, :][names], data.iloc[sp_size:, :][names]
print('only:', data.shape, train.shape, test.shape)
xgb_predict(train, test)



