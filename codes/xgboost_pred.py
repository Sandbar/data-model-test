

import xgboost as xgb
from sklearn.metrics import mean_absolute_error


def xgb_train(xtrain, ytrain):
    print('train...')
    param = {
        'max_depth': 5,
        'learning_rate': 0.05,
        'objective': 'reg:linear',
        'nthread': 8,
        'eval_metric': 'mae'
    }
    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    evallist = [(dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_boost_round=1500, evals=evallist, early_stopping_rounds=50)
    print('fit...')
    return bst


def xgb_pred(xtest, bst):
    print('pred...')
    test = xgb.DMatrix(xtest)
    ypred = bst.predict(test)
    return ypred


def evaluate(ytest, ypred):
    print('eva...')
    mae = mean_absolute_error(ytest, ypred)
    print('mae: ', mae, ' mape: ', mae*len(ytest)/sum(ytest))
    print(sum(ytest))
    return mae*len(ytest)/sum(ytest)
