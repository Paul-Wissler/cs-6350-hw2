import pandas as pd
from pathlib import Path

import EnsembleLearning as ensl


def q2a():
    x_cols = [
        'age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome',
    ]
    cols = [
        'age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome',
        'y',
    ]
    bank_data = pd.read_csv(
        Path('bank', 'train.csv'), 
        names=cols, 
        index_col=False,
    )
    test_bank_data = pd.read_csv(
            Path('bank', 'test.csv'), 
            names=cols, 
            index_col=False,
    )
    X = bank_data[x_cols]
    y = bank_data.y
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X[numeric_cols] = X[numeric_cols].astype(int)
    X_test = bank_data[x_cols]
    y_test = bank_data.y
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_test[numeric_cols] = X_test[numeric_cols].astype(int)
    model = ensl.AdaBoostModel(X, y, sample_rate=10, boost_rounds=10)
    # model.evaluate(X_test)
    # print(model.test(X_test, y_test))


def q2b():
    x_cols = [
        'age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome',
    ]
    cols = [
        'age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome',
        'y',
    ]
    bank_data = pd.read_csv(
        Path('bank', 'train.csv'), 
        names=cols, 
        index_col=False,
    )
    test_bank_data = pd.read_csv(
            Path('bank', 'test.csv'), 
            names=cols, 
            index_col=False,
    )

    X = bank_data[x_cols]
    y = bank_data.y
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X[numeric_cols] = X[numeric_cols].astype(int)
    X_test = bank_data[x_cols]
    y_test = bank_data.y
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_test[numeric_cols] = X_test[numeric_cols].astype(int)
    model = ensl.BaggerModel(X, y, sample_rate=10, bag_rounds=10)
    model.evaluate(X_test)
    print(model.test(X_test, y_test))
