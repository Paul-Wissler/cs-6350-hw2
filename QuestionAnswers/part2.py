import pandas as pd
from pathlib import Path

import EnsembleLearning as ensl
import LinearRegression as lr


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
    model = ensl.AdaBoostModel(X, y, sample_rate=10, boost_rounds=100, max_tree_depth=1)
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
    results = pd.Series()
    for t in range(1, 501):
        model = ensl.BaggerModel(X.copy(), y.copy(), sample_rate=50, bag_rounds=t)
        # model.evaluate(X_test)
        print(model.test(X_test.copy(), y_test.copy()))
        results = results.append(pd.Series(model.test(X_test.copy(), y_test.copy()))).reset_index(drop=True)
    results.to_csv('q2b_results.csv')


def q2d():
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
    for n in [2, 4, 6]:
        model = ensl.RandomForestModel(X.copy(), y.copy(), sample_rate=50, bag_rounds=20, num_sample_attributes=n)
        # model.evaluate(X_test.copy())
        print(model.test(X_test.copy(), y_test.copy()))


def q4a():
    x_cols = [
        'Cement',
        'Slag',
        'FlyAsh',
        'Water',
        'SP',
        'CoarseAggr',
        'FineAggr',
    ]
    cols = [
        'Cement',
        'Slag',
        'FlyAsh',
        'Water',
        'SP',
        'CoarseAggr',
        'FineAggr',
        'Output',
    ]

    c_data = pd.read_csv(
        Path('concrete', 'concrete', 'train.csv'), 
        names=cols, 
        index_col=False,
    )
    test_c_data = pd.read_csv(
            Path('concrete', 'concrete', 'test.csv'), 
            names=cols, 
            index_col=False,
    )

    # print(c_data.head())

    model = lr.BatchGradientDescentModel(c_data[x_cols], c_data.Output, max_rounds=5000, rate=0.01)
    cost = model.compute_cost(test_c_data[x_cols], test_c_data.Output, model.weights)
    print(cost)
    print(model.convergence_of_weights)
    print(model.weights)


def q4b():
    x_cols = [
        'Cement',
        'Slag',
        'FlyAsh',
        'Water',
        'SP',
        'CoarseAggr',
        'FineAggr',
    ]
    cols = [
        'Cement',
        'Slag',
        'FlyAsh',
        'Water',
        'SP',
        'CoarseAggr',
        'FineAggr',
        'Output',
    ]

    c_data = pd.read_csv(
        Path('concrete', 'concrete', 'train.csv'), 
        names=cols, 
        index_col=False,
    )
    test_c_data = pd.read_csv(
            Path('concrete', 'concrete', 'test.csv'), 
            names=cols, 
            index_col=False,
    )

    # print(c_data.head())

    model = lr.StochasticGradientDescentModel(c_data[x_cols], c_data.Output, max_rounds=10000, rate=0.001)
    cost = model.compute_cost(test_c_data[x_cols], test_c_data.Output, model.weights)
    print(cost)
    print(model.convergence_of_weights)
    print(model.weights)
