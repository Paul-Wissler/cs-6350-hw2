import pandas as pd
import numpy as np
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
    model = ensl.AdaBoostModel(X, y, sample_rate=10, boost_rounds=30, max_tree_depth=1)
    # model.evaluate(X_test)
    print(model.test(X_test, y_test))


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

    model = lr.BatchGradientDescentModel(
        c_data[x_cols], c_data.Output, max_rounds=10000, rate=0.01
    )
    cost = model.compute_cost(test_c_data[x_cols], test_c_data.Output, model.weights)
    
    # COST: 41.10059112261912

    # CONVERGENCE OF WEIGHTS
    # 0       6.001037e+01
    # 1       2.626977e-01
    # 2       1.334676e-01
    # 3       8.815422e-02
    # 4       6.514645e-02
    #             ...
    # 7090    1.003669e-06
    # 7091    1.002539e-06
    # 7092    1.001410e-06
    # 7093    1.000282e-06
    # 7094    9.991560e-07
    # Length: 7095, dtype: float64

    # COST OF EACH STEP
    # 0       39.120273
    # 1       30.906258
    # 2       27.101767
    # 3       24.713854
    # 4       23.352686
    #           ...
    # 7089    18.943672
    # 7090    18.943671
    # 7091    18.943671
    # 7092    18.943671
    # 7093    18.943670
    # Length: 7094, dtype: float64

    # WEIGHT VECTOR
    # Cement        0.900225
    # Slag          0.785943
    # FlyAsh        0.850665
    # Water         1.298623
    # SP            0.129834
    # CoarseAggr    1.571793
    # FineAggr      0.998347
    # MODEL_BIAS   -0.015204
    # Name: weights, dtype: float64

    print(f'COST: {cost}')
    print('\nCONVERGENCE OF WEIGHTS')
    print(model.convergence_of_weights)
    model.convergence_of_weights.to_csv('q4a_convergence_of_weights.csv')
    print('\nCOST OF EACH STEP')
    print(model.cost_of_each_step)
    model.cost_of_each_step.to_csv('q4a_cost_of_each_step.csv')
    print('\nWEIGHT VECTOR')
    print(model.weights)
    model.weights.to_csv('q4a_weights.csv')


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

    model = lr.StochasticGradientDescentModel(
        c_data[x_cols], c_data.Output, rate=0.0005, max_rounds=10000
        # c_data[x_cols], c_data.Output, rate=0.01, max_rounds=5000
    )
    cost = model.compute_cost(test_c_data[x_cols], test_c_data.Output, model.weights)
    
    print(f'COST: {cost}')
    print('\nCONVERGENCE OF WEIGHTS')
    print(model.convergence_of_weights)
    model.convergence_of_weights.to_csv('q4b_convergence_of_weights.csv')
    print('\nCOST OF EACH STEP')
    print(model.cost_of_each_step)
    model.cost_of_each_step.to_csv('q4b_cost_of_each_step.csv')
    print('\nWEIGHT VECTOR')
    print(model.weights)
    model.weights.to_csv('q4b_weights.csv')

    # COST: 41.04846572489124

    # CONVERGENCE OF WEIGHTS
    # 0       6.001037e+01
    # 1       1.491483e-04
    # 2       1.481438e-03
    # 3       6.661488e-04
    # 4       9.569676e-04
    #             ...
    # 2553    2.338325e-03
    # 2554    2.367669e-04
    # 2555    7.226300e-04
    # 2556    3.196948e-04
    # 2557    8.244933e-07
    # Length: 2558, dtype: float64

    # COST OF EACH STEP
    # 0       60.003233
    # 1       59.869479
    # 2       59.846828
    # 3       59.797806
    # 4       59.666681
    #           ...
    # 2552    31.310176
    # 2553    31.304040
    # 2554    31.311249
    # 2555    31.308078
    # 2556    31.308107
    # Length: 2557, dtype: float64

    # WEIGHT VECTOR
    # Cement        0.030679
    # Slag         -0.130154
    # FlyAsh       -0.127862
    # Water         0.262171
    # SP           -0.024053
    # CoarseAggr    0.058613
    # FineAggr     -0.018950
    # MODEL_BIAS   -0.015117
    # Name: weights, dtype: float64


def q4c():
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
    # test_c_data = pd.read_csv(
    #         Path('concrete', 'concrete', 'test.csv'), 
    #         names=cols, 
    #         index_col=False,
    # )
    ow = calc_optimal_weights(c_data[x_cols], c_data.Output)
    print(ow)

    # Cement        0.921549
    # Slag          0.808294
    # FlyAsh        0.873974
    # Water         1.314288
    # SP            0.133924
    # CoarseAggr    1.599047
    # FineAggr      1.020292
    # dtype: float64


def calc_optimal_weights(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    nX = X.to_numpy().T # Has to be transposed for lin. alg. to work
    ny = y.to_numpy()
    x_xt = np.matmul(nX, np.transpose(nX))
    x_xt_inv = np.linalg.inv(x_xt)
    x_xt_inv_x = np.matmul(x_xt_inv, nX)
    x_xt_inv_x_y = np.dot(x_xt_inv_x, ny)
    return pd.Series(x_xt_inv_x_y, index=X.columns)
