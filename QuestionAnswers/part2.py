import json

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import EnsembleLearning as ensl
import LinearRegression as lr
import DecisionTree as dtree


def load_bank_data(t='train.csv') -> pd.DataFrame:
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
    return pd.read_csv(
        Path('bank', t), 
        names=cols, 
        index_col=False,
    )


def format_bank_data(bank_data: pd.DataFrame) -> (pd.DataFrame, pd.Series):
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
    X = bank_data[x_cols]
    y = bank_data.y
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X[numeric_cols] = X[numeric_cols].astype(int)
    return X, y


def make_q2a_model():
    X, y = format_bank_data(load_bank_data('train.csv'))
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))
    
    model = ensl.AdaBoostModel(X, y, 
        sample_rate=100, 
        boost_rounds=500,
        max_tree_depth=1
    )
    return model, X, y, X_test, y_test


def q2a():
    model, X, y, X_test, y_test = make_q2a_model()

    train_results, cum_train_results = model.test_cumulative_trees(X.copy(), y.copy())
    print(train_results)
    print(cum_train_results)
    train_results.to_csv('q2a_train_results.csv', index=False)
    cum_train_results.to_csv('q2a_cum_train_results.csv', index=False)
    
    test_results, cum_test_results = model.test_cumulative_trees(X_test.copy(), y_test.copy())
    print(test_results)
    print(cum_test_results)
    test_results.to_csv('q2a_test_results.csv', index=False)
    cum_test_results.to_csv('q2a_cum_test_results.csv', index=False)


def q2a_final():
    cum_results = pd.DataFrame()
    results = pd.DataFrame()
    cum_results['test'] = 1 - pd.read_csv('TEST_q2a_test_cum_results.csv')['0']
    results['test'] = 1 - pd.read_csv('TEST_q2a_test_single_results.csv')['0']
    cum_results['train'] = 1 - pd.read_csv('TEST_q2a_train_cum_results.csv')['0']
    results['train'] = 1 - pd.read_csv('TEST_q2a_train_single_results.csv')['0']
    
    # # plt.figure(0)
    # fig = cum_results.plot()
    # plt.ylabel('Error')
    # plt.xlabel('Round')
    # plt.savefig('q2a_cum_results.png')
    
    # # plt.figure(1)
    # fig = results.plot()
    # plt.ylabel('Error')
    # plt.xlabel('Round')
    # plt.savefig('q2a_single_results.png')

    print(len(cum_results.index.to_list()), len(cum_results['train'].to_list()))

    fig, axs = plt.subplots(2)
    axs[0].plot(cum_results.index.to_list(), cum_results['train'].to_list())
    axs[0].set_title('train')
    axs[1].plot(cum_results.index.to_list(), cum_results['test'].to_list())
    axs[1].set_title('test')
    for ax in axs.flat:
        ax.set(xlabel='Round', ylabel='Error')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig('q2a_cum_results.png')
    
    fig, axs = plt.subplots(2)
    axs[0].plot(results.index.to_list(), results['train'].to_list())
    axs[0].set_title('train')
    axs[1].plot(results.index.to_list(), results['test'].to_list())
    axs[1].set_title('test')
    for ax in axs.flat:
        ax.set(xlabel='Round', ylabel='Error')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig('q2a_single_results.png')

    # plt.show()


def make_q2b_model():
    X, y = format_bank_data(load_bank_data('train.csv'))
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))
    model = ensl.BaggerModel(X.copy(), y.copy(), sample_rate=100, bag_rounds=500)
    return model, X, y, X_test, y_test


def q2b():
    model = make_q2b_model()
    ix = [
        1,
        5,
        10,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        500,
    ]

    train_results = model.test_cumulative_trees(X.copy(), y.copy(), ix)
    test_results = model.test_cumulative_trees(X_test.copy(), y_test.copy(), ix)
    
    print(train_results)
    print(test_results)
    train_results.to_csv('q2b_train_results.csv')
    test_results.to_csv('q2b_test_results.csv')


def q2b_final():
    results = pd.DataFrame()
    results['test'] = 1 - pd.read_csv('q2b_test_results.csv').set_index(['Unnamed: 0'])['0']
    results['train'] = 1 - pd.read_csv('q2b_train_results.csv').set_index(['Unnamed: 0'])['0']
    
    fig = results.plot()
    plt.ylabel('Error')
    plt.xlabel('Trees')

    plt.savefig('q2b_plot.png')

    # plt.show()


def make_q2c_model():
    X, y = format_bank_data(load_bank_data('train.csv'))
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))
    model = ensl.BaggerModel(X.copy(), y.copy(), sample_rate=100, bag_rounds=500)
    return model, X, y, X_test, y_test


def q2c():
    X, y = format_bank_data(load_bank_data('train.csv'))
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))

    bagged_tree_results = pd.DataFrame()
    single_tree_results = pd.DataFrame()

    for i in range(100):
        print(i)
        sample_X = X.sample(1000)
        sample_y = y[sample_X.index]
        sample_X = sample_X.reset_index(drop=True)
        sample_y = sample_y.reset_index(drop=True)
        model = ensl.BaggerModel(sample_X, sample_y, sample_rate=100, bag_rounds=500)
        bagged_tree_results[i] = pd.Series(model.evaluate(X_test))
        single_tree_results[i] = pd.Series(model.model[0].evaluate(X_test))

    bagged_tree_results.replace(to_replace='no', value=0, inplace=True)
    single_tree_results.replace(to_replace='no', value=0, inplace=True)
    bagged_tree_results.replace(to_replace='yes', value=1, inplace=True)
    single_tree_results.replace(to_replace='yes', value=1, inplace=True)
    bagged_tree_results.to_csv('q2c_bagged_tree_results.csv', index=False)
    single_tree_results.to_csv('q2c_single_tree_results.csv', index=False)


def q2c_final():
    X, y = format_bank_data(load_bank_data('train.csv'))
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))

    bag_df = pd.read_csv('TEST_q2c_bagged_tree_results.csv')
    stree_df = pd.read_csv('TEST_q2c_single_tree_results.csv')

    y_test.replace(to_replace='no', value=0, inplace=True)
    y_test.replace(to_replace='yes', value=1, inplace=True)

    print(f'single tree general squared error: {calc_general_squared_error(stree_df.copy(), y_test.copy())}')
    print(f'bagged tree general squared error: {calc_general_squared_error(bag_df.copy(), y_test.copy())}')

    # Bias: 0.10603268
    # Sample Variance: 0.04618517171717173
    # single tree general squared error: 0.13951096484909092
    # Bias: 0.1246864
    # Sample Variance: 0.0001187878787878788
    # bagged tree general squared error: 0.12479828357163637


def calc_bias(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    # bias = [f(X_test) - MEAN(h(X_test))]^2
    E_h_X = df.mean(axis=1)
    return np.square(y - E_h_X)


def calc_sample_var(df: pd.DataFrame) -> pd.Series:
    # s^2 = 1/(n-1) * sum[(x_i - m)^2]
    m = df.mean(axis=1)
    for col in df.columns:
        df[col] = np.square(df[col] - m)
    return 1 / (len(df.columns) - 1) * df.sum(axis=1)


def calc_general_squared_error(df: pd.DataFrame, y: pd.Series) -> float:
    # general squared error = MEAN(bias) - MEAN(variance)
    print(f'Bias: {calc_bias(df, y).mean()}')
    print(f'Sample Variance: {calc_sample_var(df).mean()}')
    return calc_bias(df, y).mean() + calc_sample_var(df).mean()


def q2d():
    X, y = format_bank_data(load_bank_data('train.csv'))
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))

    for n in [2, 4, 6]:
        model = ensl.RandomForestModel(
            X.copy(), y.copy(), 
            sample_rate=10, 
            bag_rounds = 1,
            # bag_rounds=500, 
            num_sample_attributes=n,
            reproducible_seed=False,
        )
        
        ix = [
            1,
            5,
            10,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
        ]
        print(n, model.test(X.copy(), y.copy()))
        print(n, model.test(X_test.copy(), y_test.copy()))
        # train_results = model.test_cumulative_trees(X.copy(), y.copy(), ix)
        # test_results = model.test_cumulative_trees(X_test.copy(), y_test.copy(), ix)
        # train_results.to_csv(f'q2d_train_results_attr_{n}.csv')
        # test_results.to_csv(f'q2d_test_results_attr_{n}.csv')


def q2d_final():
    results = pd.DataFrame()
    print(pd.read_csv('q2d_test_results_attr_2.csv').set_index(['Unnamed: 0']))
    results['test (2 attr.)'] = 1 - pd.read_csv('TEST_q2d_test_results_attr_2.csv').set_index(['Unnamed: 0'])['0']
    results['train (2 attr.)'] = 1 - pd.read_csv('TEST_q2d_train_results_attr_2.csv').set_index(['Unnamed: 0'])['0']
    
    results['test (4 attr.)'] = 1 - pd.read_csv('TEST_q2d_test_results_attr_4.csv').set_index(['Unnamed: 0'])['0']
    results['train (4 attr.)'] = 1 - pd.read_csv('TEST_q2d_train_results_attr_4.csv').set_index(['Unnamed: 0'])['0']
    
    results['test (6 attr.)'] = 1 - pd.read_csv('TEST_q2d_test_results_attr_6.csv').set_index(['Unnamed: 0'])['0']
    results['train (6 attr.)'] = 1 - pd.read_csv('TEST_q2d_train_results_attr_6.csv').set_index(['Unnamed: 0'])['0']
    
    fig, axs = plt.subplots(2 ,3)
    axs[0, 0].plot(results.index, results['train (2 attr.)'])
    axs[0, 0].set_title('train (2 attr.)')
    axs[0, 1].plot(results.index, results['train (4 attr.)'])
    axs[0, 1].set_title('train (4 attr.)')
    axs[0, 2].plot(results.index, results['train (6 attr.)'])
    axs[0, 2].set_title('train (6 attr.)')
    axs[1, 0].plot(results.index, results['test (2 attr.)'])
    axs[1, 0].set_title('test (2 attr.)')
    axs[1, 1].plot(results.index, results['test (4 attr.)'])
    axs[1, 1].set_title('test (4 attr.)')
    axs[1, 2].plot(results.index, results['test (6 attr.)'])
    axs[1, 2].set_title('test (6 attr.)')

    for ax in axs.flat:
        ax.set(xlabel='Trees', ylabel='Error')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('q2d_plot.png')
    
    plt.show()


def q2e():
    X, y = format_bank_data(load_bank_data('train.csv'))
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))

    bagged_tree_results = pd.DataFrame()
    single_tree_results = pd.DataFrame()
    for i in range(100):
        print(i)
        sample_X = X.sample(1000)
        sample_y = y[sample_X.index]
        sample_X = sample_X.reset_index(drop=True)
        sample_y = sample_y.reset_index(drop=True)
        model = ensl.RandomForestModel(
            sample_X.copy(), sample_y.copy(), 
            sample_rate=100, bag_rounds=500, num_sample_attributes=2
        )
        bagged_tree_results[i] = pd.Series(model.evaluate(X_test))
        single_tree_results[i] = pd.Series(model.model[0].evaluate(X_test))
    bagged_tree_results.replace(to_replace='no', value=0, inplace=True)
    single_tree_results.replace(to_replace='no', value=0, inplace=True)
    bagged_tree_results.replace(to_replace='yes', value=1, inplace=True)
    single_tree_results.replace(to_replace='yes', value=1, inplace=True)
    bagged_tree_results.to_csv('q2e_full_forest_results.csv', index=False)
    single_tree_results.to_csv('q2e_single_tree_results.csv', index=False)


def q2e_final():
    X_test, y_test = format_bank_data(load_bank_data('test.csv'))

    y_test.replace(to_replace='no', value=0, inplace=True)
    y_test.replace(to_replace='yes', value=1, inplace=True)

    full_df = pd.read_csv('TEST_q2e_full_forest_results.csv')
    stree_df = pd.read_csv('TEST_q2e_single_tree_results.csv')

    print(f'single tree general squared error: {calc_general_squared_error(stree_df, y_test)}')
    print(f'full forest general squared error: {calc_general_squared_error(full_df, y_test)}')
    
    # Bias: 0.11706194
    # Sample Variance: 0.021604101010101014
    # single tree general squared error: 0.13106269212272728
    # Bias: 0.1248
    # Sample Variance: 0.0 LIKELY BECAUSE RANDOM FOREST ONLY PREDICTS 0 ON THE DATA
    # full forest general squared error: 0.1248


def load_credit_default_data():
    return pd.read_csv(
        Path('credit_default', 'credit_default.csv'),
        index_col=False,
    )


def format_credit_default_data(df):
    df.SEX.replace(to_replace=1, value='male', inplace=True)
    df.SEX.replace(to_replace=2, value='female', inplace=True)

    df.EDUCATION.replace(to_replace=1, value='graduate school', inplace=True)
    df.EDUCATION.replace(to_replace=2, value='university', inplace=True)
    df.EDUCATION.replace(to_replace=3, value='high school', inplace=True)
    df.EDUCATION.replace(to_replace=4, value='others', inplace=True)
    df.EDUCATION.replace(to_replace=5, value='others', inplace=True)
    df.EDUCATION.replace(to_replace=6, value='others', inplace=True)
    df.EDUCATION.replace(to_replace=0, value='others', inplace=True)

    df.MARRIAGE.replace(to_replace=1, value='married', inplace=True)
    df.MARRIAGE.replace(to_replace=2, value='single', inplace=True)
    df.MARRIAGE.replace(to_replace=3, value='others', inplace=True)
    df.MARRIAGE.replace(to_replace=0, value='others', inplace=True)
    
    df['default payment next month'].replace(to_replace=1, value='yes', inplace=True)
    df['default payment next month'].replace(to_replace=0, value='no', inplace=True)

    return df


def split_credit_default_data(df: pd.DataFrame) -> (
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    df = df.sample(frac=1).reset_index(drop=True)
    split_index = 24000
    train = df.iloc[:split_index].reset_index(drop=True)
    test = df.iloc[split_index:].reset_index(drop=True)
    X_cols = [
        'LIMIT_BAL',
        'SEX',
        'EDUCATION',
        'MARRIAGE',
        'AGE',
        'PAY_0',
        'PAY_2',
        'PAY_3',
        'PAY_4',
        'PAY_5',
        'PAY_6',
        'BILL_AMT1',
        'BILL_AMT2',
        'BILL_AMT3',
        'BILL_AMT4',
        'BILL_AMT5',
        'BILL_AMT6',
        'PAY_AMT1',
        'PAY_AMT2',
        'PAY_AMT3',
        'PAY_AMT4',
        'PAY_AMT5',
        'PAY_AMT6',
    ]
    y_col = 'default payment next month'
    return train[X_cols], train[y_col], test[X_cols], test[y_col]


def q3_make_adaboost_model(X: pd.DataFrame, y: pd.Series) -> ensl.AdaBoostModel:
    return ensl.AdaBoostModel(X.copy(), y.copy(), 
        boost_rounds=500,
        max_tree_depth=1
    )


def q3_make_bagger_model(X: pd.DataFrame, y: pd.Series) -> ensl.BaggerModel:
    return ensl.BaggerModel(X.copy(), y.copy(), sample_rate=100, bag_rounds=500)


def q3_make_random_forest_model(X: pd.DataFrame, y: pd.Series) -> ensl.RandomForestModel:
    return ensl.RandomForestModel(
        X.copy(), y.copy(), 
        sample_rate=100,
        bag_rounds=500, 
        num_sample_attributes=2,
    )


def q3_plots():
    results = pd.DataFrame()
    results['test'] = 1 - pd.read_csv('q3_decision_trees_test_results.csv').set_index(['Unnamed: 0'])['0']
    results['train'] = 1 - pd.read_csv('q3_decision_trees_train_results.csv').set_index(['Unnamed: 0'])['0']
    
    fig = results.plot()
    plt.ylabel('Error')
    plt.xlabel('Tree Depth')

    plt.savefig('q3_decision_tree_plot.png')

    results = pd.DataFrame()
    results['test'] = 1 - pd.read_csv('q3_adaboost_test_results.csv').set_index(['Unnamed: 0'])['0']
    results['train'] = 1 - pd.read_csv('q3_adaboost_train_results.csv').set_index(['Unnamed: 0'])['0']
    
    fig = results.plot()
    plt.ylabel('Error')
    plt.xlabel('Trees')

    plt.savefig('q3_adaboost_plot.png')

    results = pd.DataFrame()
    results['test'] = 1 - pd.read_csv('q3_bagger_test_results.csv').set_index(['Unnamed: 0'])['0']
    results['train'] = 1 - pd.read_csv('q3_bagger_train_results.csv').set_index(['Unnamed: 0'])['0']
    
    fig = results.plot()
    plt.ylabel('Error')
    plt.xlabel('Trees')

    plt.savefig('q3_bagger_plot.png')

    results = pd.DataFrame()
    results['test'] = 1 - pd.read_csv('q3_random_forest_test_results.csv').set_index(['Unnamed: 0'])['0']
    results['train'] = 1 - pd.read_csv('q3_random_forest_train_results.csv').set_index(['Unnamed: 0'])['0']
    
    fig = results.plot()
    plt.ylabel('Error')
    plt.xlabel('Trees')

    plt.savefig('q3_random_forest_plot.png')

    plt.show()


def load_cement_data(t='train.csv') -> pd.DataFrame:
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

    return pd.read_csv(
        Path('concrete', 'concrete', t), 
        names=cols, 
        index_col=False,
    )


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
    c_data = load_cement_data('train.csv')
    test_c_data = load_cement_data('test.csv')

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
    model.weights.to_csv('q4a_weights.csv', index=False)


def q4a_final():
    data = pd.read_csv('q4a_convergence_of_weights.csv')['0']
    fig, ax = plt.subplots(1, 1)
    data.plot()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel('$||\\mathregular{w_i} - \\mathregular{w_{i-1}}||$')
    plt.xlabel('Round')
    plt.savefig('part2_q4a.png')
    # plt.show()


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
    c_data = load_cement_data('train.csv')
    test_c_data = load_cement_data('test.csv')

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


def q4b_final():
    data = pd.read_csv('q4b_convergence_of_weights.csv')['0']
    fig = data.plot().set_yscale('log')
    plt.ylabel('$||\\mathregular{w_i} - \\mathregular{w_{i-1}}||$')
    plt.xlabel('Round')
    plt.savefig('part2_q4b.png')


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
    c_data = load_cement_data('train.csv')
    ow = calc_optimal_weights(c_data[x_cols], c_data.Output)
    print(ow)

    # Cement        0.900565
    # Slag          0.786293
    # FlyAsh        0.851043
    # Water         1.298894
    # SP            0.129891
    # CoarseAggr    1.572249
    # FineAggr      0.998694
    # MODEL_BIAS   -0.015197
    # dtype: float64


def calc_optimal_weights(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    X['MODEL_BIAS'] = 1
    nX = X.to_numpy().T # Has to be transposed for lin. alg. to work
    ny = y.to_numpy()
    x_xt = np.matmul(nX, np.transpose(nX))
    x_xt_inv = np.linalg.inv(x_xt)
    x_xt_inv_x = np.matmul(x_xt_inv, nX)
    x_xt_inv_x_y = np.dot(x_xt_inv_x, ny)
    return pd.Series(x_xt_inv_x_y, index=X.columns)
