import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import repeat

from pathlib import Path

import QuestionAnswers.part1 as part1
import QuestionAnswers.part2 as part2
import EnsembleLearning as ensl
import DecisionTree as dtree

pd.options.mode.chained_assignment = None  # default='warn'


def test_cumulative_trees(bagger_model: ensl.BaggerModel, X_test: pd.DataFrame, 
        y_test: pd.Series, i) -> float:
    X_test = bagger_model.convert_numeric_vals_to_categorical(X_test.copy())
    test_results = pd.Series()
    print(i)
    predict_y = bagger_model.evaluate_specific_trees(X_test.copy(), bagger_model.model[:i+1])
    s = y_test == predict_y
    return s.sum() / s.count()


def adaboost_test_cumulative_trees(model: ensl.AdaBoostModel, X_test: pd.DataFrame, 
        y_test: pd.Series, i):
    print(i)
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_test = model.convert_numeric_vals_to_categorical(X_test.copy())
    test_results = pd.Series()
    cum_test_results = pd.Series()
    y_test = model.binarize_data(y_test)

    predict_y = model.trees[i].evaluate(X_test.copy())
    s = y_test == predict_y
    test_results = test_results.append(
        pd.Series([s.sum() / s.count()], index=[i+1])
    )

    cum_predict_y = model.evaluate_specific_trees(X_test.copy(), model.trees[:i+1], 
        model.votes[:i+1])
    s = y_test == cum_predict_y
    cum_test_results = cum_test_results.append(
        pd.Series([s.sum() / s.count()], index=[i+1])
    )
    return test_results, cum_test_results


def adaboost_test(model: ensl.AdaBoostModel, X_test: pd.DataFrame, 
        y_test: pd.Series, i):
    print(i)
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_test = model.convert_numeric_vals_to_categorical(X_test.copy())
    
    cum_test_results = pd.Series()
    y_test = model.binarize_data(y_test)

    cum_predict_y = model.evaluate_specific_trees(X_test.copy(), model.trees[:i+1], 
        model.votes[:i+1])
    s = y_test == cum_predict_y
    cum_test_results = cum_test_results.append(
        pd.Series([s.sum() / s.count()], index=[i+1])
    )
    return cum_test_results


def bagger_test_cumulative_trees(model, X_test: pd.DataFrame, y_test: pd.Series, 
            i: int) -> float:
        X_test = model.convert_numeric_vals_to_categorical(X_test.copy())
        test_results = pd.Series()
        predict_y = model.evaluate_specific_trees(X_test.copy(), model.model[:i+1])
        s = y_test == predict_y
        test_results = test_results.append(
            pd.Series([s.sum() / s.count()], index=[i])
        )
        return test_results


def q3_make_decision_tree(X: pd.DataFrame, y: pd.Series, tree_depth: int):
    print(f'Tree Depth: {tree_depth}')
    model = dtree.DecisionTreeModel(X, y, max_tree_depth=tree_depth)
    return model, tree_depth


def q3_test_decision_tree(models: dtree.DecisionTreeModel, X: pd.DataFrame, 
        y: pd.Series, i: int) -> float:
    print(i)
    return models[i].test(X, y), i


def q2c_bag_build_and_eval(_):
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
    X_test = test_bank_data[x_cols]
    y_test = test_bank_data.y
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_test[numeric_cols] = X_test[numeric_cols].astype(int)

    sample_X = X.sample(1000)
    sample_y = y[sample_X.index]
    sample_X = sample_X.reset_index(drop=True)
    sample_y = sample_y.reset_index(drop=True)
    model = ensl.BaggerModel(sample_X, sample_y, sample_rate=100, bag_rounds=500)
    # (bag, single)
    return pd.Series(model.evaluate(X_test)), pd.Series(model.model[0].evaluate(X_test))


def random_forest_attr_test(X, y, X_test, y_test, n):
    model = ensl.RandomForestModel(
        X.copy(), y.copy(), 
        sample_rate=100, 
        # bag_rounds = 1,
        bag_rounds=500, 
        num_sample_attributes=n,
    )
    print(n)
        
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
    train_results.to_csv(f'TEST_q2d_train_results_attr_{n}.csv')
    test_results.to_csv(f'TEST_q2d_test_results_attr_{n}.csv')


def q2e_forest_build_and_eval(_):
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
    X_test = test_bank_data[x_cols]
    y_test = test_bank_data.y
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_test[numeric_cols] = X_test[numeric_cols].astype(int)

    sample_X = X.sample(1000)
    sample_y = y[sample_X.index]
    sample_X = sample_X.reset_index(drop=True)
    sample_y = sample_y.reset_index(drop=True)
    model = ensl.RandomForestModel(
        sample_X.copy(), sample_y.copy(), 
        sample_rate=100, bag_rounds=500, num_sample_attributes=2
    )
    # (full_forest, single)
    return pd.Series(model.evaluate(X_test)), pd.Series(model.model[0].evaluate(X_test))


def main():
    part1.q5b()
    part1.q5c()
    part1.q5d()
    
    part2.q2b()
    part2.q2b_final()
        
    part2.q4a()
    part2.q4a_final()
    part2.q4b()
    part2.q4b_final()
    part2.q4c()

    # import json
    # print('final tree: ', json.dumps(model.tree, sort_keys=True, indent=2))

    # q2a ######################################################################
    model, X, y, X_test, y_test = part2.make_q2a_model()
    with Pool() as pool:
        train_results = pool.starmap(
            adaboost_test_cumulative_trees, 
            zip(
                repeat(model),
                repeat(X.copy()),
                repeat(y.copy()),
                range(len(model.trees))
            )
        )
        test_results = pool.starmap(
            adaboost_test_cumulative_trees, 
            zip(
                repeat(model),
                repeat(X_test.copy()),
                repeat(y_test.copy()),
                range(len(model.trees))
            )
        )

    bagged_tree_results = pd.Series()
    single_tree_results = pd.Series()
    for s, c in test_results:
        bagged_tree_results = bagged_tree_results.append(c)
        single_tree_results = single_tree_results.append(s)

    print(bagged_tree_results)
    print(single_tree_results)

    bagged_tree_results.to_csv('TEST_q2a_test_cum_results.csv')
    single_tree_results.sort_index().to_csv('TEST_q2a_test_single_results.csv')

    bagged_tree_results = pd.Series()
    single_tree_results = pd.Series()
    for s, c in train_results:
        bagged_tree_results = bagged_tree_results.append(c)
        single_tree_results = single_tree_results.append(s)

    print(bagged_tree_results)
    print(single_tree_results)

    bagged_tree_results.to_csv('TEST_q2a_train_cum_results.csv')
    single_tree_results.to_csv('TEST_q2a_train_single_results.csv')

    part2.q2a_final()

    # q2c ######################################################################
    with Pool() as pool:
        results = pool.map(q2c_bag_build_and_eval, range(100))

    bagged_tree_results = pd.DataFrame()
    single_tree_results = pd.DataFrame()
    i = 0
    for b, s in results:
        bagged_tree_results[i] = b.copy()
        single_tree_results[i] = s.copy()
        i += 1

    bagged_tree_results.replace(to_replace='no', value=0, inplace=True)
    single_tree_results.replace(to_replace='no', value=0, inplace=True)
    bagged_tree_results.replace(to_replace='yes', value=1, inplace=True)
    single_tree_results.replace(to_replace='yes', value=1, inplace=True)
    bagged_tree_results.to_csv('TEST_q2c_bagged_tree_results.csv', index=False)
    single_tree_results.to_csv('TEST_q2c_single_tree_results.csv', index=False)

    part2.q2c_final()

    # q2d ######################################################################
    X, y = part2.format_bank_data(part2.load_bank_data('train.csv'))
    X_test, y_test = part2.format_bank_data(part2.load_bank_data('test.csv'))
    with Pool() as pool:
        pool.starmap(
            random_forest_attr_test, 
            zip(
                repeat(X.copy()),
                repeat(y.copy()),
                repeat(X_test.copy()),
                repeat(y_test.copy()),
                [2, 4, 6]
            )
        )

    part2.q2d_final()

    # q2e ######################################################################
    with Pool() as pool:
        results = pool.map(q2e_forest_build_and_eval, range(100))

    bagged_tree_results = pd.DataFrame()
    single_tree_results = pd.DataFrame()
    i = 0
    for b, s in results:
        bagged_tree_results[i] = b.copy()
        single_tree_results[i] = s.copy()
        i += 1
        
    bagged_tree_results.replace(to_replace='no', value=0, inplace=True)
    single_tree_results.replace(to_replace='no', value=0, inplace=True)
    bagged_tree_results.replace(to_replace='yes', value=1, inplace=True)
    single_tree_results.replace(to_replace='yes', value=1, inplace=True)
    bagged_tree_results.to_csv('TEST_q2e_full_forest_results.csv', index=False)
    single_tree_results.to_csv('TEST_q2e_single_tree_results.csv', index=False)

    part2.q2e_final()

    # q3 ######################################################################
    X, y, X_test, y_test = part2.split_credit_default_data(
        part2.format_credit_default_data(
            part2.load_credit_default_data()
        )
    )

    # Single Decision Tree(s)
    print('SINGLE DECISION TREE(S)')
    with Pool() as pool:
        results = pool.starmap(
            q3_make_decision_tree,
            zip(
                repeat(X.copy()),
                repeat(y.copy()),
                range(1, len(X.columns)+1)
            )
        )

    models = pd.Series()
    for model, tree in results:
        models = models.append(pd.Series(model, index=[tree]))

    with Pool() as pool:
        train_results = pool.starmap(
            q3_test_decision_tree,
            zip(
                repeat(models),
                repeat(X.copy()),
                repeat(y.copy()),
                models.index.to_list(),
            )
        )
        test_results = pool.starmap(
            q3_test_decision_tree,
            zip(
                repeat(models),
                repeat(X_test.copy()),
                repeat(y_test.copy()),
                models.index.to_list(),
            )
        )

    test_results_series = pd.Series()
    for result, i in test_results:
        test_results_series = test_results_series.append(pd.Series(result, index=[i]))

    train_results_series = pd.Series()
    for result, i in train_results:
        train_results_series = train_results_series.append(pd.Series(result, index=[i]))

    print(test_results_series)
    print(train_results_series)

    test_results_series.to_csv('q3_decision_trees_test_results.csv')
    train_results_series.to_csv('q3_decision_trees_train_results.csv')

    # AdaBoost
    print('ADABOOST')
    model = part2.q3_make_adaboost_model(X.copy(), y.copy())
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

    with Pool() as pool:
        train_results = pool.starmap(
            adaboost_test, 
            zip(
                repeat(model),
                repeat(X.copy()),
                repeat(y.copy()),
                ix
            )
        )
        test_results = pool.starmap(
            adaboost_test, 
            zip(
                repeat(model),
                repeat(X_test.copy()),
                repeat(y_test.copy()),
                ix
            )
        )

    train_results_series = pd.Series()
    for r in train_results:
        train_results_series = train_results_series.append(r)

    test_results_series = pd.Series()
    for r in test_results:
        test_results_series = test_results_series.append(r)

    print(train_results_series)
    print(test_results_series)

    train_results_series.to_csv('q3_adaboost_train_results.csv')
    test_results_series.to_csv('q3_adaboost_test_results.csv')

    # Bagger
    print('BAGGER')
    model = part2.q3_make_bagger_model(X.copy(), y.copy())
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

    with Pool() as pool:
        train_results = pool.starmap(
            bagger_test_cumulative_trees, 
            zip(
                repeat(model),
                repeat(X.copy()),
                repeat(y.copy()),
                ix
            )
        )
        test_results = pool.starmap(
            bagger_test_cumulative_trees, 
            zip(
                repeat(model),
                repeat(X_test.copy()),
                repeat(y_test.copy()),
                ix
            )
        )

    train_results_series = pd.Series()
    for r in train_results:
        train_results_series = train_results_series.append(r)

    test_results_series = pd.Series()
    for r in test_results:
        test_results_series = test_results_series.append(r)
    
    print(train_results_series)
    print(test_results_series)
    train_results_series.to_csv('q3_bagger_train_results.csv')
    test_results_series.to_csv('q3_bagger_test_results.csv')

    # Random Forest
    print('RANDOM FOREST')
    model = part2.q3_make_random_forest_model(X.copy(), y.copy())
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

    with Pool() as pool:
        train_results = pool.starmap(
            bagger_test_cumulative_trees, 
            zip(
                repeat(model),
                repeat(X.copy()),
                repeat(y.copy()),
                ix
            )
        )
        test_results = pool.starmap(
            bagger_test_cumulative_trees, 
            zip(
                repeat(model),
                repeat(X_test.copy()),
                repeat(y_test.copy()),
                ix
            )
        )

    train_results_series = pd.Series()
    for r in train_results:
        train_results_series = train_results_series.append(r)

    test_results_series = pd.Series()
    for r in test_results:
        test_results_series = test_results_series.append(r)
    
    print(train_results_series)
    print(test_results_series)
    train_results_series.to_csv('q3_random_forest_train_results.csv')
    test_results_series.to_csv('q3_random_forest_test_results.csv')
    
    part2.q3_plots()


if __name__ == '__main__':
    main()
