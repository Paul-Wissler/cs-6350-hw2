import math
import pandas as pd
import numpy as np

import DecisionTree as dtree
from .error_calcs import calc_weighted_gain


class AdaBoostModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            boost_rounds=100, error_f=dtree.calc_entropy, max_tree_depth=2, 
            default_value_selection='majority', reproducible_seed=True):
        self.y_mode = y.mode().iloc[0]
        self.X = X.copy()
        self.y = self.binarize_data(y.copy())
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.max_tree_depth = max_tree_depth
        self.model = self.create_booster(
            self.convert_numeric_vals_to_categorical(X.copy()), 
            self.binarize_data(y.copy()), sample_rate=sample_rate, 
            boost_rounds=boost_rounds, error_f=error_f, 
            max_tree_depth=self.max_tree_depth, reproducible_seed=reproducible_seed
        )

    def convert_numeric_vals_to_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_cols:
            return X
        for col, m in self.median.iteritems():
            is_gte_m = X[col] >= m
            X[col].loc[is_gte_m] = f'>={m}'
            X[col].loc[~is_gte_m] = f'<{m}'
        return X

    @property
    def numeric_cols(self) -> list:
        return self.X.select_dtypes(include=np.number).columns.tolist()

    @property
    def median(self) -> pd.Series:
        return self.X[self.numeric_cols].median()

    def binarize_data(self, y):
        is_mode = y == self.y_mode
        y.loc[is_mode] = '1'
        y.loc[~is_mode] = '-1'
        return y

    def create_booster(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            boost_rounds=100, error_f=dtree.calc_entropy, max_tree_depth=None, 
            reproducible_seed=True) -> list:
        boosted_model = list()
        current_weights = [1 / len(X)] * len(X)
        weights = list()
        error = 0
        for t in range(boost_rounds):
            print(t)
            tree = AdaBoostDecisionTreeModel(X, y, current_weights,
                max_tree_depth=self.max_tree_depth, error_f=error_f)
            # print(tree.tree)
            temp_error = error + 0
            error = 1 - tree.test(X, y)
            if temp_error != error:
                print(temp_error)
                print(error)
            # print(error)
            vote = self.calc_vote(error)
            boosted_model.append(tree)
            temp_weights = list()
            for i in range(len(y)):
                y_x_h = int(y[i]) * int(tree.check_tree(X.iloc[i], tree.tree))
                # print(y_x_h)
                # if y_x_h == -1:
                    # print('success')
                temp_weights.append(current_weights[i] * math.exp(-1 * vote * y_x_h))
            # current_weights = [weight / sum(current_weights) for weight in temp_weights]
            current_weights = [weight for weight in temp_weights]
            # print(sum(current_weights))
            # print(max(current_weights))
            # print(min(current_weights))
            # print(vote)
        return boosted_model

    @staticmethod
    def calc_vote(error):
        return 0.5 * math.log((1 - error) / error)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        X_test = self.convert_numeric_vals_to_categorical(X_test)
        predict_y = self.evaluate(X_test)
        s = y_test == self.binarize_data(predict_y)
        return s.sum() / s.count()

    def evaluate(self, X_test: pd.DataFrame) -> float:
        # i = 0
        # eval_df = pd.DataFrame()
        # for tree in self.model:
        #     eval_df[i] = tree.evaluate(X_test)
        #     i += 1
        # h = eval_df.mode(axis=1)[0]
        # return h
        return


class AdaBoostDecisionTreeModel(dtree.DecisionTreeModel):

    def __init__(self, X: pd.DataFrame, y: pd.Series, weights: list, 
            error_f=dtree.calc_entropy, max_tree_depth=None, 
            default_value_selection='majority'):
        self.X = X.copy()
        self.y = y.copy()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.input_max_tree_depth = max_tree_depth
        self.tree = self.make_decision_tree(
            self.convert_numeric_vals_to_categorical(X.copy()), y.copy(), weights, 
            error_f=error_f, max_tree_depth=self.max_tree_depth
        )

    def make_decision_tree(self, X: pd.DataFrame, y: pd.Series, w: list,
            error_f=dtree.calc_entropy, max_tree_depth=None) -> dict:
        split_node = self.determine_split(X, y, w, error_f)
        d = {split_node: dict()}
        for v in X[split_node].unique():
            X_v_cols = X.columns[X.columns != split_node]
            X_v = X[X_v_cols].loc[X[split_node] == v]
            y_v = y.loc[X[split_node] == v]
            if len(y_v.unique()) == 1:
                d[split_node][v] = y_v.unique()[0]
            elif max_tree_depth == 1:
                d[split_node][v] = self.default_value(y)
            else:
                d[split_node][v] = self.make_decision_tree(X_v, y_v, w, 
                    error_f, max_tree_depth - 1)
        return d

    @staticmethod
    def determine_split(X: pd.DataFrame, y: pd.Series, w: list, error_f) -> str:
        current_gain = 0
        split_node = X.columns[0]
        for col in X.columns:
            new_gain = calc_weighted_gain(X[col].values, y.values, w, f=error_f)
            if new_gain > current_gain:
                split_node = col + ''
                current_gain = new_gain + 0
        return split_node
