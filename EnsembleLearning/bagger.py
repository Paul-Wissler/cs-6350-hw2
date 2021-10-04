import pandas as pd
import numpy as np

import DecisionTree as dtree
from DecisionTree import calc_gain, calc_entropy

class BaggerModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, bag_rounds=100,
            error_f=calc_entropy, max_tree_depth=None, 
            default_value_selection='majority', reproducible_seed=True):
        self.X = X.copy()
        self.y = y.copy()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.max_tree_depth = max_tree_depth
        self.model = self.create_bagger(
            self.convert_numeric_vals_to_categorical(X.copy()), y, 
            sample_rate=sample_rate, bag_rounds=bag_rounds, error_f=error_f, 
            max_tree_depth=self.max_tree_depth, reproducible_seed=reproducible_seed
        )

    def convert_numeric_vals_to_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_cols:
            return X
        for col, m in self.median.iteritems():
            try:
                is_gte_m = X[col] >= m
            except TypeError:
                print(X[col])
            X[col].loc[is_gte_m] = f'>={m}'
            X[col].loc[~is_gte_m] = f'<{m}'
        return X

    @property
    def numeric_cols(self) -> list:
        return self.X.select_dtypes(include=np.number).columns.tolist()

    @property
    def median(self) -> pd.Series:
        return self.X[self.numeric_cols].median()

    def create_bagger(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            bag_rounds=100, error_f=calc_entropy, max_tree_depth=None, 
            reproducible_seed=True) -> list:
        if not sample_rate:
            sample_rate = self.auto_calc_subset(len(X))

        bag = list()
        for t in range(bag_rounds):
            if reproducible_seed:
                X_s = X.sample(n=sample_rate, replace=True, random_state=t * 100000)
            else:
                X_s = X.sample(n=sample_rate, replace=True)
            y_s = y.iloc[X_s.index]
            bag.append(dtree.DecisionTreeModel(X_s, y_s))
        return bag

    @staticmethod
    def auto_calc_subset(i):
        return round(.2 * i)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        X_test = self.convert_numeric_vals_to_categorical(X_test)
        predict_y = self.evaluate(X_test)
        s = y_test == predict_y
        return s.sum() / s.count()

    def evaluate(self, X_test: pd.DataFrame) -> float:
        i = 0
        eval_df = pd.DataFrame()
        for tree in self.model:
            eval_df[i] = tree.evaluate(X_test)
            i += 1
        h = eval_df.mode(axis=1)[0]
        return h
