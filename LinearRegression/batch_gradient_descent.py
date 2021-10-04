import numpy as np
import pandas as pd


class BatchGradientDescentModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, rate=0.1, error=0.05, max_rounds=100,):
        self.X = X.copy()
        self.y = y.copy()
        self.max_rounds = max_rounds
        self.rate = rate
        self.error = error
        self.convergence_of_weights = pd.Series()
        self.weights = self.create_model(X.copy(), y.copy())

    def create_model(self, X: pd.DataFrame, y: pd.Series):
        w = pd.Series([0] * len(X.columns), index=X.columns)
        i = 0
        while i <= self.max_rounds + 1: #TODO: add error check to condition
            if i == self.max_rounds:
                print('WARNING: Model failed to converge below specified error')
                # print(w)
                # print(self.convergence_of_weights)
                return w
            i += 1
            w = self.compute_new_weights(self.compute_gradient(w), w)
            # cost = self.compute_cost(w)
            # e = self.compute_mean_error(w) # Not sure about this implementation . . .
            # print(e)
            # print(self.compute_norm(e))
            # print(np.abs(np.mean(e)))
        # print(w)
        return w

    def compute_new_weights(self, gradient: pd.Series, weights: pd.Series) -> pd.Series:
        new_weights = weights - self.rate * gradient
        new_weights.name = 'weights'
        self.convergence_of_weights = (
            self.convergence_of_weights.append(
                pd.Series([self.compute_norm(new_weights - weights)]), 
                ignore_index=True
            ).reset_index(drop=True)
        )
        return new_weights

    def compute_gradient(self, weights: pd.Series) -> pd.Series:
        gradient = pd.Series(index=weights.index, name='gradient')
        for col, _ in gradient.iteritems():
            gradient[col] = self.compute_dJ_dw_j(weights.copy(), col)
        return gradient

    def compute_dJ_dw_j(self, weights: pd.Series, col: str) -> float:
        x_i_multiply_w = np.dot(self.X.to_numpy(), weights.to_numpy())
        error = self.y.to_numpy() - x_i_multiply_w
        return -np.dot(error, self.X[col].to_numpy())

    def compute_cost(self, X, y, weights: pd.Series) -> float:
        x_i_multiply_w = np.dot(X.to_numpy(), weights.to_numpy())
        error = np.square(y.to_numpy() - x_i_multiply_w)
        return np.sum(0.5 * np.square(error))

    def compute_mean_error(self, weights: pd.Series) -> np.ndarray:
        e = self.compute_point_error(weights)
        return np.abs(np.mean(e))

    # TODO: Implement in other functions?
    def compute_point_error(self, weights: pd.Series) -> np.ndarray:
        x_i_multiply_w = np.dot(self.X.to_numpy(), weights.to_numpy())
        return self.y.to_numpy() - x_i_multiply_w

    @staticmethod
    def compute_norm(x) -> float:
        return np.linalg.norm(x)
