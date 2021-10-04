import numpy as np
import pandas as pd

from .batch_gradient_descent import BatchGradientDescentModel


class StochasticGradientDescentModel(BatchGradientDescentModel):

    def compute_dJ_dw_j(self, weights: pd.Series, col: str) -> float:
        sample_x = self.X.sample(1)
        x_i_multiply_w = np.dot(sample_x.to_numpy(), weights.to_numpy())
        error = self.y[sample_x.index[0]] - x_i_multiply_w
        return -np.dot(error, sample_x[col].to_numpy())
