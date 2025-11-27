import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EnforceValueRanges(BaseEstimator, TransformerMixin):
    def __init__(self, ranges=None):
        self.ranges = ranges

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.ranges is None:
            return X

        for col, valid_range in self.ranges.items():
            if col not in X.columns:
                continue
            if valid_range is None:
                continue  

            low, high = valid_range
            X[col] = pd.to_numeric(X[col], errors='coerce')

            mask = ~X[col].between(low, high, inclusive='both')
            X.loc[mask, col] = np.nan

        return X