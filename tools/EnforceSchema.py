import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import dateparser

class EnforceSchema(BaseEstimator, TransformerMixin):
    def __init__(self, schema=None):
        self.schema = schema

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.schema is None:
            return X

        X = X.copy()

        for col, col_type in self.schema.items():
            if col not in X.columns:
                continue
            if col_type == 'int':
                X[col] = pd.to_numeric(X[col], errors='coerce').astype('Int64')

            elif col_type == 'float':
                X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)

            elif col_type == 'numeric':
                X[col] = pd.to_numeric(X[col], errors='coerce')

            elif col_type == 'date':
                X[col] = X[col].apply(
                    lambda x: dateparser.parse(str(x))
                    if pd.notnull(x) else pd.NaT
                )

            elif col_type == 'string':
                X[col] = X[col].astype(str)
                X[col] = X[col].replace('nan', np.nan)

            elif col_type == 'bool':
                X[col] = (
                    X[col]
                    .astype(str)
                    .str.lower()
                    .map({'true': True, 'false': False, '1': True, '0': False})
                )

            elif col_type == 'category':
                X[col] = X[col].astype('category')

    
        return X