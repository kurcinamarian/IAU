import pandas as pd

def drop_na(X, y=None, how='any', subset=None):
    mask = X.dropna(how=how, subset=subset).index
    X_transformed = X.loc[mask].copy()
    if y is not None:
        y_transformed = y.loc[mask].copy()
        return X_transformed, y_transformed
    return X_transformed