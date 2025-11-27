import pandas as pd

def drop_columns(X, y=None, columns=None):
    X_transformed = X.drop(columns=columns or [], errors='ignore').copy()
    if y is not None:
        return X_transformed, y
    return X_transformed