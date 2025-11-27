import pandas as pd

def ParseLocation(X, column='location'):
    X = X.copy()
    if column not in X.columns:
        return X
    
    split_cols = X[column].astype(str).str.split('/', n=1, expand=True)
    X['continent'] = split_cols[0].str.strip()
    X['city'] = split_cols[1].str.strip() if split_cols.shape[1] > 1 else None
    X = X.drop(columns=[column], errors='ignore')
    return X