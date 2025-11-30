import pandas as pd
from tools.DropColumns import drop_columns
from tools.DropNa import drop_na
from tools.ParseLocation import ParseLocation
from tools.EnforceSchema import EnforceSchema
from tools.EnforceValueRanges import EnforceValueRanges

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, FunctionTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

SEED = 33


observation_schema = {
    'SpO₂':'float',
    'HR':'float',
    'PI':'float',
    'RR':'float',
    'EtCO₂':'float',
    'FiO₂':'float',
    'PRV':'float',
    'BP':'float',
    'Skin Temperature':'float',
    'Motion/Activity index':'float',
    'PVI':'float',
    'Hb level':'float',
    'SV':'float',
    'CO':'float',
    'Blood Flow Index':'float',
    'PPG waveform features':'float',
    'Signal Quality Index':'float',
    'Respiratory effort':'float',
    'O₂ extraction ratio':'float',
    'SNR':'float',
    'oximetry':'int',
    'latitude':'float',
    'longitude':'float'
}

valid_ranges = {
    'SpO₂': (95, 100),
    'HR': (60, 100),
    'PI': (0.2, 20),
    'RR': (12, 20),
    'EtCO₂': (35, 45),
    'FiO₂': (21, 100),
    'PRV': (20, 200),
    'BP': (60, 120),
    'Skin Temperature': (33, 38),
    'Motion/Activity index': None,
    'PVI': (10, 20),
    'Hb level': (12, 18),
    'SV': (60, 100),
    'CO': (4, 8),
    'Blood Flow Index': None,
    'PPG waveform features': None,
    'Signal Quality Index': (0, 100),
    'Respiratory effort': None,
    'O₂ extraction ratio': (0.2, 3),
    'SNR': (20, 40)
}

def get_numeric_columns(df):
    return df.select_dtypes(include=['int', 'float']).columns.tolist()

def get_pipeline_4(df):
    numeric_pipeline = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson")),
        #("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        #("scaler", MinMaxScaler(feature_range=(-1, 1)))
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    stack = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=300, 
                max_depth=10, 
                max_features=None,
                min_samples_leaf=2,
                min_samples_split=5,
                random_state=SEED)),
            ('knn', KNeighborsClassifier(n_neighbors=2)),
            ('logreg', LogisticRegression(max_iter=1000))
        ],
        final_estimator=LogisticRegression(),
        passthrough=True)
    
    observation_pipeline = Pipeline([
        ("schema", EnforceSchema(schema=observation_schema)),
        ("ranges", EnforceValueRanges(ranges=valid_ranges)),
        ("drop_geo", FunctionTransformer(drop_columns, kw_args={'columns': ['latitude', 'longitude']}, validate=False)),
        ("drop_na", FunctionTransformer(drop_na, kw_args={'how': 'any'}, validate=False)),
    
        ("encode", ColumnTransformer([
            ('cat', cat_pipeline, []),
            ('num', numeric_pipeline, get_numeric_columns)
        ], remainder='drop').set_output(transform="pandas")),
    
        ("variance_threshold", VarianceThreshold(threshold=0.01).set_output(transform="pandas")),
        #("select_kbest", SelectKBest(score_func=f_regression, k=10).set_output(transform="pandas")), 
        ("RFE", RFE(SVR(kernel="linear"), n_features_to_select=10, step=1).set_output(transform="pandas")),
        ("stack", stack)
    ])

    return observation_pipeline


