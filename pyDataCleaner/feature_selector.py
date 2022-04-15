from sklearn.datasets import make_regression
import matplotlib as pt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from matplotlib import pyplot as pt
import pandas as pd
import numpy as np


def numerical_feature_selection(df, labels, test_size, percentage, k, selection_function=f_regression):
    '''
    params: 
        `df`: DataFrame
        `test_size` : test_size from 0 - 1
        `percentage` : train and test spliting percentage
        `k` : number of features that will be selected
        `selection_function` : A selection function if not provided default is f_regression/mutual_info_regression from the `sklearn.feature_selection` module. 
        `labels` : list of column names that the model will try to predict
    '''
    features = [col for col in df.columns.values if col not in labels]
    x_train, x_test, y_train, y_test = train_test_split(
        df[features],
        df[labels],
        test_size=test_size
    )

    fs = SelectKBest(
        score_func=selection_function,
        k=k
    )

    fs.fit(x_train, y_train)
    x_train_fs = fs.transform(x_train)
    x_test_fs = fs.transform(x_test)
    pt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    return x_train_fs, x_test_fs, y_train, y_test


def test_regression_model(
    x_train_ts,
    x_test_ts,
    y_train,
    y_test,
):
    '''
    This methods tests a data set with a regression model
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    model = LinearRegression()
    model.fit(x_train_ts, y_train)
    y_pred = model.predict(x_test_ts)
    print('MAE(Mean Absolute Error): ', mean_absolute_error(y_test, y_pred))
    print('MSE(Mean Squared Error): ', mean_squared_error(y_test, y_pred))
    return model


# Categorical feature selection


def categorical_feature_selection(
    df: pd.DataFrame,
    labels,
    test_size,
    selection_func,
    k="all",
):
    '''
    This method will select categorical features
    '''
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    features = [col for col in df.columns.values if col not in labels]

    features_df = df[features].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        features_df,
        df[labels],
        test_size=test_size
    )

    # encode input features/data

    encoder = OrdinalEncoder()
    encoder.fit(x_train)
    x_train_encoded = encoder.transform(x_train)
    x_test_encoded = encoder.transform(x_test)

    # encode target variables using LabelEncoder

    le_encoder = LabelEncoder()
    le_encoder.fit(y_train)

    y_train_encoded = le_encoder.transform(y_train)
    y_test_encoded = le_encoder.transform(y_test)

    # 1. Mutual information feature with (mutual_info_classif)
    # 2. Mutual information feature with chi-squared (chi2)

    fs = SelectKBest(
        score_func=selection_func,
        k=k
    )

    fs.fit(x_train_encoded, y_train_encoded)
    x_train_fs = fs.transform(x_train_encoded)
    x_test_fs = fs.transform(x_test_encoded)

    pt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pt.show()

    return x_train_fs, x_test_fs, fs


def rfe_for_regression(
    X,
    y,
):
    '''
    This method will test the model with the test data set
    '''

    from numpy import mean
    from numpy import std
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import RFE

    rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=5)
    model = DecisionTreeRegressor()
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    # evaluate model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_jobs=-1)
    # report performance
    print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    return model

