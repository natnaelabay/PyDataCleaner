import matplotlib as pt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from matplotlib import pyplot as pt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def numerical_feature_selection(
    df,
    labels,
    test_size,
    k,
    selection_function=f_regression
):
    '''
    For numerical features this method will select the best features and it has a few parameters to customize the process
    and selection function.

    params: 
        `df`: `required` : DataFrame
        `test_size` : `required` : test_size from 0 - 1
        `k` : `required` : number of features to select
        `selection_function` : `required` : function to select features
    return:
        tuple(x_train, x_test, y_train, y_test)

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
    This function will test a numerical model provided the training and test data set using a simple LinearRegression model

    params:
        `x_train_ts` : `required` : training data set
        `x_test_ts` : `required` : test data set
        `y_train` : `required` : training labels
        `y_test` : `required` : test labels
    return:
        tuple(x_train, x_test, y_train, y_test)    
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

    params:
        `df`: `required` : DataFrame    
        `test_size` : `required` : test_size from 0 - 1
        `k` : `required` : number of features to select
        `selection_func` : `required` : function to select features
    return:
        tuple(x_train, x_test, y_train, y_test)
    '''
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
    This method will select the best features using Recursive Feature Elimination

    params:
        `X` : `required` : training data set
        `y` : `required` : training labels
    return:
        tuple(x_train, x_test, y_train, y_test)
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


def feature_selection(
    df, labels,
    test_size,
    k,
    selection_function
):
    '''
    This method is the same as numerical_feature_selection but can be used for more generic feature selection
    by just passing in the selection function

    params:
        `df`: `required` : DataFrame
        `test_size` : `required` : test_size from 0 - 1
        `k` : `required` : number of features that will be selected
        `selection_function` : `required` : A selection function that will be used as a parameter for `SelectKBest` 

    returns:
        tuple(x_train_fs, x_test_fs, y_train, y_test)
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
