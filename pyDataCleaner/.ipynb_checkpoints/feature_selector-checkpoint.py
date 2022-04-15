from pydoc import doc
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class FeatureSelector:
    def __init__(self, df):
        self.df = df

    def __auto_selector(self):
        """
        accepts out_out and input columns and applies the ncessary feature selection algorithms from sklearn library

        """
        # impute missing values
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(self.df)
        self.df = pd.DataFrame(imp.transform(self.df), columns=self.df.columns)

        # encode categorical data
        le = LabelEncoder()
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = le.fit_transform(self.df[col])

        # create correlation matrix
        corr = self.df.corr()
        # plot correlation matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)
        plt.show()

        # select features based on correlation
        corr_var = corr.index
        corr_var = corr_var[abs(corr[corr_var] > 0.8)]
        print(corr_var)

        # select features based on chi-square test
        X = self.df.drop(['out_out'], axis=1)
        y = self.df['out_out']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train.shape, X_test.shape, y_train.shape, y_test.shape

        # select best features based on chi-square test
        chi2_selector = SelectKBest(chi2, k=10)
        chi2_selector.fit(X_train, y_train)
        chi2_selector.get_support()

        # select best features based on correlation
        corr_selector = SelectKBest(chi2, k=10)