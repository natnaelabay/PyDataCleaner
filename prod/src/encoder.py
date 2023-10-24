from py_loader import PyLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from utils.funcs import isnotebook
from sklearn.model_selection import train_test_split


class CategoricalEncoder:

    def __init__(self, loader: PyLoader) -> None:
        self.loader = loader
        self.temp_df = loader.df.clone_df()
        self.categorical_columns = None

    def detect_categorical_cols(self, certain_cat_cols: list = []):
        '''
        This member function trys to detect categorical features present
        in the dataframe. if provided a list of certain_cat_cols 
        it will merge those with the detected ones 
        and return the dataframe.

        params
        ------

            ``certain_cat_cols``: list of categorical cols that you are sure of. :  default []

        returns
        -------
            self or pd.DataFrame
        '''

        temp = self.temp_df.copy()
        other_cols = temp.select_dtypes(
            include=[np.number, 'datetime64', 'timedelta64']).columns.values
        categorical_columns = []

        for col in temp.columns:
            if col not in other_cols and col not in categorical_columns and col not in certain_cat_cols:
                categorical_columns.append(col)
                temp[col] = temp[col].astype("category")

        categorical_columns += certain_cat_cols

        if plot:
            for col in categorical_columns:
                labels = temp[col].astype('category').cat.categories.tolist()
                counts = temp[col].value_counts()
                sizes = [counts[var_cat] for var_cat in labels]
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                        shadow=True)  # autopct is show the % on plot
                ax1.axis('equal')
            plt.show()

        self.categorical_columns = categorical_columns

        if isnotebook():
            display(temp[categorical_columns])

        return self.categorical_columns

    def encode_categorical_columns(self, encoding_type, replace_map: dict = None):
        '''
        This member function encodes the categorical columns present in the dataframe.

        params
        ------

            ``encoding_type``: str, one of the following:

            ``replace_map``: dict, if encoding_type is 'replace'
                this dict will be used to replace the original values with the encoded values.

        returns
        -------
            self or pd.DataFrame
        '''
        if self.temp_df is None:
            self.temp_df = self.df.copy()

        if self.categorical_columns is None:
            self.detect_categorical_columns()

        if encoding_type == "replace":
            if replace_map is None:
                for col in self.categorical_columns:
                    labels = self.temp_df[col].astype(
                        'category').cat.categories.tolist()
                    replace_map_comp = {col: {k: v for k, v in zip(
                        labels, list(range(1, len(labels)+1)))}}
                    self.temp_df[col] = self.temp_df[col].replace(
                        replace_map_comp[col]
                    )

                if isnotebook():
                    display(self.temp_df)

            else:
                for col in self.categorical_columns:
                    self.temp_df[col].replace(replace_map[col], inplace=True)

                if isnotebook():
                    display(self.temp_df)

            return (self.temp_df, self.temp_df[self.categorical_columns])

        elif encoding_type == "one-hot":
            from sklearn.preprocessing import OneHotEncoder

            train, test = train_test_split(
                self.temp_df[categorical_columns],
                test_size=0.2
            )

            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(train)
            train_encoded = encoder.transform(train)
            test_encoded = encoder.transform(test)

            self.temp_df = pd.concat([train_encoded, test_encoded,
                                      self.temp_df[self.temp_df.difference(
                                          self.categorical_columns
                                      )]], axis=1)

            return (self.temp_df, self.temp_df[self.categorical_columns])

            # self.temp_df = pd.DataFrame(
            #     encoder.transform(
            #         self.temp_df[self.categorical_columns]).toarray(),
            #     columns=encoder.get_feature_names(self.categorical_columns)
            # )

            # self.temp_df = pd.get_dummies(
            #     self.temp_df, columns=self.categorical_columns)

        elif encoding_type == "label":
            label_encoder = LabelEncoder()
            for col in self.categorical_columns:
                self.temp_df[col] = label_encoder.fit_transform(
                    self.temp_df[col])

            return (self.temp_df, self.temp_df[self.categorical_columns])

            # self.temp_df = pd.get_dummies(
            #     self.temp_df, columns=self.categorical_columns, prefix=self.categorical_columns)

        else:
            print("=============PyDataCleaner=============")
            print("==Please choose a valid encoding type==")
            print("=============PyDataCleaner=============")

        return (self.temp_df, self.temp_df[self.categorical_columns])

    def persist(self):
        if self.temp_df:
            self.loader.df = self.temp_df.copy()
            
        return self.loader.df
