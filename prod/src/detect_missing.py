import pandas as pd
import numpy as np
from .py_loader import PyLoader
from sklearn.impute import SimpleImputer


class DetectMissing:
    def __init__(self, loader: PyLoader) -> None:

        # for readability purposes we gave an alias variable to the loaders data_frame variable
        self.loader = loader
        self.df = loader.data_frame
        self.df_stat = None
        self.cleaning_options = None
        self.temp_df = loader.clone_df()

    def describe_dataset(self, deep: bool = False):
        '''
        This function will describe the dataframe and return a dataframe with the following columns:
        Column_Name, missing value percentage, missing values count, unique values count 
        [use full for categorical data types]
        params:
            deep: bool -> if True, will return the full description of the dataframe
                        -> if False, will return a simplified description of the dataframe
        returns: pd.DataFrame
        '''

        if deep:
            return self.loader.data_frame.describe()

        unique_values = self.df.nunique()
        row_size = len(self.df)
        missing_values = self.df.isnull().sum()
        missing_values_percentage = round((missing_values * 100) / row_size, 2)
        self.df_stat = {
            "Column_Name": self.df.columns.values,
            "Unique_Values": unique_values,
            "Data_Types": self.df.dtypes,
            "Missing_Values_Count": missing_values,
            "Missing_Values_Percentage": missing_values_percentage,
        }

        # display(pd.DataFrame(self.df_stat))

        return self.df_stat

    def detect_options(self):
        '''
        This function will suggest cleaning options for the dataframe.
        (NB: The suggestion is based on the most common patterns
        from many datasets and further analysis may be necessary based on your data) 

        returns: pd.DataFrame
        '''

        if not self.df_stat:
            self.describe_data()

        nan_percentage = []
        imputation_techniques = []
        cols_with_null = []

        # if percentage is greater than 50% then suggest imputation technique as removal else as ModelBased imputation or single Imputation
        for index, percentage in enumerate(self.df_stat["Missing_Values_Percentage"]):
            if percentage > 50:
                cols_with_null.append(self.df_stat["Column_Name"][index])
                nan_percentage.append(percentage)
                imputation_techniques.append("Removal")
            elif percentage != 0:
                cols_with_null.append(self.df_stat["Column_Name"][index])
                nan_percentage.append(percentage)
                imputation_techniques.append("ModelBased or SingleImputation")

        self.cleaning_options = pd.DataFrame({
            "Columns": cols_with_null,
            "Null Percentage": list(filter((0).__ne__, nan_percentage)),
            "Suggested Cleaning Method": imputation_techniques
        })

        # display(self.cleaning_suggestions)

        return self.cleaning_options

    def clean_missing(self,
                      columns: list = None,
                      strategy: str = "mean",
                      drop_numerical: list = False,
                      drop_non_numerical: list = False,
                      numerical=[],
                      non_numerical=[],
                      axis: int = 1,
                      missing_values=np.nan,
                      persist=True
                      ):
        '''
        Provides ways of cleaning/clearing missing values in your dataframe
        with a few common ways that occur in every dataset.

        params
        ------

        ``columns``: list of column name: default ``None``
            list of columns to be handled

        ``strategy``: str, defailt ``mean``, could be one of  ['mean', 'median', 'most_frequent', 'constant']
            imputation strategy

        ``drop_numerical``: list, 
            list of numerical columns to be dropped

        `drop_non_numerical`: list, 
            list of non numerical columns to be dropped

        `numerical`: list, default: ``[]`` (but will be auto filled if provided nothing)
            list of numerical columns to work on.

        `non_numerical`: list, default: ``[]`` (but will be auto filled if provided nothing)
            list of non numerical columns to work on.

        ``axis``: int, default ``1``
            axis to be used

        ``missing_values``: Any (list, str), defailt ``np.nan``,
            value to be used for imputation


        returns
        -------
        pd.DataFrame
        '''

        if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
            raise Exception("Invalid argument")

        elif strategy is None:
            strategy = "mean"

        if self.temp_df is None:
            self.temp_df = self.df.copy()

        if not columns:
            columns = self.temp_df.columns.tolist()

        if drop_numerical and (len(drop_numerical) == len(columns) == len(self.temp_df.columns)):
            print("=================PyDataCleaner====================")
            print("You are attempting to remove all the numerical columns")
            print("=================PyDataCleaner====================")

        if drop_non_numerical and \
            (len(drop_non_numerical) ==
             len(columns) ==
                len(self.temp_df.columns)):
            print("=================PyDataCleaner====================")
            print("You are attempting to remove all the non numerical columns")
            print("=================PyDataCleaner====================")

        if (drop_non_numerical and drop_numerical) and (len(drop_non_numerical) + len(drop_numerical)) == len(columns) == len(self.temp_df.columns):
            print("WARNING: Tried to drop both numerical and non-numerical columns")
            return

        if drop_numerical:

            self.temp_df.drop(
                drop_numerical,
                axis=axis,
                inplace=True
            )

            columns = [
                col for col in columns if col not in drop_numerical
            ]

        if drop_non_numerical:

            self.temp_df.drop(
                drop_non_numerical,
                axis=axis,
                inplace=True
            )

            columns = [
                col for col in columns if col not in drop_non_numerical
            ]

            # columns = self.temp_df.columns.tolist()

        self.non_numeric_df = None
        self.numeric_df = None
        non_numeric_columns = []
        numeric_columns = []

        if len(columns) != 0:

            # checks if there are columns categorized as numerical or non numerical

            if non_numerical:
                non_numeric_columns = non_numerical
            else:
                non_numeric_columns = self.temp_df.select_dtypes(
                    include=np.object).columns.tolist()

            # checks if there are columns categorized as numerical or non numerical

            if numerical:
                numeric_columns = numerical
            else:
                numeric_columns = self.temp_df.select_dtypes(
                    include=np.number).columns.tolist()

            if len(non_numeric_columns) != 0:

                self.non_numeric_df = self.temp_df[non_numeric_columns]

                # drop non numeric columns. They will be handled separately and be merged with the main dataframe once handled.

                self.temp_df.drop(
                    non_numeric_columns,
                    axis="columns",
                    inplace=True
                )

                # null or nan values for non numeric values will be filed by using their mode

                self.non_numeric_df = self.non_numeric_df.fillna(
                    self.non_numeric_df
                    .mode()
                    .iloc[0]
                )

                # non_numeric_columns = self.temp_df.select_dtypes(include=np.object).columns.tolist()

            if len(numeric_columns) != 0:

                self.numeric_df = self.temp_df[numeric_columns]

                self.numeric_df.replace('?', missing_values, inplace=True)

                # we are using a model based imputation technique using ``SimpleImputedr`` from the sklearn.impute module.
                imp = SimpleImputer(
                    missing_values=missing_values, strategy=strategy)

                idf = pd.DataFrame(imp.fit_transform(self.numeric_df))
                idf.columns = self.numeric_df.columns
                idf.index = self.numeric_df.index
                self.numeric_df = idf

                # roundeed to 2 decimal places.
                self.numeric_df = idf.round(decimals=2)

                # merge the two data frames containing numerical and non numerical columns into one single dataframe.
                if self.non_numeric_df is not None:

                    self.numeric_df = self.numeric_df.join(self.non_numeric_df)
                    temp_cols = self.numeric_df.columns.tolist()
                    cols = [col for col in self.df.columns.tolist()
                            if col not in temp_cols]
                    self.temp_df = self.df[cols].join(self.numeric_df)

                else:

                    temp_cols = self.numeric_df.columns.tolist()
                    cols = [col for col in self.df.columns.tolist()
                            if col not in temp_cols]
                    self.temp_df = self.df[cols].join(self.numeric_df)

            else:

                if len(numeric_columns) == 0 and len(non_numeric_columns) != 0:
                    temp_cols = [col for col in self.df.columns.tolist(
                    ) if col not in non_numeric_columns]
                    self.temp_df = self.df[temp_cols].join(self.non_numeric_df)

        else:
            print("=================PyDataCleaner====================")
            print("All columns have been droped")
            print("=================PyDataCleaner====================")

        if persist:
            self.df = self.temp_df.clone_df()
        return self

    def persist(self):
        print("===========PyEhealth===============")
        print("Persisting")
        self.loader.df = self.temp_df
        print("Done!")
        print("===========PyEhealth===============")
