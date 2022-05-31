from pydoc import doc
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from pydoc import doc
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class AutoCleaner:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
        self.df_stat = None
        self.cleaning_suggestions = None
        self.temp_df = None
        self.categorical_columns = None

    def describe_data(self, deep: bool = False) -> pd.DataFrame:
        '''
        This function will describe the dataframe and return a dataframe with the following columns:
        Column_Name, missing value percentage, missing values count, unique values count [use full for categorical data types]
        params:
            deep: bool -> if True, will return the full description of the dataframe
                       -> if False, will return a simplified description of the dataframe
        returns: pd.DataFrame
        '''

        if deep:
            return self.df.describe()

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
        display(pd.DataFrame(self.df_stat))

        return self

    def suggest_cleaning_options(self):
        '''
        This function will suggest cleaning options for the dataframe. 
        (NB: The suggestion is based on the most common patterns from many datasets and further analysis
        may be necessary based on your data) 

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

        self.cleaning_suggestions = pd.DataFrame({
            "Columns": cols_with_null,
            "Null Percentage": list(filter((0).__ne__, nan_percentage)),
            "Suggested Cleaning Method": imputation_techniques
        })

        display(self.cleaning_suggestions)
        return self

    def handle_missing_values(
            self,
            columns: list = None,
            strategy: str = "mean",
            drop_numerical: list = False,
            drop_non_numerical: list = False,
            axis: int = 1,
            missing_values=np.nan,
    ):
        '''
        This function will handle missing values in the dataframe.

        params:
            columns: list -> list of columns to be handled
            strategy: str -> imputation strategy
            drop_numerical: list -> list of columns to be dropped
            drop_non_numerical: list -> list of columns to be dropped
            axis: int -> axis to be used
            missing_values: np.nan -> value to be used for imputation
        returns: pd.DataFrame
        '''
        if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
            raise Exception("Invalid argument")
        elif strategy is None:
            strategy = "mean"
            
        if self.temp_df is None:
            self.temp_df = self.df.copy()

        if not columns:
            columns = self.temp_df.columns.tolist()

        if (drop_non_numerical and drop_numerical) and (len(drop_non_numerical) + len(drop_numerical)) == len(columns) == len(self.temp_df.columns):
            print("WARNING: Tried to drop both numerical and non-numerical columns")
#             return self.temp_df

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
            columns = [
                col for col in columns if col not in drop_non_numerical
            ]

            self.temp_df.drop(
                drop_non_numerical,
                axis=axis,
                inplace=True
            )
            columns = self.temp_df.columns.tolist()

        self.non_numeric_df = None
        if len(columns) != 0:
                            
            non_numeric_columns = self.temp_df.select_dtypes(
                include=np.object).columns.tolist()
            
            numeric_columns = self.temp_df.select_dtypes(
                include=np.number).columns.tolist()

            if len(non_numeric_columns) != 0:
                
                self.non_numeric_df = self.temp_df[non_numeric_columns]

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

                non_numeric_columns = self.temp_df.select_dtypes(
                    include=np.object).columns.tolist()

            if len(numeric_columns) != 0:
                self.temp_df.replace('?', missing_values, inplace=True)
                imp = SimpleImputer(missing_values=np.NaN, strategy=strategy)
                idf = pd.DataFrame(imp.fit_transform(self.temp_df))
                idf.columns = self.temp_df.columns
                idf.index = self.temp_df.index
                self.temp_df = idf
                self.temp_df = idf.round(decimals=2)
                if self.non_numeric_df is not None:
                    self.temp_df = self.temp_df.join(self.non_numeric_df)
            else:
                print(non_numeric_columns)
                if len(non_numeric_columns) != 0:
                    self.temp_df = self.non_numeric_df
        else:
            print("All columns have been droped")
        return self

    def plot_missing_values(self):
        '''
            plot missing values from 
        '''
        pass

    def auto_clean(self):
        '''
        This function will automate the cleaning process by finding nan values and removing
        and replacing them by applying the SimpleImputer function and identifying corelation
        between the columns.
        '''
        self.describe_data().suggest_cleaning_options().handle_missing_values(strategy="mean").commit().handle_outliers().detect_categorical_columns().encode_categorical_columns("one-hot").commit()
        self.df.to_csv("cleaned_data.csv", encoding="utf-8", sep=",")
        return self.df

    def handle_outliers(self,
                       lower_quantile
                       ) -> pd.DataFrame:
        '''
        features, label,multiplier=1.5
        '''
        # get both numerical and non-numerical columns
        if self.temp_df is None:
            self.temp_df = self.df.copy()

        numerical_columns = self.temp_df.select_dtypes(
            include=[np.number]).columns.values
        non_numeric_columns = self.temp_df.select_dtypes(
            include=[np.object]).columns.values

        lower = self.temp_df[numerical_columns].quantile(0.25)
        upper = self.temp_df[numerical_columns].quantile(0.75)

        # calculate IQR
        IQR = upper - lower
        # calculate the cut_off
        cut_off = 1.5 * IQR

        lower_bound = lower - cut_off
        upper_bound = upper + cut_off
        updated_outlier = self.temp_df[((self.temp_df[numerical_columns] > lower_bound) | (
            self.temp_df[numerical_columns] < upper_bound))]
        for col in non_numeric_columns:
            updated_outlier[col] = self.temp_df[col]

        return self

    def commit(self):
        self.df = self.temp_df
        return self

    def get_df(self) -> pd.DataFrame:
        return self.df

    def detect_categorical_columns(self, plot: bool = True):
        '''
        This function will detect categorical columns in the dataframe.
        '''
        temp = self.df.copy()
        numerical_columns = temp.select_dtypes(
            include=[np.number]).columns.values
        categorical_columns = []
        
        for col in temp.columns:
            if col not in numerical_columns:
                categorical_columns.append(col)
                temp[col] = temp[col].astype("category")

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
        display(temp[categorical_columns])

        return self

    def encode_categorical_columns(self, encoding_type, replace_map: dict = None):
        '''
        This function will encode categorical columns to numerical values.
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
                display(self.temp_df)

            else:
                for col in self.categorical_columns:
                    self.temp_df[col].replace(replace_map[col], inplace=True)
                display(self.temp_df)

        elif encoding_type == "one-hot":
            self.temp_df = pd.get_dummies(
                self.temp_df, columns=self.categorical_columns)

        elif encoding_type == "label":
            label_encoder = LabelEncoder()
            for col in self.categorical_columns:
                self.temp_df[col] = label_encoder.fit_transform(
                    self.temp_df[col])

            # self.temp_df = pd.get_dummies(
            #     self.temp_df, columns=self.categorical_columns, prefix=self.categorical_columns)

        else:
            print("Please choose a valid encoding type")
        return self
    