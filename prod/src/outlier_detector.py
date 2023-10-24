

from mimetypes import init

import numpy as np
from .py_loader import PyLoader
import numpy as np
import pandas as pd

class OutlierDetector:
    def __init__(self, loader: PyLoader) -> None:
        self.loader = loader
        self.temp_df = loader.df.clone_df()

    def handle_outlier(self,lower_quantile, non_numerical, numerical):
        '''
        This member function trys to detect outliers in dataset and handle it for us.

        params
        ------

        non_numerica: list
            a list of non numerical columns.
        
        numerical: list
            a list of numerical columns.

        returns
        -------
        pd.DataFrame or self.
        
        '''
        
        non_numeric_columns = []
        numerical_columns = []
        
        # get both numerical and non-numerical columns
        
        if self.temp_df is None:
            self.temp_df = self.loader.df.copy()
        
        if non_numerical:
            non_numeric_columns = non_numerical
        else:
            non_numeric_columns = self.temp_df.select_dtypes(
                    include=np.object).columns.tolist()

        # checks if there are columns categorized as numerical or non numerical
        
        if numerical:
            numerical_columns = numerical
        else:
            numerical_columns = self.temp_df.select_dtypes(
                    include=np.number).columns.tolist()
            
        lower = self.temp_df[numerical_columns].quantile(0.25)
        upper = self.temp_df[numerical_columns].quantile(0.75)

        # calculate IQR
        IQR = upper - lower
        # calculate the cut_off
        cut_off = 1.5 * IQR

        lower_bound = lower - cut_off
        upper_bound = upper + cut_off
        
        self.temp_df = self.temp_df[((self.temp_df[numerical_columns] > lower_bound) | (
            self.temp_df[numerical_columns] < upper_bound))]

        # for col in non_numeric_columns:
        #     updated_outlier[col] = self.temp_df[col]

        return self.temp_df

    def perrsist(self):
        if self.temp_df:
            self.loader.df = self.temp_df.copy()
        return self.loader.df