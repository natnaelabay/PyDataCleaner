from py_loader import PyLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


class CategoricalEncoder:
    
    def __init__(self, loader: PyLoader) -> None:
        self.loader = loader
        self.temp_df = loader.df.clone_df()

    def detect_categorical_cols(self):
        '''
        This member function trys to detect categorical features present
        in the dataframe.

        params:
        -------

            None.

        returns
        -------
            self or pd.DataFrame
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
