import traceback
import pandas as pd
import pyreadstat as ps

class PyLoader:
    def __init__(self, path,columns_list= None, the_first = None, the_last= None) -> None:
        '''
        It helps load a dataset of type different types, with a fredom of how the 
        dataset should be loaded. It is used as a parameter to the other 
        utility funcitons and/or classes for cleaning.

        It also comes with a few utilty functions. (Check the API Doc for details)

        Parameters
        ----------
        path: str
            The path to the location of the dataset
            .. You should include the right directory where the dataset exist!
            
        columns_list: list of string, default ``None``, optional
            It help us to load specific list of columns optionally. 
        
        the_first : int, default ``None``, optional
            Help to load the first n number of(head) observation/rows of the dataset.
            
        the_last : int, default ``None``, optional
            Help to load the last n number (tail) of observation/rows of the dataset.
        
        Examples
        --------
        self.path = "../ET_2016_DHS_11082019_632_141211/ETBR71SV/BR.csv"
        >>>list_columns(path)
        >>>list_columns(path,columns_list = ['V001','V384D'])
        >>>list_columns(path,columns_list= ['V001','V384D'], the_first = 10)
        >>>list_columns(path,the_last = 22)
        '''

        self.path = path
        self.columns_list = columns_list
        self.the_first = the_first
        self.the_last = the_last
        self.data_frame = self.__load_dataset()
    
    def __load_dataset(self)->pd.DataFrame:
        try:
            if self.path.endswith('.csv') or self.path.endswith('.CSV') :
                if self.columns_list:
                    if self.the_first:
                        return pd.read_csv(self.path, usecols = self.columns_list).head(self.the_first)
                    if self.the_last : 
                        return pd.read_csv(self.path, usecols = self.columns_list).tail(self.the_last)
                    return pd.read_csv(self.path, usecols = self.columns_list)
                else:
                    if self.the_first:
                        return pd.read_csv(self.path).head(self.the_first)
                    if self.the_last : 
                        return pd.read_csv(self.path).tail(self.the_last)
                    return pd.read_csv(self.path)
            
            elif self.path.endswith('.DTA') or self.path.endswith('.dta'):
                if self.columns_list:
                    if self.the_first:
                        df,meta = ps.read_dta(self.path, usecols = self.columns_list)
                        return df.head(self.the_first)
                    if self.the_last : 
                        df,meta = ps.read_dta(self.path, usecols = self.columns_list)
                        return df.tail(self.the_last)
                    df,meta = ps.read_dta(self.path, usecols = self.columns_list)
                    return df
                else:
                    if self.the_first:
                        df,meta = ps.read_dta(self.path)
                        return df.head(self.the_first)
                    if self.the_last : 
                        df,meta = ps.read_dta(self.path)
                        return df.tail(self.the_last)
                    
                    df,meta = ps.read_dta(self.path)
                    return df
                
            elif self.path.endswith('.SAV') or self.path.endswith('.sav'):
                if self.columns_list:
                    if self.the_first:
                        df,meta = ps.read_sav(self.path, usecols = self.columns_list)
                        return df.head(self.the_first)
                    if self.the_last : 
                        df,meta = ps.read_sav(self.path, usecols = self.columns_list)
                        return df.tail(self.the_last)
                    df,meta = ps.read_sav(self.path, usecols = self.columns_list)
                    return df
                else:
                    if self.the_first:
                        df,meta = ps.read_sav(self.path)
                        return df.head(self.the_first)
                    if self.the_last : 
                        df,meta = ps.read_sav(self.path)
                        return df.tail(self.the_last)
                    
                    df,meta = ps.read_sav(self.path)
                    return df
        except FileNotFoundError:
            print("===============PyHealth=========================")
            print(f"file path : {self.path}")
            print("File Not found => Please provide the correct path to your dataset")
            print("===============PyHealth=========================")
                
    def clone_df(self) -> pd.DataFrame:
        return self.data_frame.copy()

    def export_to_csv(self, filename, sep=",",encoding='utf-8') -> bool:
        '''

        Exports dataframe to csv file.
        Parameters
        ----------
        filename: str
            File name to export the dataframe to.

        sep: str, default ``,``, optional
            Separator used to export data to csv, 
        
        encoding: str, default ``utf-8``
            Type of encoding used to encode the data from the dataframe.

        Returns:
        --------
        bool: `True` if it was successful, ``False`` otherwise

        '''

        try:
            self.data_frame.to_csv(filename, sep=sep, encoding=encoding)
            return True
        except:
            print("================PyEheath========================")
            print(traceback.print_exc())
            print("================PyEheath========================")
            return False
