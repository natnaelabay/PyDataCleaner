"""
This Module contain tools for reading different files(dataset) format .
"""
import csv
import pyreadstat as ps
import pandas as pd 
#import ijson
import json
from csv import reader

def list_columns(filename):
    """ 
    It read different dataset formats and return list of columns. 
    It helps us to see and select the required columns/attributes 
    before loading the whole datasets so that we can save loading time. 

    Parameters
    ----------
    filename: str
        Dataset name 
        .. You should include the right directory where the dataset exist!
    Returns
    -------
    List
        It return list of columns name in string format 
        
    Examples
    --------
    >>>list_columns( "../ETBR71SV/ETBR71DT/ETBR71FL.DTA")
    """   

    if filename.endswith('.csv') or filename.endswith('.CSV'):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            col_csv = next(reader)
        return col_csv
    
    elif filename.endswith('.DTA') or filename.endswith('.dta'):
        col_stata, meta = ps.read_sas7bdat(filename, metadataonly=True)
        return list(col_stata)
    
    elif filename.endswith('.sav') or filename.endswith('.SAV'):
        col_sav, meta = ps.read_sas7bdat(filename, metadataonly=True)
        return list(col_sav)
    else:
        return "Your file format type doesn't included to the package yet"

def load_data(filename, columns_list= None, the_first = None, the_last= None):
    """ 
    It read and load different dataset formats into DataFrame. 
    Also support optionally to load specified list of columns only.
    Additionaly help us to load the first or the last specified number
    of observation/rows of the dataset. 

    Parameters
    ----------
    filename: str
        Dataset name 
        .. You should include the right directory where the dataset exist!
        
    columns_list: list of string, default ``None``, optional
        It help us to load specific list of columns optionally. 
    
    the_first : int, default ``None``, optional
        Help to load the first n number of(head) observation/rows of the dataset.
        
    the_last : int, default ``None``, optional
        Help to load the last n number (tail) of observation/rows of the dataset.
    
    Returns
    -------
    ``DataFrame``
        It return the required DataFrame  
    Examples
    --------
    filename = "../ET_2016_DHS_11082019_632_141211/ETBR71SV/BR.csv"
    >>>list_columns(filename)
    >>>list_columns(filename,columns_list = ['V001','V384D'])
    >>>list_columns(filename,columns_list= ['V001','V384D'], the_first = 10)
    >>>list_columns(filename,the_last = 22)
    """   
    if filename.endswith('.csv') or filename.endswith('.CSV') :
        if columns_list:
            if the_first:
                return pd.read_csv(filename, usecols = columns_list).head(the_first)
            if the_last : 
                return pd.read_csv(filename, usecols = columns_list).tail(the_last)
            return pd.read_csv(filename, usecols = columns_list)
        else:
            if the_first:
                return pd.read_csv(filename).head(the_first)
            if the_last : 
                return pd.read_csv(filename).tail(the_last)
            return pd.read_csv(filename)
    
    elif filename.endswith('.DTA') or filename.endswith('.dta'):
        if columns_list:
            if the_first:
                df,meta = ps.read_dta(filename, usecols = columns_list)
                return df.head(the_first)
            if the_last : 
                df,meta = ps.read_dta(filename, usecols = columns_list)
                return df.tail(the_last)
            df,meta = ps.read_dta(filename, usecols = columns_list)
            return df
        else:
            if the_first:
                df,meta = ps.read_dta(filename)
                return df.head(the_first)
            if the_last : 
                df,meta = ps.read_dta(filename)
                return df.tail(the_last)
            
            df,meta = ps.read_dta(filename)
            return df
        
    elif filename.endswith('.SAV') or filename.endswith('.sav'):
        if columns_list:
            if the_first:
                df,meta = ps.read_sav(filename, usecols = columns_list)
                return df.head(the_first)
            if the_last : 
                df,meta = ps.read_sav(filename, usecols = columns_list)
                return df.tail(the_last)
            df,meta = ps.read_sav(filename, usecols = columns_list)
            return df
        else:
            if the_first:
                df,meta = ps.read_sav(filename)
                return df.head(the_first)
            if the_last : 
                df,meta = ps.read_sav(filename)
                return df.tail(the_last)
            
            df,meta = ps.read_sav(filename)
            return df