from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def ordinal_encode(df, columns):
    """
    Perform ordinal encoding on specified columns of a dataframe.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        columns (list): A list of column names to be encoded.
    
    Returns:
        pandas.DataFrame: The dataframe with ordinal encoded columns.
    """
    
    # Initialize the ordinal encoder
    encoder = OrdinalEncoder()
    
    # Iterate over the specified columns
    for col in columns:
        # Extract the column values
        col_values = df[col].values.reshape(-1, 1)
        
        # Perform ordinal encoding
        encoded_values = encoder.fit_transform(col_values)
        
        # Replace the original column with the encoded values
        df[col] = encoded_values.flatten()
    
    return df

# one hot encode the categorical features
def one_hot_encode(data, cat_cols=None):
    '''Returns the data with categorical features one hot encoded.'''
    
    # one hot encode the categorical features
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True, dtype='int', dummy_na=True)
    
    # return the data
    return data