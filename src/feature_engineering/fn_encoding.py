from sklearn.preprocessing import OrdinalEncoder

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