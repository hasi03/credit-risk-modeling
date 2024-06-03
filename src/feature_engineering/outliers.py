from scipy.stats import zscore

def non_outliers_zscore(X, threshold=4):

    # Filterout continuous columns
    unique_counts = X.nunique()

    columns = unique_counts[unique_counts > 58].index.tolist() # 58 because ORGANIZATION_TYPE has 58 unique values

    # Calculate Z-scores for each data point
    z_scores = zscore(X[columns])
    # Identify outliers
    outliers = (z_scores > threshold) | (z_scores < -threshold)
    return X[(outliers == False).all(axis=1)]