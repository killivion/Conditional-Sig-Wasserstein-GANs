import pandas as pd


def transform_returns_series(df_or_series):
    """
    Assumes df_or_series is either a pandas DataFrame with one column or a Series of returns.
    Returns a Series where the first value is 1 and subsequent values are computed as:
    y[i] = y[i-1]*(1 + return[i])
    """
    # If a DataFrame, select the first column
    if isinstance(df_or_series, pd.DataFrame):
        series = df_or_series.T.iloc[:, 0]
    else:
        series = df_or_series

    # Compute cumulative product of (1+ return) and shift it to get the desired effect:
    # For i>=1: y[i] = prod_{j=0}^{i-1}(1+ return[j]), and y[0]=1.
    y = (1 + series).cumprod().shift(1, fill_value=1)
    return y

if __name__ == '__main__':
    # Example usage:
    df = pd.DataFrame({"returns": [0.1, -0.05, 0.2]})
    transformed_series = transform_returns_series(df)
    print(transformed_series)