import pandas as pd
import numpy as np
import yfinance as yf


def get_market_data(tickers, start_date, end_date):
    """
    Retrieves historical stock data for a list of tickers within a specified date range.

    Parameters:
    -----------
    tickers : list of str
        List of stock tickers to retrieve data for.
    start_date : str
        Start date of data retrieval period in the format 'YYYY-MM-DD'.
    end_date : str
        End date of data retrieval period in the format 'YYYY-MM-DD'.

    Returns:
    --------
    data : pd.DataFrame
        DataFrame containing the historical stock data for the specified tickers within the specified date range.
    """
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    data.dropna(inplace=True)
    return data


def preprocess_market_data(data):
    """
    Preprocesses the raw historical stock data.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the raw historical stock data.

    Returns:
    --------
    preprocessed_data : pd.DataFrame
        DataFrame containing the preprocessed stock data.
    """
    # Create a new column for daily returns
    preprocessed_data = data.copy()
    preprocessed_data['daily_returns'] = preprocessed_data['Close'].pct_change()

    # Compute the logarithmic returns
    preprocessed_data['log_returns'] = np.log(1 + preprocessed_data['daily_returns'])

    # Compute the rolling averages of the closing prices
    preprocessed_data['rolling_5d_avg'] = preprocessed_data['Close'].rolling(window=5).mean()
    preprocessed_data['rolling_10d_avg'] = preprocessed_data['Close'].rolling(window=10).mean()
    preprocessed_data['rolling_20d_avg'] = pre
