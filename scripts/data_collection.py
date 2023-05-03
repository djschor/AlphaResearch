import pandas as pd
import yfinance as yf

def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves stock data for a given ticker symbol and date range using Yahoo Finance API.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol of the stock to retrieve data for.
    start_date : str
        The start date of the date range to retrieve data for in 'YYYY-MM-DD' format.
    end_date : str
        The end date of the date range to retrieve data for in 'YYYY-MM-DD' format.
    
    Returns:
    --------
    data : pd.DataFrame
        A Pandas DataFrame containing the stock data for the given date range.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
    return data
