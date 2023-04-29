import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.stattools import adfuller

def calculate_alpha(data: pd.DataFrame) -> np.ndarray:
    """
    Calculates the alpha factor for a given dataset using a complex statistical model.

    Args:
        data (pd.DataFrame): The dataset to be analyzed.

    Returns:
        np.ndarray: An array of alpha factors.
    """
    relevant_data = data.filter(items=['column1', 'column2', 'column3'])
    time_series_data = relevant_data.rolling(window=10).mean().diff().shift(-1)
    normalized_data = time_series_data.div(time_series_data.abs().sum(axis=1), axis=0)
    model = RandomForestRegressor(n_estimators=100, max_depth=5)
    model.fit(normalized_data, data['target_variable'])
    alpha_factors = model.feature_importances_
    
    return alpha_factors


def calc_signal(data: pd.DataFrame) -> pd.Series:
    """
    Calculates the signal for each stock in the input data using a custom alpha factor.

    The custom alpha factor is derived using a combination of statistical techniques 
    including linear regression, statistical arbitrage, and mean reversion.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input data containing the necessary fields for calculating the alpha factor. 
        Columns should include 'open', 'high', 'low', 'close', 'volume', and any other 
        necessary columns for the alpha factor.
        
    Returns:
    --------
    signal : pd.Series
        A series containing the calculated signal for each stock in the input data.
    """
    
    # Calculate the daily returns for each stock
    data['returns'] = data['close'].pct_change()
    
    # Compute the standard deviation of daily returns over a rolling 20-day window
    data['rolling_std'] = data['returns'].rolling(window=20).std()
    
    # Compute the z-score of the daily returns relative to their rolling standard deviation
    data['z_score'] = (data['returns'] - data['returns'].rolling(window=20).mean()) / data['rolling_std']
    
    # Remove any NaN values from the z-score column
    data.dropna(subset=['z_score'], inplace=True)
    
    # Create a binary signal that is 1 when the z-score is above 1 standard deviation and 0 otherwise
    data['signal'] = np.where(data['z_score'] > 1, 1, 0)
    
    # Compute the rolling 10-day mean of the signal
    data['rolling_signal_mean'] = data['signal'].rolling(window=10).mean()
    
    # Drop any NaN values from the rolling signal mean column
    data.dropna(subset=['rolling_signal_mean'], inplace=True)
    
    # Create a new dataframe containing the mean signal and the stock returns
    signal_df = data.groupby('symbol')['rolling_signal_mean'].mean().to_frame().join(data.groupby('symbol')['returns'].mean().to_frame())
    
    # Compute the slope of the linear regression between the mean signal and the stock returns
    X = signal_df['rolling_signal_mean'].values.reshape(-1, 1)
    y = signal_df['returns'].values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)
    slope = lr.coef_[0][0]
    
    # Check for stationarity in the residuals of the linear regression
    residuals = y - lr.predict(X)
    adf_result = adfuller(residuals)
    if adf_result[0] > adf_result[4]['5%']:
        # If the residuals are not stationary, apply a first-order difference to the mean signal 
        # and recompute the slope and residuals
        signal_df['rolling_signal_mean_diff'] = signal_df['rolling_signal_mean'].diff()
        signal_df.dropna(subset=['rolling_signal_mean_diff'], inplace=True)
        X = signal_df['rolling_signal_mean_diff'].values.reshape(-1, 1)
        y = signal_df['returns'].values.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, y)
        slope = lr.coef_[0][0]
        residuals = y - lr.predict(X)

    # Compute the z-score of the residuals
    residuals_std = np.std(residuals)
    residuals_z_score = (residuals - np.mean(residuals)) / residuals_std

    # Create a new binary signal that is 1 when the residuals z-score is below -1 standard deviation and 0 otherwise
    signal_df['residual_signal'] = np.where(residuals_z_score < -1, 1, 0)

    # Compute the rolling 10-day mean of the residual signal
    signal_df['rolling_residual_signal_mean'] = signal_df['residual_signal'].rolling(window=10).mean()

    # Create a final signal that is the product of the original signal and the residual signal
    signal_df['final_signal'] = signal_df['rolling_signal_mean'] * signal_df['rolling_residual_signal_mean']

    # Create a series containing the final signal for each stock in the input data
    signal = signal_df['final_signal']

    return signal

def rank_stocks(signal: pd.Series) -> pd.Series:
    """
    Ranks the stocks based on the calculated signal.
    
    Parameters:
    -----------
    signal : pd.Series
        The calculated signal for each stock.
        
    Returns:
    --------
    rank : pd.Series
        A series containing the rank of each stock based on the signal.
    """
    rank = signal.rank(ascending=False)
    return rank

def calculate_factor_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the pairwise correlations between a set of alpha factors.
    
    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the historical data for a set of alpha factors.
        
    Returns
    -------
    pd.DataFrame
        A correlation matrix showing the pairwise correlations between the factors.
    """
    return data.corr()

def calculate_factor_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the pairwise correlations between a set of alpha factors.
    
    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the historical data for a set of alpha factors.
        
    Returns
    -------
    pd.DataFrame
        A correlation matrix showing the pairwise correlations between the factors.
    """
    return data.corr()


def run_alpha_research(data: pd.DataFrame) -> pd.Series:
    """
    Driver function that runs the alpha research pipeline.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input data containing the necessary fields for calculating
        the alpha factor. Columns should include 'open', 'high', 'low',
        'close', 'volume', and any other necessary columns for the alpha
        factor.
        
    Returns:
    --------
    rank : pd.Series
        A series containing the rank of each stock based on the calculated
        signal.
    """
    signal = calc_signal(data)
    rank = rank_stocks(signal)
    return rank

def load_data(file_path):
    """
    Loads data from a CSV file.

    Args:
    file_path (str): The path to the CSV file containing the data.

    Returns:
    pandas.DataFrame: A DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def calculate_sma_factor(data, window):
    """
    Calculates the simple moving average (SMA) factor for the given data.

    Args:
    data (pandas.DataFrame): The data to calculate the SMA factor for.
    window (int): The number of periods to use in the SMA calculation.

    Returns:
    pandas.Series: A Series containing the calculated SMA factor values.
    """
    sma = data['Close'].rolling(window=window).mean()
    return (data['Close'] - sma) / sma


def calculate_ema_factor(data, window):
    """
    Calculates the exponential moving average (EMA) factor for the given data.

    Args:
    data (pandas.DataFrame): The data to calculate the EMA factor for.
    window (int): The number of periods to use in the EMA calculation.

    Returns:
    pandas.Series: A Series containing the calculated EMA factor values.
    """
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    return (data['Close'] - ema) / ema


def calculate_rsi_factor(data, window):
    """
    Calculates the relative strength index (RSI) factor for the given data.

    Args:
    data (pandas.DataFrame): The data to calculate the RSI factor for.
    window (int): The number of periods to use in the RSI calculation.

    Returns:
    pandas.Series: A Series containing the calculated RSI factor values.
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_roe_factor(data, window):
    """
    Calculates the return on equity (ROE) factor for the given data.

    Args:
    data (pandas.DataFrame): The data to calculate the ROE factor for.
    window (int): The number of periods to use in the ROE calculation.

    Returns:
    pandas.Series: A Series containing the calculated ROE factor values.
    """
    net_income = data['Net Income'].rolling(window=window).mean()
    shareholder_equity = data['Shareholder Equity'].rolling(window=window).mean()
    roe = net_income / shareholder_equity
    return roe


def calculate_pe_ratio_factor(data, window):
    """
    Calculates the price-to-earnings (PE) ratio factor for the given data.

    Args:
    data (pandas.DataFrame): The data to calculate the PE ratio factor for.
    window (int): The number of periods to use in the PE ratio calculation.

    Returns:
    pandas.Series: A Series containing the calculated PE ratio factor values.
    """
    price = data['Close']
    earnings = data['Earnings'].rolling(window=window).mean()
    pe_ratio = price / earnings
    return pe_ratio

def combine_factors(factor_dict, weights):
    """
    Combine individual factor data frames into a single composite factor data frame.

    Args:
        factor_dict (dict): A dictionary where keys are factor names and values are the corresponding data frames.
        weights (list): A list of weights for each factor in factor_dict.

    Returns:
        A data frame containing the composite factor with the same index as the input factors.
    """
    # Check that factor_dict and weights have the same length
    if len(factor_dict) != len(weights):
        raise ValueError("Length of factor_dict and weights must be the same.")

    # Combine the factors into a single data frame
    combined_df = pd.concat(factor_dict.values(), axis=1)

    # Normalize the data frame by z-score
    combined_df = (combined_df - combined_df.mean()) / combined_df.std()

    # Multiply each factor by its corresponding weight
    weighted_df = combined_df.multiply(weights)

    # Sum up the weighted factors along axis 1 to get the composite factor
    composite_factor = weighted_df.sum(axis=1)

    return composite_factor


def run_pipeline(file_path: str, sma_window: int = 20, ema_window: int = 20, rsi_window: int = 14) -> pd.Series:
    """
    Runs the end-to-end pipeline to generate alpha rankings.

    Args:
        file_path (str): The path to the CSV file containing the data.
        sma_window (int): The number of periods to use in the SMA calculation. Defaults to 20.
        ema_window (int): The number of periods to use in the EMA calculation. Defaults to 20.
        rsi_window (int): The number of periods to use in the RSI calculation. Defaults to 14.

    Returns:
        pd.Series: A series containing the rank of each stock based on the calculated signal.
    """
    # Load data
    data = load_data(file_path)

    # Calculate SMA factor
    sma_factor = calculate_sma_factor(data, sma_window)

    # Calculate EMA factor
    ema_factor = calculate_ema_factor(data, ema_window)

    # Calculate RSI factor
    rsi_factor = calculate_rsi_factor(data, rsi_window)

    # Combine factors
    alpha_factors = combine_factors(sma_factor, ema_factor, rsi_factor)

    # Calculate signal
    signal = calc_signal(data, alpha_factors)

    # Rank stocks
    rank = rank_stocks(signal)

    return rank
