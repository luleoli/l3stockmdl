
import requests
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

not_modeling_variables = ['date', 'stock', 'date_refreshed', 'dif', 'target',
                          'open', 'high', 'low', 'close', 'volume']

def get_data(ticker, api_key='8JDWE75B6RS73XYH'):
    # Get data from api
    # ticker: str, stock ticker 'PETR4.SA.SAO'
    # api_key: str, api key to access the data


    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full'
    data = requests.get(url).json()

    time_series = data['Time Series (Daily)']
    meta_data = data['Meta Data']

    data, open_price, high, low, close, volume, stock, date_refreshed = [], [], [], [], [], [], [], []

    for date in time_series.keys():
        data.append(date)
        open_price.append(time_series[date]['1. open'])
        high.append(time_series[date]['2. high'])
        low.append(time_series[date]['3. low'])
        close.append(time_series[date]['4. close'])
        volume.append(time_series[date]['5. volume'])
        stock.append(meta_data['2. Symbol'][:5])
        date_refreshed.append(meta_data['3. Last Refreshed'])

    df = pd.DataFrame(data={'date': data, 
                            'open': open_price,
                            'high': high,
                            'low': low,
                            'close': close, 
                            'volume': volume, 
                            'stock': stock, 
                            'date_refreshed': date_refreshed})
    
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    df.to_csv(f'./{stock[0]}_{current_date}.csv', index=False)
    
    return df, meta_data



def modeling(df):
    X = df[[columns for columns in df.columns if columns not in not_modeling_variables]]
    y = df['target']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid for randomized search
    param_grid = {
        'num_leaves': [20, 30, 50, 70, 100],
        'n_estimators': [10, 20, 100, 200, 500],
        'max_depth': [3, 5, 12, 24, 48, 64, 128],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'boosting_type': ['gbdt', 'dart', 'goss']
    }
    
    # Create the LightGBM regressor
    lgb_model = lgb.LGBMRegressor(verbose=-1)
    
    # Perform randomized search for best parameters
    randomized_search = RandomizedSearchCV(lgb_model, 
                                           param_grid, 
                                           n_iter=150, 
                                           scoring='neg_root_mean_squared_error', 
                                           cv=3, 
                                           random_state=42)
    
    randomized_search.fit(X_train, y_train)
    
    # Get the best parameters and score
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_
    
    # Train the model with the best parameters
    lgb_model = lgb.LGBMRegressor(**best_params, verbose=-1)
    lgb_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = lgb_model.predict(X_test)

    return lgb_model, best_params, best_score, y_pred, y_test


def stock_feature_prep(df):
    # Prepare the data to be used in the model
    # df: pd.DataFrame, data from the stock

    df = df.sort_values(by='date', ascending=False).head(360)
    
    df.sort_values(by='date', ascending=True, inplace=True)

    for column in ['open', 'high', 'low', 'close', 'volume']:
        df[column] = df[column].apply(float)

    for column in ['open', 'high', 'low', 'close', 'volume']:
        for day in [1, 3, 7, 15, 30]:
            # Calculate moving average
            df[f'{column}_avg_{day}'] = df[column].rolling(window=day).mean()

            # Calculate maximum from previous row
            df[f'{column}_max_{day}'] = df[column].rolling(window=day).max()

            # Calculate minimum from previous row
            df[f'{column}_min_{day}'] = df[column].rolling(window=day).min()

            # Calculate standard deviation from previous row
            df[f'{column}_std_{day}'] = df[column].rolling(window=day).std()

            # Calculate the value based on the lag
            df[f'{column}_lag_{day}'] = df[column].shift(day)
            
    df['date'] = pd.to_datetime(df['date'])

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    week = df['date'].dt.isocalendar().week.astype(df['date'].dt.day.dtype) if hasattr(df['date'].dt, 'isocalendar') else df['date'].dt.week

    for n in attr: df[n] = getattr(df['date'].dt, n.lower()) if n != 'Week' else week

    df['target'] = df['close'].shift(-1)

    return df
