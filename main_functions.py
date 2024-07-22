# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:03:38 2024

@author: apod7
"""

import numpy as np
import robin_stocks as r
import pandas as pd
# import credentials
import polars as pl
import pandas as pd
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import math
import csv
import os
import json
import pickle 
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
# from sklearn.preprocessing import nMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sqlalchemy import create_engine, select, column, case, func, text, desc, Integer, Float
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta, date

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from itertools import combinations
from scipy.optimize import minimize
from sqlalchemy import select, column, func, desc, text
from sqlalchemy import Float
from sqlalchemy import create_engine, text, column, select, case, func, Integer, desc
from sqlalchemy.types import Integer
from sqlalchemy.sql import select, case, func
from sqlalchemy import func, column
from sqlalchemy.types import Numeric
from sqlalchemy import create_engine
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
from dateutil.relativedelta import relativedelta


from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()
RH_Login = os.getenv('RH_Login')
RH_Pass = os.getenv('RH_Pass')
GMAIL_ACCT = os.getenv('GMAIL_ACCT')
GMAIL_PASS = os.getenv('GMAIL_PASS')
from pmdarima import auto_arima
 # engine = create_engine('sqlite:///database.db')
from joblib import dump, load


def calculate_sector_metrics(historical_df):
    market_returns = historical_df[historical_df['Symbol'] == 'SPY'].set_index('Week')['Close Price'].pct_change().dropna()
    
    sector_metrics = []
    
    for sector in historical_df['sector'].unique():
        sector_data = historical_df[historical_df['sector'] == sector]
        sector_returns = sector_data.groupby('Week')['Close Price'].mean().pct_change().dropna()
        
        # Align sector_returns and market_returns
        aligned_returns = pd.concat([sector_returns, market_returns], axis=1, join='inner')
        aligned_returns.columns = ['sector_return', 'market_return']
        
        # Calculate rolling beta (using the whole period up to each date)
        rolling_cov = aligned_returns['sector_return'].rolling(window=len(aligned_returns), min_periods=30).cov(aligned_returns['market_return'])
        rolling_var = aligned_returns['market_return'].rolling(window=len(aligned_returns), min_periods=30).var()
        beta = rolling_cov / rolling_var
        
        # Calculate daily alpha
        risk_free_rate = 0.03 / 252  # Assuming 3% annual risk-free rate
        alpha = aligned_returns['sector_return'] - (risk_free_rate + beta * (aligned_returns['market_return'] - risk_free_rate))
        
        sector_metrics.append(pd.DataFrame({
            'sector': sector,
            'date': alpha.index,
            'sector_beta': beta,  # Changed from 'beta' to 'sector_beta'
            'sector_alpha': alpha  # Changed from 'alpha' to 'sector_alpha'
        }))
    
    return pd.concat(sector_metrics, ignore_index=True)



# Calculate sector metrics


def sec_optimize_portfolio(sec_sector_metrics, objective='min_beta', min_sectors=3):
    def portfolio_metrics(weights):
        beta = np.dot(weights, sec_sector_metrics['beta'])
        alpha = np.dot(weights, sec_sector_metrics['alpha'])
        return beta, alpha

    def objective_function(weights):
        beta, alpha = portfolio_metrics(weights)
        if objective == 'min_beta':
            return beta
        elif objective == 'max_alpha':
            return -alpha

    n_sectors = len(sec_sector_metrics)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: np.sum(x > 0.01) - min_sectors}
    ]
    bounds = [(0, 1) for _ in range(n_sectors)]
    
    result = minimize(objective_function, np.ones(n_sectors) / n_sectors, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

# 7.18 - moved over from prepare_data_functions to remove that file from flow

def calculate_sector_volume_proportions(historical_df):
    # Convert 'Week' to datetime if it's not already
    historical_df['Week'] = pd.to_datetime(historical_df['Week'])
    
    # Calculate total volume for each week
    total_volume = historical_df.groupby('Week')['volume'].sum().reset_index()
    
    sector_proportions = pd.DataFrame()
    sector_proportions['Week'] = historical_df['Week'].unique()
    sector_proportions.set_index('Week', inplace=True)
    
    for sector in historical_df['sector'].unique():
        col_name = f'sector_volume_prop_{sector}'
        sector_data = historical_df[historical_df['sector'] == sector]
        
        # Calculate sector volume for each week
        sector_volume = sector_data.groupby('Week')['volume'].sum().reset_index()
        
        # Merge sector volume with total volume
        merged_volume = pd.merge(sector_volume, total_volume, on='Week', suffixes=('_sector', '_total'))
        
        # Calculate proportion
        merged_volume['proportion'] = merged_volume['volume_sector'] / merged_volume['volume_total']
        
        # Add the proportion to the sector_proportions DataFrame
        sector_proportions[col_name] = merged_volume.set_index('Week')['proportion']
    
    # Fill NaN values using forward fill and backward fill
    sector_proportions = sector_proportions.ffill().bfill()
    
    # If there are still any NaN values, fill them with 0
    sector_proportions = sector_proportions.fillna(0)
    
    # Reset the index to make 'Week' a column again
    sector_proportions = sector_proportions.reset_index()
    
    return sector_proportions


# original before 7.8
# def calculate_sector_volume_proportions(historical_df):
#     historical_df['Trading_Volume_Dollar'] = historical_df['volume'] * historical_df['Close Price']
#     total_volume = historical_df.groupby('Week')['Trading_Volume_Dollar'].sum()
    
#     sector_proportions = historical_df.groupby(['Week', 'sector'])['Trading_Volume_Dollar'].sum().unstack()
#     sector_proportions = sector_proportions.div(total_volume, axis=0)
    
#     return sector_proportions


# original (before 7/8)
# def calculate_sector_price_change(historical_df):
#     sector_price_models = {}
    
#     for sector in historical_df['sector'].unique():
#         sector_data = historical_df[historical_df['sector'] == sector].groupby('Week')['Close Price'].mean()
#         sector_returns = sector_data.pct_change().dropna()
        
#         # Fit exponential smoothing model
#         es_model = ExponentialSmoothing(sector_returns, trend='add', seasonal='add', seasonal_periods=5).fit()
        
#         # Fit polynomial regression model
#         X = np.arange(len(sector_returns)).reshape(-1, 1)
#         poly_features = PolynomialFeatures(degree=5, include_bias=False)
#         X_poly = poly_features.fit_transform(X)
#         poly_model = LinearRegression().fit(X_poly, sector_returns)
        
#         sector_price_models[sector] = {
#             'es_model': es_model,
#             'poly_model': poly_model,
#             'poly_features': poly_features
#         }
    
#     return sector_price_models

# 7/8 version - outdated already
# def calculate_sector_price_change(historical_df):
#     sector_price_models = {}
    
#     for sector in historical_df['sector'].unique():
#         sector_data = historical_df[historical_df['sector'] == sector].groupby('Week')['Close Price'].mean()
#         sector_returns = sector_data.pct_change().dropna()
        
#         # Reset index and create a numeric index
#         sector_returns = sector_returns.reset_index(drop=True)
        
#         # Fit exponential smoothing model
#         try:
#             es_model = ExponentialSmoothing(sector_returns, trend='add', seasonal='add', seasonal_periods=5).fit()
#         except Exception as e:
#             print(f"Error fitting exponential smoothing model for {sector}: {str(e)}")
#             es_model = None
        
#         # Fit polynomial regression model
#         X = np.arange(len(sector_returns)).reshape(-1, 1)
#         poly_features = PolynomialFeatures(degree=5, include_bias=False)
#         X_poly = poly_features.fit_transform(X)
#         poly_model = LinearRegression().fit(X_poly, sector_returns)
        
#         sector_price_models[sector] = {
#             'es_model': es_model,
#             'poly_model': poly_model,
#             'poly_features': poly_features
#         }
    
#     return sector_price_models


# 7.18 - moved over from prpare_data_functions
def fit_sector_volume_models(sector_volume_proportions):
    sector_volume_models = {}
    
    for sector in sector_volume_proportions.columns:
        if sector == 'Week':
            continue
        
        print(f"Fitting model for sector: {sector}")
        
        # Extract the sector data and drop NaN values
        sector_data = sector_volume_proportions[['Week', sector]].dropna()
        
        if sector_data.empty:
            print(f"No data available for sector: {sector}")
            sector_volume_models[sector] = None
            continue
        
        # Set 'Week' as index and sort
        sector_data = sector_data.set_index('Week').sort_index()
        
        # Create numeric index
        sector_data['numeric_index'] = range(len(sector_data))
        
        try:
            # Fit exponential smoothing model
            es_model = ExponentialSmoothing(sector_data[sector], trend='add', seasonal='add', seasonal_periods=7)
            es_fit = es_model.fit()
            
            # Fit polynomial regression model
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(sector_data[['numeric_index']])
            poly_model = LinearRegression()
            poly_model.fit(X_poly, sector_data[sector])
            
            sector_volume_models[sector] = {
                'es_model': es_fit,
                'poly_features': poly_features,
                'poly_model': poly_model
            }
        except Exception as e:
            print(f"Error fitting models for sector {sector}: {str(e)}")
            sector_volume_models[sector] = None
    
    return sector_volume_models


# original (before 7.8)
# def fit_sector_volume_models(sector_volume_proportions):
#     sector_volume_models = {}
    
#     for sector in sector_volume_proportions.columns:
#         sector_data = sector_volume_proportions[sector].dropna()
        
#         # Fit exponential smoothing model
#         es_model = ExponentialSmoothing(sector_data, trend='add', seasonal='add', seasonal_periods=5).fit()
        
#         # Fit polynomial regression model
#         X = np.arange(len(sector_data)).reshape(-1, 1)
#         poly_features = PolynomialFeatures(degree=5, include_bias=False)
#         X_poly = poly_features.fit_transform(X)
#         poly_model = LinearRegression().fit(X_poly, sector_data)
        
#         sector_volume_models[sector] = {
#             'es_model': es_model,
#             'poly_model': poly_model,
#             'poly_features': poly_features
#         }
    
#     return sector_volume_models


# # 7.8.24 version - already might be outdated..
# def fit_sector_volume_models(sector_volume_proportions):
#     sector_volume_models = {}
    
#     for sector in sector_volume_proportions.columns:
#         sector_data = sector_volume_proportions[sector].dropna()
        
#         # Reset index and create a numeric index
#         sector_data = sector_data.reset_index(drop=True)
        
#         # Fit exponential smoothing model
#         try:
#             es_model = ExponentialSmoothing(sector_data, trend='add', seasonal='add', seasonal_periods=5).fit()
#         except Exception as e:
#             print(f"Error fitting exponential smoothing model for {sector}: {str(e)}")
#             es_model = None
        
#         # Fit polynomial regression model
#         X = np.arange(len(sector_data)).reshape(-1, 1)
#         poly_features = PolynomialFeatures(degree=5, include_bias=False)
#         X_poly = poly_features.fit_transform(X)
#         poly_model = LinearRegression().fit(X_poly, sector_data)
        
#         sector_volume_models[sector] = {
#             'es_model': es_model,
#             'poly_model': poly_model,
#             'poly_features': poly_features
#         }
    
#     return sector_volume_models

# 7.11 - depreciated
# 7.8 version 
# def fit_sector_volume_models(sector_volume_proportions):
#     sector_volume_models = {}
    
#     # Filter out non-sector volume proportion columns
#     sector_columns = [col for col in sector_volume_proportions.columns if col.startswith('sector_volume_prop_')]
    
#     for sector in sector_columns:
#         print(f"Fitting model for sector: {sector}")
        
#         # Extract the sector data and drop NaN values
#         sector_data = sector_volume_proportions[['Week', sector]].dropna()
        
#         if sector_data.empty:
#             print(f"No data available for sector: {sector}")
#             sector_volume_models[sector] = None
#             continue
        
#         # Set 'Week' as index and sort
#         sector_data = sector_data.set_index('Week').sort_index()
        
#         # Create numeric index
#         sector_data['numeric_index'] = range(len(sector_data))
        
#         try:
#             # Fit exponential smoothing model
#             es_model = ExponentialSmoothing(sector_data[sector], trend='add', seasonal='add', seasonal_periods=7)
#             es_fit = es_model.fit(optimized=True)
            
#             # Fit polynomial regression model
#             poly_features = PolynomialFeatures(degree=2)
#             X_poly = poly_features.fit_transform(sector_data[['numeric_index']])
#             poly_model = LinearRegression()
#             poly_model.fit(X_poly, sector_data[sector])
            
#             sector_volume_models[sector] = {
#                 'es_model': es_fit,
#                 'poly_features': poly_features,
#                 'poly_model': poly_model
#             }
#         except Exception as e:
#             print(f"Error fitting models for sector {sector}: {str(e)}")
#             sector_volume_models[sector] = None
    
#     return sector_volume_models


# 7.8 version 
def add_sector_and_industry_info(historical_df, all_stock_symbols_df):
    # Ensure 'symbol' column names match
    historical_df = historical_df.rename(columns={'Symbol': 'symbol'})
    all_stock_symbols_df = all_stock_symbols_df.rename(columns={'symbol': 'symbol'})
    
    # Remove duplicates from all_stock_symbols_df, keeping the first occurrence
    all_stock_symbols_df = all_stock_symbols_df.drop_duplicates(subset='symbol', keep='first')
    
    # Merge sector and industry information
    historical_df = historical_df.merge(all_stock_symbols_df[['symbol', 'sector', 'industry']], 
                                        on='symbol', 
                                        how='left',
                                        validate='many_to_one')
    
    # Fill any missing sectors or industries with 'Unknown'
    historical_df['sector'] = historical_df['sector'].fillna('Unknown')
    historical_df['industry'] = historical_df['industry'].fillna('Unknown')
    
    # Rename 'symbol' back to 'Symbol' if needed
    historical_df = historical_df.rename(columns={'symbol': 'Symbol'})
    
    return historical_df


# 7.11 - depreciated ??
# 7.8 new
# def calculate_sector_volume_proportions(historical_df):
#     # Convert 'Week' to datetime if it's not already
#     historical_df['Week'] = pd.to_datetime(historical_df['Week'])
    
#     # Calculate total volume for each week
#     total_volume = historical_df.groupby('Week')['volume'].sum().reset_index()
    
#     sector_proportions = pd.DataFrame()
#     sector_proportions['Week'] = historical_df['Week'].unique()
#     sector_proportions.set_index('Week', inplace=True)
    
#     for sector in historical_df['sector'].unique():
#         col_name = f'sector_volume_prop_{sector}'
#         sector_data = historical_df[historical_df['sector'] == sector]
        
#         # Calculate sector volume for each week
#         sector_volume = sector_data.groupby('Week')['volume'].sum().reset_index()
        
#         # Merge sector volume with total volume
#         merged_volume = pd.merge(sector_volume, total_volume, on='Week', suffixes=('_sector', '_total'))
        
#         # Calculate proportion
#         merged_volume['proportion'] = merged_volume['volume_sector'] / merged_volume['volume_total']
        
#         # Add the proportion to the sector_proportions DataFrame
#         sector_proportions[col_name] = merged_volume.set_index('Week')['proportion']
    
#     # Fill NaN values using forward fill and backward fill
#     sector_proportions = sector_proportions.ffill().bfill()
    
#     # If there are still any NaN values, fill them with 0
#     sector_proportions = sector_proportions.fillna(0)
    
#     # Reset the index to make 'Week' a column again
#     sector_proportions = sector_proportions.reset_index()
    
#     return sector_proportions

# 7.8 new
def calculate_sector_price_change(historical_df):
    sector_price_models = {}
    
    print("Columns in historical_df:")
    print(historical_df.columns)
    
    for sector in historical_df['sector'].unique():
        col_name = f'sector_price_prop_{sector}'
        if col_name not in historical_df.columns:
            print(f"Creating column {col_name}")
            historical_df[col_name] = historical_df.loc[historical_df['sector'] == sector, 'Close Price'] / historical_df.groupby('Week')['Close Price'].transform('sum')
        
        sector_data = historical_df[historical_df['sector'] == sector]
        
        # Convert 'Week' to datetime if it's not already
        sector_data['Week'] = pd.to_datetime(sector_data['Week'])
        
        # Sort the data by date
        sector_data = sector_data.sort_values('Week')
        
        # Calculate price changes
        sector_returns = sector_data.groupby('Week')['Close Price'].mean().pct_change().dropna()
        
        print(f"Sector: {sector}")
        print(f"Price changes dtype: {sector_returns.dtype}")
        print(f"Price changes head: {sector_returns.head()}")
        print(f"Price changes contains NaN: {sector_returns.isna().any()}")
        
        # Reset index and create a numeric index
        sector_returns = sector_returns.reset_index(drop=True)
        
        # Fit exponential smoothing model
        try:
            es_model = ExponentialSmoothing(sector_returns, trend='add', seasonal='add', seasonal_periods=5).fit()
        except Exception as e:
            print(f"Error fitting exponential smoothing model for {sector}: {str(e)}")
            es_model = None
        
        # Fit polynomial regression model
        X = np.arange(len(sector_returns)).reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=5, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression().fit(X_poly, sector_returns)
        
        sector_price_models[sector] = {
            'es_model': es_model,
            'poly_model': poly_model,
            'poly_features': poly_features
        }
    
    return sector_price_models






def add_market_cap(historical_df, all_stock_symbols_df):
    # Ensure 'symbol' column names match
    historical_df = historical_df.rename(columns={'Symbol': 'symbol'})
    all_stock_symbols_df = all_stock_symbols_df.rename(columns={'symbol': 'symbol'})
    
    # Remove duplicates from all_stock_symbols_df, keeping the first occurrence
    all_stock_symbols_df = all_stock_symbols_df.drop_duplicates(subset='symbol', keep='first')
    
    # Diagnostics before merge
    print("Number of unique symbols in all_stock_symbols_df:", all_stock_symbols_df['symbol'].nunique())
    print("Total number of rows in all_stock_symbols_df:", len(all_stock_symbols_df))
    
    # Merge market_cap information
    historical_df = historical_df.merge(all_stock_symbols_df[['symbol', 'market_cap']], 
                                        on='symbol', 
                                        how='left',
                                        validate='many_to_one')
    
    # Rename the column to 'Market_Cap'
    historical_df = historical_df.rename(columns={'market_cap': 'Market_Cap'})
    
    # Rename 'symbol' back to 'Symbol' if needed
    historical_df = historical_df.rename(columns={'symbol': 'Symbol'})
    
    # Diagnostics after merge
    print("Number of rows in historical_df after merge:", len(historical_df))
    print("Number of unique symbols in historical_df after merge:", historical_df['Symbol'].nunique())
    print("Columns in historical_df after merge:", historical_df.columns)
    
    return historical_df


def remove_invalid_features(df, features_to_process):
    # Ensure all features exist in the DataFrame
    valid_features = [feature for feature in features_to_process if feature in df.columns]

    print("Features not found in the DataFrame:")
    print(set(features_to_process) - set(valid_features))

    # Remove features with all missing values or single value
    features_to_remove = []
    for feature in valid_features:
        if df[feature].isnull().all():
            features_to_remove.append(feature)
            print(f"Removing {feature}: All values are missing")
        elif df[feature].nunique() == 1:
            features_to_remove.append(feature)
            print(f"Removing {feature}: Single value {df[feature].iloc[0]}")

    valid_features = [feature for feature in valid_features if feature not in features_to_remove]

    print("\nFeatures removed due to all missing values or single value:")
    print(features_to_remove)

    return valid_features


def analyze_features(df, features_to_process):
    analysis_results = []

    for feature in features_to_process:
        if feature not in df.columns:
            analysis_results.append({
                'Feature': feature,
                'Status': 'Missing',
                'Unique Values': 0,
                'Min': None,
                'Max': None
            })
        else:
            feature_data = df[feature]
            if isinstance(feature_data, pd.DataFrame):
                print(f"Warning: {feature} is a DataFrame, not a Series. Using first column.")
                feature_data = feature_data.iloc[:, 0]
            
            unique_values = feature_data.nunique()
            
            if pd.api.types.is_numeric_dtype(feature_data):
                min_value = feature_data.min()
                max_value = feature_data.max()
            else:
                min_value = None
                max_value = None

            if unique_values == 0:
                status = 'All Null'
            elif unique_values == 1:
                status = 'Single Value'
            else:
                status = 'OK'

            analysis_results.append({
                'Feature': feature,
                'Status': status,
                'Unique Values': unique_values,
                'Min': min_value,
                'Max': max_value
            })

    results_df = pd.DataFrame(analysis_results)
    
    print("Features with issues:")
    print(results_df[results_df['Status'] != 'OK'])
    
    print("\nSummary:")
    print(results_df['Status'].value_counts())
    
    return results_df


def add_sector_info(historical_df, all_stock_symbols_df):
    # Ensure 'symbol' column names match
    historical_df = historical_df.rename(columns={'Symbol': 'symbol'})
    all_stock_symbols_df = all_stock_symbols_df.rename(columns={'symbol': 'symbol'})
    
    # Merge sector information
    historical_df = historical_df.merge(all_stock_symbols_df[['symbol', 'sector']], on='symbol', how='left')
    
    # Fill any missing sectors with 'Unknown'
    historical_df['sector'] = historical_df['sector'].fillna('Unknown')
    
    # Rename 'symbol' back to 'Symbol' if needed
    historical_df = historical_df.rename(columns={'symbol': 'Symbol'})
    
    return historical_df


def select_representative_stocks(df, max_stocks=200):
    # Group by sector and sort by market cap within each sector
    df_sorted = df.sort_values(['sector', 'market_cap'], ascending=[True, False])
    
    # Calculate the number of stocks to select from each sector
    sector_counts = df_sorted['sector'].value_counts()
    total_stocks = min(len(df), max_stocks)
    stocks_per_sector = (sector_counts / len(df) * total_stocks).round().astype(int)
    
    # Ensure we don't exceed max_stocks
    while stocks_per_sector.sum() > max_stocks:
        stocks_per_sector[stocks_per_sector.idxmax()] -= 1
    
    # Select top stocks from each sector
    selected_stocks = pd.DataFrame()
    for sector in sector_counts.index:
        sector_df = df_sorted[df_sorted['sector'] == sector]
        selected_stocks = pd.concat([selected_stocks, sector_df.head(stocks_per_sector[sector])])
    
    return selected_stocks



def fill_nan_values(df):
    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    
    return df



def sec_prepare_data(df):
    sec_df = df.copy()
    sec_df['Week'] = pd.to_datetime(sec_df['Week'])
    sec_df = sec_df.sort_values(['Symbol', 'Week'])
    
    # Fill NaN values
    numeric_columns = sec_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        sec_df[col] = sec_df[col].fillna(sec_df[col].median())
    
    categorical_columns = sec_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col == 'sector':
            sec_df[col] = sec_df[col].fillna('Unknown')
        else:
            sec_df[col] = sec_df[col].fillna(sec_df[col].mode().iloc[0])
    
    return sec_df


def calculate_ma_slopes(df, periods=[7, 14, 30]):
    for period in periods:
        df[f'MA_{period}'] = df['Close Price'].rolling(window=period).mean()
        df[f'Slope_MA_{period}'] = (df[f'MA_{period}'] - df[f'MA_{period}'].shift(1)) / df[f'MA_{period}'].shift(1)
    return df

# 7.8 changed this to be 1 std instead of 2 for spike definition (the other definition in main function uses 1.2x)
def calculate_volume_spike_features(df):
    df['Volume_Change'] = df['volume'].pct_change()
    df['Volume_Spike'] = (df['Volume_Change'] > df['Volume_Change'].rolling(window=30).mean() +  df['Volume_Change'].rolling(window=30).std()).astype(int) 
    df['Days_Since_Volume_Spike'] = df['Volume_Spike'].groupby(df['Symbol']).cumsum() - df['Volume_Spike'].groupby(df['Symbol']).cumsum().where(df['Volume_Spike'] == 1).ffill().fillna(0)
    return df



# 7.18 - moved over from prepare_data_functions

# 7.9 current version (newer than in main_functions)
def prepare_data(historical_df, filtered_df_categories, sector_metrics, sector_volume_proportions, sector_volume_models, sector_price_models):
    print("Shape of historical_df at start of prepare_data:", historical_df.shape)
    print("Columns in historical_df:", historical_df.columns)

    # Ensure 'Week' column is datetime type
    historical_df['Week'] = pd.to_datetime(historical_df['Week'])
    # Check for duplicates in 'Week' column
    print("Duplicate weeks in historical_df:", historical_df['Week'].duplicated().sum())

    # Merge sector metrics and volume proportions
    if not sector_metrics.empty:
        sector_metrics['date'] = pd.to_datetime(sector_metrics['date'])
        historical_df = historical_df.merge(
            sector_metrics, 
            left_on=['Week', 'sector'], 
            right_on=['date', 'sector'], 
            how='left'
        )
        historical_df = historical_df.drop('date', axis=1)
    
    print("Shape of sector_volume_proportions:", sector_volume_proportions.shape)
    print("Duplicate weeks in sector_volume_proportions:", sector_volume_proportions['Week'].duplicated().sum())

    if not sector_volume_proportions.empty:
        sector_volume_proportions['Week'] = pd.to_datetime(sector_volume_proportions['Week'])
        historical_df = historical_df.merge(
            sector_volume_proportions, 
            on='Week',
            how='left'
        )

    print("Shape after merging sector data:", historical_df.shape)
    print("Columns after merging sector data:", historical_df.columns)
 
    # Calculate average volume and Market Cap
    historical_df['Avg_Volume'] = historical_df.groupby('Symbol')['volume'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    historical_df['Market_Cap'] = historical_df['Close Price'] * historical_df['volume']
    
    # Identify volume spikes
    historical_df['Volume_Spike_day'] = historical_df['volume'] / historical_df['Avg_Volume']
    historical_df['Significant_Spike'] = (historical_df['Volume_Spike_day'] > 1.2).astype(int)

    # Create sector volume proportion columns
    # sectors = historical_df['sector'].unique()
    # for sector in sectors:
    #     col_name = f'sector_volume_prop_{sector}'
    #     if col_name not in historical_df.columns:
    #         historical_df[col_name] = 0
    #         historical_df.loc[historical_df['sector'] == sector, col_name] = historical_df.loc[historical_df['sector'] == sector, 'volume'] / historical_df.groupby('Week')['volume'].transform('sum')
# 7.9 new
    sectors = historical_df['sector'].unique()
    for sector in sectors:
        col_name = f'sector_volume_prop_{sector}'
        if col_name not in historical_df.columns:
            historical_df[col_name] = 0.0  # Initialize with float
            sector_mask = historical_df['sector'] == sector
            historical_df.loc[sector_mask, col_name] = (
                historical_df.loc[sector_mask, 'volume'].astype(float) / 
                historical_df.groupby('Week')['volume'].transform('sum').astype(float)
            ) 
    # Calculate expected sector trading volume change
    if sector_volume_models:
        for sector in historical_df['sector'].unique():
            col_name = f'sector_volume_prop_{sector}'
            if col_name in historical_df.columns:
                if sector in sector_volume_models and sector_volume_models[sector] is not None:
                    models = sector_volume_models[sector]
                    es_forecast = models['es_model'].forecast(1)
                    X_future = np.array([[len(historical_df)]])
                    X_poly_future = models['poly_features'].transform(X_future)
                    poly_forecast = models['poly_model'].predict(X_poly_future)
                    expected_change = (es_forecast.values[0] + poly_forecast[0]) / 2 - historical_df[col_name].iloc[-1]
                else:
                    ma_forecast = simple_moving_average_forecast(historical_df[col_name])
                    expected_change = ma_forecast - historical_df[col_name].iloc[-1]
                    print(f"Using fallback strategy for sector volume: {sector}")
                
                historical_df[f'expected_volume_change_{sector}'] = expected_change
            else:
                print(f"Warning: Column {col_name} not found. Skipping volume change calculation for {sector}.")

    # Calculate expected sector price changes
    if sector_price_models:
        for sector in historical_df['sector'].unique():
            col_name = f'sector_price_prop_{sector}'
            if col_name in historical_df.columns:
                if sector in sector_price_models and sector_price_models[sector] is not None:
                    models = sector_price_models[sector]
                    es_forecast = models['es_model'].forecast(1)
                    X_future = np.array([[len(historical_df)]])
                    X_poly_future = models['poly_features'].transform(X_future)
                    poly_forecast = models['poly_model'].predict(X_poly_future)
                    expected_change = (es_forecast.values[0] + poly_forecast[0]) / 2
                else:
                    ma_forecast = simple_moving_average_forecast(historical_df[col_name])
                    expected_change = ma_forecast
                    print(f"Using fallback strategy for sector price: {sector}")
                
                historical_df[f'expected_price_change_{sector}'] = expected_change
            else:
                print(f"Warning: Column {col_name} not found. Skipping price change calculation for {sector}.")

    # Calculate Sector and Industry Market Cap
    if 'Market_Cap' in historical_df.columns:
        historical_df['Sector_Market_Cap'] = historical_df.groupby(['Week', 'sector'])['Market_Cap'].transform('sum')
        historical_df['Industry_Market_Cap'] = historical_df.groupby(['Week', 'industry'])['Market_Cap'].transform('sum')
    else:
        print("Warning: 'Market_Cap' column not found. Skipping market cap calculations.")

    # Sort the data by Symbol and Week
    historical_df = historical_df.sort_values(['Symbol', 'Week'])

    # Find last significant spike
    def last_spike_price(group):
        significant_spikes = group[group['Significant_Spike'] == 1]
        if not significant_spikes.empty:
            return significant_spikes['Close Price'].iloc[-1]
        else:
            return group['Close Price'].iloc[0]

    historical_df['Last_Spike_Price'] = historical_df.groupby('Symbol').apply(last_spike_price).reset_index(level=0, drop=True)
    
    # Calculate strength index of volume spike
    historical_df['Spike_Strength'] = (historical_df['Volume_Spike_day'] - 1.2) / 0.2  # Normalized to 1 at 20% above average
    
    # Calculate weekly slopes for 3, 7, 15, 25 weeks
    for weeks in [3, 7, 15, 25]:
        historical_df[f'Slope_{weeks}w'] = historical_df.groupby('Symbol')['Close Price'].transform(lambda x: x.diff(weeks) / (weeks * 7))
        historical_df[f'Slope_Change_{weeks}w'] = historical_df.groupby('Symbol')[f'Slope_{weeks}w'].diff()
    
    # Calculate 14-day moving average
    historical_df['MA_14'] = historical_df.groupby('Symbol')['Close Price'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    
    # Calculate standard deviation for Skew variable
    historical_df['StdDev_30'] = historical_df.groupby('Symbol')['Close Price'].transform(lambda x: x.rolling(window=30, min_periods=1).std())
    
    # Calculate ExpLow and ExpHigh
    historical_df['ExpLow'] = historical_df['Close Price'].shift(1) + historical_df['MA_14'] - historical_df['StdDev_30']
    historical_df['ExpHigh'] = historical_df['Close Price'].shift(1) + historical_df['MA_14'] + historical_df['StdDev_30']
    
    # Calculate Skew
    historical_df['Skew'] = (100 * ((historical_df['Close Price'] - historical_df['ExpLow']) / (historical_df['ExpHigh'] - historical_df['ExpLow'])) + 100) / 100
    
    # Calculate Skew Averages and SkewIndex for different time periods
    for period in [1, 3, 5, 7, 15, 30, 60, 120, 180]:
        historical_df[f'SkewAvg_{period}d'] = historical_df.groupby('Symbol')['Skew'].transform(lambda x: x.rolling(window=period, min_periods=1).mean())
        historical_df[f'SkewIndex_{period}d'] = historical_df['Skew'] / historical_df[f'SkewAvg_{period}d']
    
    # Create outcome variables for different time periods
    for period in range(1, 15):  # 1 to 14 days
        historical_df[f'Future_Price_{period}d'] = historical_df.groupby('Symbol')['Close Price'].shift(-period)
        historical_df[f'Win_{period}d'] = (historical_df[f'Future_Price_{period}d'] > historical_df['Close Price']).astype(int)
        historical_df[f'Return_{period}d'] = (historical_df[f'Future_Price_{period}d'] - historical_df['Close Price']) / historical_df['Close Price']


# 7.12 let's just use Slope_MA_14 for Veered away calc
    historical_df = calculate_ma_slopes(historical_df)

    
    # Calculate slope of expected price trajectory to 14-day MA
    # historical_df['Slope_to_MA'] = (historical_df['MA_14'] - historical_df['Close Price']) / 14
    
    # Calculate deviation from MA_14
    historical_df['Deviation_from_MA'] = (historical_df['Close Price'] - historical_df['Slope_MA_14']) / historical_df['Slope_MA_14']
   
    # Identify periods when price veered away from MA_14
    historical_df['Veered_Away'] = (historical_df['Deviation_from_MA'].abs() > 0.05).astype(int)
   
    # Calculate days since last veer
    # 7.11 - fixed this to be 0 so that it find last time it was within range of MA
    historical_df['Days_Since_Veer'] = historical_df.groupby('Symbol')['Veered_Away'].transform(lambda x: x.cumsum() - x.cumsum().where(x == 0).ffill().fillna(0))



    # Add this before the Slope_to_MA calculation
    print("Days_Since_Veer data types:")
    print(historical_df['Days_Since_Veer'].dtype)
    print(historical_df['Days_Since_Veer'].value_counts().head())

    # Recalculate Slope_to_MA - new for 7/12
    # historical_df['Slope_to_MA'] = historical_df.groupby('Symbol').apply(calculate_slope_to_ma).reset_index(level=0, drop=True)
    # # historical_df['Slope_to_MA'] = historical_df.groupby('Symbol').apply(lambda x: (x['Close Price'].iloc[-1] - x['Close Price'].iloc[-x['Days_Since_Veer'].iloc[-1]]) / x['Days_Since_Veer'].iloc[-1] if x['Days_Since_Veer'].iloc[-1] > 0 else 0).reset_index(level=0, drop=True)

    historical_df = calculate_volume_spike_features(historical_df)

    # Add this after the Slope_to_MA calculation
    print("Slope_to_MA calculation complete")
    print(historical_df['Slope_MA_14'].describe())
    
    # Fill NaN values - new for 7.10 (from last good ROI version)
    numerical_columns = historical_df.select_dtypes(include=[np.number]).columns
    categorical_columns = historical_df.select_dtypes(include=['object']).columns
    
    for col in numerical_columns:
        historical_df[col] = historical_df[col].fillna(historical_df[col].median())
    
    for col in categorical_columns:
        historical_df[col] = historical_df[col].fillna(historical_df[col].mode().iloc[0])


    print("Final shape of historical_df:", historical_df.shape)
    return historical_df






#7.12 -  For very large models, consider using a database designed for large object storage, like MongoDB with GridFS.
def save_model(models, features, parameters, cap_size="Large"):
    """
    Save the models, features, and parameters to pickle files.

    Parameters:
    - models (dict): Dictionary containing trained models.
    - features (list or array): Features used for training.
    - parameters (dict): Additional parameters related to the models.
    - cap_size (str): Size designation for models (default is "Large").
    """

    # Get the current date
    today = date.today().strftime("%Y%m%d")

    # Create the directory path for the models folder
    path = r'C:\Users\apod7\StockPicker\models'
    models_dir = path

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Create the filename with version ID
    filename = f"trained_models_{cap_size}_{today}.pkl"
    filepath = os.path.join(models_dir, filename)

    # Create a dictionary that includes models, features, and parameters
    result = {
        'models': models,
        'features': features,
        'parameters': parameters
    }

    # Save the models, features, and parameters
    # 7.12 change - ran out of memory again
    # with open(filepath, 'wb') as f:
    #     pickle.dump(result, f)
    
    # 7.12 - stream the data to disk - still not working (memory)
    # with open(filepath, 'wb') as f:
    #     for item in [models, features, parameters]:
    #         pickle.dump(item, f)
    dump(result, filepath, compress=('gzip', 3))
    
    # Save additional model info
    model_info = {
        'cap_size': cap_size,
        'date_trained': today,
        'model_keys': list(models.keys()),
        'num_features': len(features)
    }

    info_filename = f"model_info_{cap_size}_{today}.pkl"
    info_filepath = os.path.join(models_dir, info_filename)

    with open(info_filepath, 'wb') as f:
        pickle.dump(model_info, f)

    print(f"Models, features, and parameters saved as: {filepath}")
    print(f"Model info saved as: {info_filepath}")

    # Optionally return the filepaths for further use
    return filepath, info_filepath


from joblib import load
import os

def load_model(filepath):
    """
    Load the models, features, and parameters from a saved file.

    Parameters:
    - filepath (str): Path to the saved model file.

    Returns:
    - models (dict): Dictionary containing trained models.
    - features (list): Features used for training.
    - parameters (dict): Additional parameters related to the models.
    """
    # Load the compressed data
    result = load(filepath)

    # Extract models, features, and parameters
    models = result['models']
    features = result['features']
    parameters = result['parameters']

    return models, features, parameters





def simple_moving_average_forecast(data, window=7):
    return data.rolling(window=window).mean().iloc[-1]





# 7.9 v1 - currently used
def create_market_cap_index(historical_df):
    # Convert 'Week' to datetime
    historical_df['Week'] = pd.to_datetime(historical_df['Week'])
    
    # Aggregate data to sector and industry level
    sector_data = historical_df.groupby(['Week', 'sector']).agg({
        'Market_Cap': 'sum',
        'volume': 'sum',
        'Close Price': 'mean'
    }).reset_index()
    
    industry_data = historical_df.groupby(['Week', 'industry']).agg({
        'Market_Cap': 'sum',
        'volume': 'sum',
        'Close Price': 'mean'
    }).reset_index()
    
    # Calculate changes
    for df in [sector_data, industry_data]:
        df['Market_Cap_Change'] = df.groupby(df.columns[1])['Market_Cap'].pct_change()
        df['Volume_Change'] = df.groupby(df.columns[1])['volume'].pct_change()
        df['Price_Change'] = df.groupby(df.columns[1])['Close Price'].pct_change()
    
    # Calculate moving averages
    for window in [5, 10, 30, 60]:
        sector_data[f'Sector_Market_Cap_MA_{window}'] = sector_data.groupby('sector')['Market_Cap_Change'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        industry_data[f'Industry_Market_Cap_MA_{window}'] = industry_data.groupby('industry')['Market_Cap_Change'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    
    def fit_model_and_forecast(series, periods=[5, 10, 15]):
        forecasts = {}
        if len(series) < max(periods):
            print(f"Warning: Not enough data for forecasting. Series length: {len(series)}")
            return {period: np.nan for period in periods}
        try:
            # Try ARIMA first
            model = ARIMA(series, order=(1,1,1))
            results = model.fit()
            for period in periods:
                forecast = results.forecast(steps=period)
                forecasts[period] = forecast.iloc[-1] if isinstance(forecast, pd.Series) else forecast[-1]
        except Exception as e:
            print(f"ARIMA failed: {e}. Falling back to exponential smoothing.")
            try:
                # Fallback to exponential smoothing
                model = ExponentialSmoothing(series)
                results = model.fit()
                for period in periods:
                    forecast = results.forecast(period)
                    forecasts[period] = forecast.iloc[-1] if isinstance(forecast, pd.Series) else forecast[-1]
            except Exception as e:
                print(f"Exponential smoothing failed: {e}. Using mean of series as forecast.")
                mean_value = series.mean()
                forecasts = {period: mean_value for period in periods}
        return forecasts
    
    # Calculate forecasts
    for df, group_col in [(sector_data, 'sector'), (industry_data, 'industry')]:
        for metric in ['Market_Cap_Change', 'Volume_Change', 'Price_Change']:
            for entity in df[group_col].unique():
                series = df[df[group_col] == entity][metric].dropna()
                if len(series) > 0:
                    forecasts = fit_model_and_forecast(series)
                    for period, forecast in forecasts.items():
                        df.loc[df[group_col] == entity, f'{metric}_Forecast_{period}d'] = forecast
    
    # Calculate Market Cap Indicators
    def categorize_forecast(forecast):
        if forecast < -0.05:
            return -2
        elif -0.05 <= forecast < -0.01:
            return -1
        elif -0.01 <= forecast <= 0.01:
            return 0
        elif 0.01 < forecast <= 0.05:
            return 1
        else:
            return 2

    sector_data['Sector_Market_Cap_Indicator'] = sector_data['Market_Cap_Change_Forecast_5d'].apply(categorize_forecast)
    industry_data['Industry_Market_Cap_Indicator'] = industry_data['Market_Cap_Change_Forecast_5d'].apply(categorize_forecast)
    
    # Merge forecasts back to historical_df
    historical_df = historical_df.merge(sector_data, on=['Week', 'sector'], how='left', suffixes=('', '_sector'))
    historical_df = historical_df.merge(industry_data, on=['Week', 'industry'], how='left', suffixes=('', '_industry'))
    
    # Calculate sector beta and alpha
    spy_data = historical_df[historical_df['Symbol'] == 'SPY']['Close Price'].pct_change().dropna()
    
    def calculate_beta_alpha(group):
        if len(group) < 2:
            return pd.Series({'sector_beta': np.nan, 'sector_alpha': np.nan})
        
        sector_returns = group['Close Price'].pct_change().dropna()
        if len(sector_returns) != len(spy_data):
            return pd.Series({'sector_beta': np.nan, 'sector_alpha': np.nan})
        
        covariance = np.cov(sector_returns, spy_data)[0][1]
        variance = np.var(spy_data)
        beta = covariance / variance
        alpha = np.mean(sector_returns) - beta * np.mean(spy_data)
        return pd.Series({'sector_beta': beta, 'sector_alpha': alpha})
    
    sector_beta_alpha = historical_df.groupby('sector').apply(calculate_beta_alpha).reset_index()
    historical_df = historical_df.merge(sector_beta_alpha, on='sector', how='left')
    
    return historical_df



    

# 7.10.24 - trying this WOE creation with max 10 bins for categorical - may be too slow to keep recalculating...
def calculate_woe_and_bins(df, feature, target='Win_3d', num_bins=20, special_values=None):
    print(f"Calculating WOE and bins for feature: {feature}")
    
    # Handle missing values
    df = df.copy()
    df[feature] = df[feature].fillna(np.nan)
    
    if df[feature].dtype in ['object', 'category']:
        woe_dict, bins = calculate_woe_categorical(df, feature, target, special_values)
    else:
        woe_dict, bins = calculate_woe_numeric(df, feature, target, num_bins, special_values)
    
    return woe_dict, bins

def calculate_woe_categorical(df, feature, target, special_values=None):
    if special_values is None:
        special_values = []

    # Separate special values
    special_df = df[df[feature].isin(special_values + [np.nan])].copy()
    regular_df = df[~df[feature].isin(special_values + [np.nan])].copy()

    # Group low-frequency categories
    value_counts = regular_df[feature].value_counts()
    top_categories = value_counts[value_counts.cumsum() <= 0.95 * len(regular_df)].index
    regular_df.loc[~regular_df[feature].isin(top_categories), feature] = 'Other'
    
    bins = regular_df[feature].unique()

    woe_dict = calculate_woe_for_bins(df, feature, target, bins, special_values)
    
    return woe_dict, bins

def calculate_woe_numeric(df, feature, target, num_bins, special_values=None):
    if special_values is None:
        special_values = []

    # Separate special values
    special_df = df[df[feature].isin(special_values + [np.nan, np.inf, -np.inf])].copy()
    regular_df = df[~df[feature].isin(special_values + [np.nan, np.inf, -np.inf])].copy()

    # Create bins
    try:
        bins = pd.qcut(regular_df[feature], q=num_bins, duplicates='drop')
        if isinstance(bins, pd.Series):
            bin_labels = bins.cat.categories
        else:
            bin_labels = bins
    except ValueError:
        # If qcut fails, try manual binning
        min_val = regular_df[feature].min()
        max_val = regular_df[feature].max()
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bins = pd.cut(regular_df[feature], bins=bin_edges, include_lowest=True)
        bin_labels = bins.cat.categories

    regular_df.loc[:, 'bins'] = bins

    woe_dict = calculate_woe_for_bins(df, feature, target, bins, special_values)
    
    # Add special values to woe_dict
    for val in special_values + [np.nan, np.inf, -np.inf]:
        special_data = special_df[special_df[feature] == val]
        if len(special_data) > 0:
            good = special_data[target].sum()
            bad = len(special_data) - good
            woe = safe_woe(good, bad, df[target].sum(), len(df) - df[target].sum())
            woe_dict[val] = woe

    return woe_dict, bin_labels

def safe_woe(good, bad, total_good, total_bad, epsilon=0.5):
    good_ratio = (good + epsilon) / (total_good + epsilon)
    bad_ratio = (bad + epsilon) / (total_bad + epsilon)
    return np.log(good_ratio / bad_ratio)

def calculate_woe_for_bins(df, feature, target, bins, special_values):
    woe_dict = {}
    total_good = df[target].sum()
    total_bad = len(df) - total_good

    if isinstance(bins, pd.Series) and hasattr(bins, 'cat'):
        bin_categories = bins.cat.categories
    else:
        bin_categories = bins

    for bin_val in bin_categories:
        if isinstance(bin_val, pd.Interval):
            bin_data = df[(df[feature] >= bin_val.left) & (df[feature] < bin_val.right)]
        else:
            bin_data = df[df[feature] == bin_val]
        
        good = bin_data[target].sum()
        bad = len(bin_data) - good
        woe = safe_woe(good, bad, total_good, total_bad)
        woe_dict[bin_val] = woe

    return woe_dict





# beginning of modeling 


def split_train_validate(df):
    # Get unique dates
    dates = df['Week'].unique()
    split_date = dates[int(len(dates) * 2/3)]
    
    train_df = df[df['Week'] <= split_date]
    validate_df = df[df['Week'] > split_date]
    
    return train_df, validate_df

def split_train_validate_oot(df):
    # Get unique dates
    dates = sorted(df['Week'].unique())
    
    # Calculate split points
    # first_split = int(len(dates) * 1/3)  # not able to allocate 8gb for this with 40 mos kept prior, need to reduce more for now (until more horsepower arrives!)
    # second_split = int(len(dates) * 2/3)

    first_split = int(len(dates) * 0.1)
    second_split = int(len(dates) * 9/10)
    
    # Split dates
    unused_dates = dates[:first_split]
    train_dates = dates[first_split:second_split]
    validate_dates = dates[second_split:second_split + (len(dates) - second_split) // 2]
    validate_oot_dates = dates[second_split + (len(dates) - second_split) // 2:]
    
    # Create DataFrames
    unused_df = df[df['Week'].isin(unused_dates)]
    train_df = df[df['Week'].isin(train_dates)]
    validate_df = df[df['Week'].isin(validate_dates)]
    validate_oot_df = df[df['Week'].isin(validate_oot_dates)]
    
    print(f"Unused data: {len(unused_dates)} weeks, {len(unused_df)} rows")
    print(f"Training data: {len(train_dates)} weeks, {len(train_df)} rows")
    print(f"Validation data: {len(validate_dates)} weeks, {len(validate_df)} rows")
    print(f"Out-of-time validation data: {len(validate_oot_dates)} weeks, {len(validate_oot_df)} rows")
    
    return unused_df, train_df, validate_df, validate_oot_df


def split_large_mid_small_cap(historical_df, holdings_df, filtered_df_categories, split_ratio_large=0.2, split_ratio_mid=0.5, total_stocks=1000):
    # Print column names for debugging
    print("Columns in filtered_df_categories:", filtered_df_categories.columns)
    print("Columns in historical_df:", historical_df.columns)

    # Adjust column names based on actual names in filtered_df_categories
    symbol_col = 'Symbol'
    sector_col = 'sector'
    market_cap_col = 'Total_Market_Cap_Trillion'

    # Merge historical_df with filtered_df_categories to get sector information
    merged_df = historical_df.merge(filtered_df_categories[[symbol_col, sector_col, market_cap_col]], 
                                    on='Symbol', how='left')
    
    print("Merged DataFrame columns:", merged_df.columns)
    print("Number of rows after merge:", len(merged_df))
    
    # Get SPY and holdings symbols
    spy_symbol = 'SPY'
    holdings_symbols = holdings_df['Ticker'].tolist()
    special_symbols = [spy_symbol] + holdings_symbols
    
    # Sort all stocks by market cap in descending order, excluding special symbols
    merged_df_sorted = merged_df[~merged_df['Symbol'].isin(special_symbols)].sort_values(market_cap_col, ascending=False)
    
    # Calculate the split points
    total_regular_stocks = min(len(merged_df_sorted), total_stocks - len(special_symbols))
    large_cap_split = int(total_regular_stocks * split_ratio_large)
    mid_cap_split = int(total_regular_stocks * (split_ratio_large + split_ratio_mid))
    
    print(f"Large cap split point: {large_cap_split}")
    print(f"Mid cap split point: {mid_cap_split}")
    
    # Split the dataframe
    large_cap_symbols = merged_df_sorted['Symbol'].unique()[:large_cap_split]
    mid_cap_symbols = merged_df_sorted['Symbol'].unique()[large_cap_split:mid_cap_split]
    small_cap_symbols = merged_df_sorted['Symbol'].unique()[mid_cap_split:total_regular_stocks]
    
    # Add special symbols to all cap sizes
    large_cap_symbols = np.concatenate([large_cap_symbols, special_symbols])
    mid_cap_symbols = np.concatenate([mid_cap_symbols, special_symbols])
    small_cap_symbols = np.concatenate([small_cap_symbols, special_symbols])
    
    large_cap_df = historical_df[historical_df['Symbol'].isin(large_cap_symbols)]
    mid_cap_df = historical_df[historical_df['Symbol'].isin(mid_cap_symbols)]
    small_cap_df = historical_df[historical_df['Symbol'].isin(small_cap_symbols)]
    
    print(f"Number of large cap symbols: {len(large_cap_symbols)}")
    print(f"Number of mid cap symbols: {len(mid_cap_symbols)}")
    print(f"Number of small cap symbols: {len(small_cap_symbols)}")
    
    # Ensure sector representation in all dataframes
    sectors = filtered_df_categories[sector_col].unique()
    for sector in sectors:
        sector_stocks = filtered_df_categories[filtered_df_categories[sector_col] == sector][symbol_col]
        
        for cap_df, cap_name in [(large_cap_df, "large"), (mid_cap_df, "mid"), (small_cap_df, "small")]:
            if not any(symbol in cap_df['Symbol'].values for symbol in sector_stocks):
                sector_cap = merged_df_sorted[merged_df_sorted['Symbol'].isin(sector_stocks)].nlargest(1, market_cap_col)
                cap_df = pd.concat([cap_df, historical_df[historical_df['Symbol'] == sector_cap['Symbol'].iloc[0]]])
                print(f"Added {sector_cap['Symbol'].iloc[0]} to {cap_name} cap for sector {sector}")
    
    print(f"Final number of large cap stocks: {len(large_cap_df['Symbol'].unique())}")
    print(f"Final number of mid cap stocks: {len(mid_cap_df['Symbol'].unique())}")
    print(f"Final number of small cap stocks: {len(small_cap_df['Symbol'].unique())}")
    
    # Generate and print statistics table
    stats_table = generate_cap_statistics(large_cap_df, mid_cap_df, small_cap_df, filtered_df_categories)
    print("\nMarket Cap Statistics:")
    print(stats_table)
    
    return large_cap_df, mid_cap_df, small_cap_df




def generate_cap_statistics(large_cap_df, mid_cap_df, small_cap_df, filtered_df_categories):
    def get_stats(df, cap_type):
        symbols = df['Symbol'].unique()
        market_caps = filtered_df_categories[filtered_df_categories['Symbol'].isin(symbols)]['Total_Market_Cap_Trillion']
        return {
            'Cap Type': cap_type,
            'Number of Symbols': len(symbols),
            'Total Market Cap': market_caps.sum(),
            'Min Market Cap': market_caps.min(),
            'Max Market Cap': market_caps.max()
        }

    large_cap_stats = get_stats(large_cap_df, 'Large Cap')
    mid_cap_stats = get_stats(mid_cap_df, 'Mid Cap')
    small_cap_stats = get_stats(small_cap_df, 'Small Cap')  # Add this line

    stats_df = pd.DataFrame([large_cap_stats, mid_cap_stats, small_cap_stats])  # Include small_cap_stats
    stats_df = stats_df.set_index('Cap Type')

    return stats_df


def print_cap_statistics(large_cap_df, mid_cap_df, filtered_df_categories):
    def get_sector_stats(df, filtered_df_categories):
        merged = df.merge(filtered_df_categories[['Symbol', 'sector', 'Total_Market_Cap_Trillion']], on='Symbol', how='left')
        stats = merged.groupby('sector').agg({
            'Symbol': 'nunique',
            'Total_Market_Cap_Trillion': 'sum'
        }).reset_index()
        stats.columns = ['Sector', 'Num_Symbols', 'Total_Market_Cap_Trillion']
        return stats

    large_cap_stats = get_sector_stats(large_cap_df, filtered_df_categories)
    mid_cap_stats = get_sector_stats(mid_cap_df, filtered_df_categories)

    # Combine stats
    combined_stats = large_cap_stats.merge(mid_cap_stats, on='Sector', suffixes=('_Large', '_Mid'))
    combined_stats['Total_Symbols'] = combined_stats['Num_Symbols_Large'] + combined_stats['Num_Symbols_Mid']
    combined_stats['Total_Market_Cap'] = combined_stats['Total_Market_Cap_Trillion_Large'] + combined_stats['Total_Market_Cap_Trillion_Mid']

    # Calculate totals
    totals = combined_stats.sum(numeric_only=True)
    totals['Sector'] = 'Total'
    combined_stats = combined_stats.append(totals, ignore_index=True)

    # Format the table
    combined_stats['Total_Market_Cap_Trillion_Large'] = combined_stats['Total_Market_Cap_Trillion_Large'].round(3)
    combined_stats['Total_Market_Cap_Trillion_Mid'] = combined_stats['Total_Market_Cap_Trillion_Mid'].round(3)
    combined_stats['Total_Market_Cap'] = combined_stats['Total_Market_Cap'].round(3)

    # Print the table
    print("\nMarket Cap Statistics by Sector:")
    print(combined_stats.to_string(index=False))

    # Calculate and print percentages
    total_symbols = combined_stats.loc[combined_stats['Sector'] == 'Total', 'Total_Symbols'].values[0]
    total_market_cap = combined_stats.loc[combined_stats['Sector'] == 'Total', 'Total_Market_Cap'].values[0]

    print("\nPercentages:")
    print(f"Large Cap Symbols: {combined_stats['Num_Symbols_Large'].sum() / total_symbols:.2%}")
    print(f"Mid Cap Symbols: {combined_stats['Num_Symbols_Mid'].sum() / total_symbols:.2%}")
    print(f"Large Cap Market Cap: {combined_stats['Total_Market_Cap_Trillion_Large'].sum() / total_market_cap:.2%}")
    print(f"Mid Cap Market Cap: {combined_stats['Total_Market_Cap_Trillion_Mid'].sum() / total_market_cap:.2%}")

    

# 7/10 version with categorical handling
def plot_features_vs_return(train_df, features, max_features_per_figure=20):
    num_features = len(features)
    num_figures = math.ceil(num_features / max_features_per_figure)
    
    for fig_num in range(num_figures):
        start_idx = fig_num * max_features_per_figure
        end_idx = min((fig_num + 1) * max_features_per_figure, num_features)
        current_features = features[start_idx:end_idx]
        
        num_rows = math.ceil(len(current_features) / 4)
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
        fig.suptitle(f'Features vs Return (Figure {fig_num + 1})', fontsize=16)
        
        for i, feature in enumerate(current_features):
            row = i // 4
            col = i % 4
            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            if train_df[feature].dtype in ['object', 'category']:
                # For categorical features, use a box plot
                sns.boxplot(x=feature, y='Return_3d', data=train_df, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                # For numerical features, use a scatter plot
                ax.scatter(train_df[feature], train_df['Return_3d'], alpha=0.5)
            
            ax.set_title(feature)
            ax.set_xlabel(feature)
            ax.set_ylabel('Return_3d') #7.11 updated all to 3 day instead of 1day charts
        
        plt.tight_layout()
        plt.show()
        



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os


def plot_features_vs_target(df_sample, features, plot_dir):
    today = datetime.now().strftime("%Y%m%d")
    
    for feature in features:
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f'Feature: {feature}', fontsize=16)
        
        periods = [1, 7, 14]
        
        for i, period in enumerate(periods):
            # Win plot (left column)
            win_target = f'Win_{period}d'
            df_sample[win_target] = df_sample[win_target].astype(int)  # Ensure it's 0 or 1
            
            if pd.api.types.is_numeric_dtype(df_sample[feature]):
                # For numeric features, use binning
                bins = pd.qcut(df_sample[feature], q=10, duplicates='drop')
                win_rates = df_sample.groupby(bins)[win_target].mean()
                win_rates.plot(kind='bar', ax=axes[i, 0])
                axes[i, 0].set_xlabel(feature)
                axes[i, 0].set_ylabel(f'Average Win Rate ({period}d)')
            else:
                # For categorical features, use groupby
                win_rates = df_sample.groupby(feature)[win_target].mean()
                win_rates.plot(kind='bar', ax=axes[i, 0])
                axes[i, 0].set_xlabel(feature)
                axes[i, 0].set_ylabel(f'Average Win Rate ({period}d)')
            
            axes[i, 0].set_title(f'{feature} vs {win_target}')
            axes[i, 0].tick_params(axis='x', rotation=45)
            
            # Return plot (right column)
            return_target = f'Return_{period}d'
            if pd.api.types.is_numeric_dtype(df_sample[feature]):
                sns.scatterplot(x=df_sample[feature], y=df_sample[return_target], ax=axes[i, 1])
            else:
                sns.boxplot(x=df_sample[feature], y=df_sample[return_target], ax=axes[i, 1])
            axes[i, 1].set_title(f'{feature} vs {return_target}')
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel(f'Return ({period}d)')
            axes[i, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{feature}_plot_{today}.png'))
        plt.close()

def reduce_variables(df, features_for_modeling, target='Win_7d', association_threshold=0.3, correlation_threshold=0.7, sample_size=0.1):
    # Sample 10% of the data
    df_sample = df.sample(frac=sample_size, random_state=42)
    
    # Ensure Win columns are boolean (0 or 1)
    for col in ['Win_1d', 'Win_7d', 'Win_14d']:
        df_sample[col] = df_sample[col].astype(int)
    
    def calculate_association(series, target):
        if pd.api.types.is_numeric_dtype(series):
            return np.abs(np.corrcoef(series, df_sample[target])[0, 1])
        else:
            contingency_table = pd.crosstab(series, df_sample[target])
            chi2, _, _, _ = stats.chi2_contingency(contingency_table)
            return np.sqrt(chi2 / (len(df_sample) * (min(contingency_table.shape) - 1)))

    # Calculate association with target for features in features_for_modeling
    associations = {}
    for column in features_for_modeling:
        if column != target and column not in ['Win_1d', 'Win_14d', 'Return_1d', 'Return_7d', 'Return_14d']:
            associations[column] = calculate_association(df_sample[column], target)

    # Sort features by association strength
    sorted_features = sorted(associations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Keep top X% of features
    num_features_to_keep = int(len(sorted_features) * (1 - association_threshold))
    selected_features = [feature for feature, _ in sorted_features[:num_features_to_keep]]

    # Separate numeric and categorical features
    numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(df_sample[f])]
    categorical_features = [f for f in selected_features if f not in numeric_features]

    # Calculate correlation matrix for numeric features
    correlation_matrix = df_sample[numeric_features].corr()

    # Remove highly correlated features
    final_modeling_features = []
    for i in range(len(numeric_features)):
        if numeric_features[i] not in final_modeling_features:
            correlated_features = correlation_matrix.index[correlation_matrix[numeric_features[i]].abs() > correlation_threshold].tolist()
            correlated_features = [feat for feat in correlated_features if feat != numeric_features[i] and feat in numeric_features]
            
            if correlated_features:
                best_feature = max([(feat, abs(associations[feat])) for feat in correlated_features + [numeric_features[i]]], key=lambda x: x[1])[0]
                final_modeling_features.append(best_feature)
            else:
                final_modeling_features.append(numeric_features[i])

    # Add categorical features to final list
    final_modeling_features.extend(categorical_features)

    # Create plots
    plot_dir = r"C:\Users\apod7\StockPicker\models\WOE_BIN"
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_features_vs_target(df_sample, final_modeling_features, plot_dir)

    # Save summary
    today = datetime.now().strftime("%Y%m%d")
    summary = pd.DataFrame({
        'Feature': final_modeling_features,
        'Association': [associations[feat] for feat in final_modeling_features]
    })
    summary.to_csv(os.path.join(plot_dir, f'variable_reduction_summary_{today}.csv'), index=False)

    return final_modeling_features



# version just before..
def reduce_variables(df, features_for_modeling, target='Win_7d', association_threshold=0.3, correlation_threshold=0.7, sample_size=0.1):
    # Sample 10% of the data
    df_sample = df.sample(frac=sample_size, random_state=42)
    
    def calculate_association(series, target):
        if pd.api.types.is_numeric_dtype(series):
            return stats.pointbiserialr(df_sample[target], series)[0]
        else:
            contingency_table = pd.crosstab(series, df_sample[target])
            return stats.chi2_contingency(contingency_table)[0] / len(df_sample)

    # Calculate association with target for features in features_for_modeling
    associations = {}
    for column in features_for_modeling:
        if column != target and column not in ['Win_1d', 'Win_14d', 'Return_1d', 'Return_7d', 'Return_14d']:
            associations[column] = calculate_association(df_sample[column], target)

    # Sort features by association strength
    sorted_features = sorted(associations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Keep top X% of features
    num_features_to_keep = int(len(sorted_features) * (1 - association_threshold))
    selected_features = [feature for feature, _ in sorted_features[:num_features_to_keep]]

    # Separate numeric and categorical features
    numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(df_sample[f])]
    categorical_features = [f for f in selected_features if f not in numeric_features]

    # Calculate correlation matrix for numeric features
    correlation_matrix = df_sample[numeric_features].corr()

    # Remove highly correlated features
    final_modeling_features = []
    for i in range(len(numeric_features)):
        if numeric_features[i] not in final_modeling_features:
            correlated_features = correlation_matrix.index[correlation_matrix[numeric_features[i]].abs() > correlation_threshold].tolist()
            correlated_features = [feat for feat in correlated_features if feat != numeric_features[i] and feat in numeric_features]
            
            if correlated_features:
                best_feature = max([(feat, abs(associations[feat])) for feat in correlated_features + [numeric_features[i]]], key=lambda x: x[1])[0]
                final_modeling_features.append(best_feature)
            else:
                final_modeling_features.append(numeric_features[i])

    # Add categorical features to final list
    final_modeling_features.extend(categorical_features)

    # Create plots
    today = datetime.now().strftime("%Y%m%d")
    plot_dir = r"C:\Users\apod7\StockPicker\models\WOE_BIN"
    os.makedirs(plot_dir, exist_ok=True)

    for feature in final_modeling_features:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Feature: {feature}', fontsize=16)

        for i, win_target in enumerate(['Win_1d', 'Win_7d', 'Win_14d']):
            if pd.api.types.is_numeric_dtype(df_sample[feature]):
                sns.boxplot(x=df_sample[win_target].astype('category'), y=df_sample[feature], ax=axes[0, i])
            else:
                sns.countplot(x=df_sample[feature], hue=df_sample[win_target].astype('category'), ax=axes[0, i])
            axes[0, i].set_title(f'{feature} vs {win_target}')
            if not pd.api.types.is_numeric_dtype(df_sample[feature]):
                axes[0, i].tick_params(axis='x', rotation=45)

        for i, return_target in enumerate(['Return_1d', 'Return_7d', 'Return_14d']):
            if pd.api.types.is_numeric_dtype(df_sample[feature]):
                sns.scatterplot(x=df_sample[return_target], y=df_sample[feature], ax=axes[1, i])
            else:
                sns.boxplot(x=df_sample[feature], y=df_sample[return_target], ax=axes[1, i])
            axes[1, i].set_title(f'{feature} vs {return_target}')
            if not pd.api.types.is_numeric_dtype(df_sample[feature]):
                axes[1, i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{feature}_plot_{today}.png'))
        plt.close()

    # Save summary
    summary = pd.DataFrame({
        'Feature': final_modeling_features,
        'Association': [associations[feat] for feat in final_modeling_features]
    })
    summary.to_csv(os.path.join(plot_dir, f'variable_reduction_summary_{today}.csv'), index=False)

    return final_modeling_features





# 7.12
def apply_woe_and_binning(df, woe_dicts, bin_dicts):
    df = df.copy()
    df = df.reset_index(drop=True)
    for feature in woe_dicts.keys():
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in the dataframe. Skipping.")
            continue
        
        if df[feature].isna().all():
            print(f"Warning: Feature '{feature}' contains all NaN values. Skipping.")
            continue
        
        print(f"Applying WOE and binning for feature: {feature}")
        
        # Check if the feature is categorical or if bins contain non-numeric values
        is_categorical = df[feature].dtype in ['object', 'category'] or any(not isinstance(x, (int, float)) for x in bin_dicts[feature])
        
        if is_categorical:
            # For categorical features
            df[f'{feature}_woe'] = df[feature].map(woe_dicts[feature]).fillna(0)
            df[f'{feature}_binned'] = df[feature].map(lambda x: x if x in bin_dicts[feature] else 'Other')
        else:
            # For numerical features
            df[f'{feature}_binned'] = pd.cut(df[feature], bins=bin_dicts[feature], labels=False, include_lowest=True)
            df[f'{feature}_woe'] = df[f'{feature}_binned'].map(lambda x: woe_dicts[feature][x] if x in woe_dicts[feature] else 0)
    
    return df





def add_missing_features(df, required_features):
    for feature in required_features:
        if feature not in df.columns:
            print(f"Adding missing feature: {feature}")
            df[feature] = 0  # or any other appropriate default value
    return df

def get_common_features(train_df, validate_df):
    train_features = set(train_df.columns)
    validate_features = set(validate_df.columns)
    common_features = list(train_features.intersection(validate_features))
    return common_features



# new version 
def save_woe_and_bins_to_csv_and_pickle(woe_dicts, bin_dicts):
    directory = r'C:\Users\apod7\StockPicker\models\WOE_BIN'
    os.makedirs(directory, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    for feature in woe_dicts.keys():
        woe_dict = woe_dicts[feature]
        bins = bin_dicts[feature]
        
        for key, value in woe_dict.items():
            csv_data.append({
                'variable': feature,
                'type': 'woe',
                'key': key,
                'value': value
            })
        
        for i, bin_edge in enumerate(bins):
            csv_data.append({
                'variable': feature,
                'type': 'bin',
                'key': i,
                'value': bin_edge
            })

    # Save as CSV
    csv_filepath = os.path.join(directory, 'woe_and_bins.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filepath, index=False)
    print(f"WOE and Bins saved to {csv_filepath}")
    
    # Save as pickle
    pkl_filepath = os.path.join(directory, 'woe_and_bins.pkl')
    with open(pkl_filepath, 'wb') as pklfile:
        pickle.dump({'woe_dicts': woe_dicts, 'bin_dicts': bin_dicts}, pklfile)
    print(f"WOE and Bins saved to {pkl_filepath}")
    
    
#7.4.24 new version that doesn't have woe and binning - done in separate section - now also prepare_features 

def prepare_features(df):
    df = df.copy()
    
    # Handle outliers
    columns_to_check = [col for col in df.columns if col.startswith(('SkewIndex_', 'Slope_', 'expected_volume_change_', 'expected_price_change_'))]
    df = handle_outliers(df, columns_to_check, method='clip')
    
    # Create binary and WOE variables
    features_to_bin = [col for col in df.columns if col.startswith(('SkewIndex_', 'Slope_', 'expected_volume_change_', 'expected_price_change_'))]
    
    for feature in features_to_bin:
        # Create binary version
        median = df[feature].median()
        df[f'{feature}_binary'] = (df[feature] > median).astype(int)
        
        # Create WOE version
        try:
            binned, _ = pd.qcut(df[feature], q=10, labels=False, retbins=True, duplicates='drop')
            woe_dict = calculate_woe(df, feature, target='Win_1d', num_bins=10, special_values=[0])
            df[f'{feature}_woe'] = binned.map(woe_dict)
        except Exception as e:
            print(f"Error calculating WOE for {feature}: {str(e)}")
            df[f'{feature}_woe'] = 0  # Assign a neutral value
    
    return df





import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 7.11 am version - WORKS!!!!
def train_models(train_df, existing_features):
    print("\n*** STATUS: Starting train_models function ***")
    print("*** STATUS: Preparing features list ***")
    
    # Separate numerical and categorical features
    numerical_features = [f for f in existing_features if train_df[f].dtype in ['int64', 'float64']]
    categorical_features = [f for f in existing_features if train_df[f].dtype in ['object', 'category']]
    
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Plot separately (assuming you have a plot_features_vs_return function)
    plot_features_vs_return(train_df, numerical_features, max_features_per_figure=10)
    plot_features_vs_return(train_df, categorical_features, max_features_per_figure=6)

    print("*** STATUS: Preparing feature matrix X ***")
    
    # Prepare the feature matrix
    X = train_df[numerical_features].copy()
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Apply pd.get_dummies only if categorical features are present
    if categorical_features:
        X_cat = pd.get_dummies(train_df[categorical_features], drop_first=True)
        X = pd.concat([X, X_cat], axis=1)
    
    # Remove constant columns
    X = X.loc[:, (X != X.iloc[0]).any()]
    
    # Remove zero-variance columns
    X = X.loc[:, X.std() != 0]

    # Standardize numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    print("*** STATUS: Training models ***")
    models = {}
    parameters = {}
    for days in range(1, 15):
        print(f"*** STATUS: Training models for {days} day(s) horizon ***")
        y_win = train_df[f'Win_{days}d']
        y_return = train_df[f'Return_{days}d']

        # Handle potential NaN or infinite values in target variables
        mask = ~(np.isnan(y_win) | np.isinf(y_win) | np.isnan(y_return) | np.isinf(y_return))
        X_filtered = X[mask]
        y_win_filtered = y_win[mask]
        y_return_filtered = y_return[mask]

        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10, 
                                     min_samples_split=3, min_samples_leaf=1, max_features='sqrt')
        clf.fit(X_filtered, y_win_filtered)
        models[f'Win_{days}d'] = clf
        parameters[f'Win_{days}d'] = clf.get_params()

        reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=None, 
                                    min_samples_split=3, min_samples_leaf=1, max_features=None)
        reg.fit(X_filtered, y_return_filtered)
        models[f'Return_{days}d'] = reg
        parameters[f'Return_{days}d'] = reg.get_params()

    print("*** STATUS: All models trained successfully ***")
    print("Models trained. Keys in models dictionary:", list(models.keys()))
    print("Features used for prediction:", list(X.columns))
    return models, list(X.columns), parameters


def predict_outcomes(df, models, train_features):
    df = df.copy()
    
    X = pd.get_dummies(df, drop_first=True)
    X = X.reindex(columns=train_features, fill_value=0)

    prediction_columns = {}
    for period in range(1, 15):
        win_key = f'Win_{period}d'
        return_key = f'Return_{period}d'
        
        if win_key in models:
            prediction_columns[f'P_Win_{period}d'] = models[win_key].predict_proba(X)[:, 1]
        else:
            print(f"Warning: Model for {win_key} not found. Skipping win probability prediction for this period.")
        
        if return_key in models:
            prediction_columns[f'P_Return_{period}d'] = models[return_key].predict(X)
        else:
            print(f"Warning: Model for {return_key} not found. Skipping return prediction for this period.")

    df = pd.concat([df, pd.DataFrame(prediction_columns, index=df.index)], axis=1)
    return df


def get_last_iteration():
    csv_path = r"C:\Users\apod7\StockPicker\models\model_library.csv"
    try:
        with open(csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)  # Skip the header row
            iteration_index = headers.index("Iteration")
            last_row = None
            for row in csvreader:
                last_row = row
            if last_row:
                return int(last_row[iteration_index])
            else:
                raise ValueError("CSV file is empty")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None



# 7.11 - new version resolved!!! 
# Now will need to paste and attach some good things to it....


# 7.15 - current versionintegrating both run-time and model building details, together with final result (stocks selected for a buy, current portolios stats)

def send_email(model_iteration, start_time, end_time, error_log, email_list, results_string, files_to_attach, future_date):
    import os
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.mime.base import MIMEBase
    from email import encoders
    import datetime
    
    import base64
    import os
    import glob
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    
    
    gmail_user = os.getenv('GMAIL_ACCT')
    gmail_password = os.getenv('GMAIL_PASS')
    
    future_date_formatted = future_date.strftime("%Y-%m-%d")
    start_time_formatted = start_time.strftime("%Y-%m-%d %H:%M")
    end_time_formatted = end_time.strftime("%Y-%m-%d %H:%M")
    sender_email = gmail_user
    recipient_email = email_list
    subject = f"Model run #{model_iteration} {'completed successfully' if not error_log else 'encountered errors'}"
    
    duration = end_time - start_time
    
    msg = MIMEMultipart()
    msg['From'] = f"ZF <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    error_log_str = '<br>'.join(error_log) if error_log else 'None'
    
    html_body = f"""
    <html>
      <body>
        <p>May the riches be with you..</p>
        <p>Alpha run for {future_date_formatted} complete.</p>
        <p>Model Iteration: {model_iteration}</p>
        <p>Start Time: {start_time_formatted}</p>
        <p>End Time: {end_time_formatted}</p>
        <p>Duration: {duration}</p>
        <p>Error Log:</p>
        <pre>{error_log_str}</pre>
        <p>Results:</p>
        <pre>{results_string}</pre>
        <p><img src="data:image/png;base64,{get_image_base64()}" alt="ZoltarSurf"></p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_body, 'html'))
    
    # Attach all files
    for file_path in files_to_attach:
        try:
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(file_path)}',
            )
            msg.attach(part)
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        print('Email sent successfully!')
    except Exception as e:
        print(f'Error sending email: {e}')
        
        
        

# 7.15 - version that sends more relevant info as an updated (but including more above so this one on hold)
# import os
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.base import MIMEBase
# from email import encoders
# from datetime import datetime
# import glob


def get_image_base64():
    import base64
    # Function to read and encode the image as base64
    image_path = r'C:\Users\apod7\StockPicker\docs\ZoltarSurf.png'
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
    return img_base64



def update_features_for_modeling(df, features_to_process, include_original=False):
    # Initialize the features_for_modeling list
    features_for_modeling = []

    # Append variables containing '_woe' and '_binary'
    for column in df.columns:
        if '_woe' in column or '_binned' in column:
            features_for_modeling.append(column)

    # Optionally include original features_to_process
    if include_original:
        features_for_modeling.extend([f for f in features_to_process if f in df.columns])

    # Remove duplicates while preserving order
    features_for_modeling = list(dict.fromkeys(features_for_modeling))

    print(f"Total features for modeling: {len(features_for_modeling)}")
    print("First 10 features:", features_for_modeling[:10])
    print("Last 10 features:", features_for_modeling[-10:])

    return features_for_modeling


def update_model_library(models, features, parameters, start_time, end_time, error_log):
    from datetime import datetime
    library_file = r'C:\Users\apod7\StockPicker\models\model_library.csv'
    fieldnames = ['Iteration', 'Date', 'Time', 'Number_of_Models', 'Number_of_Features', 'Time_to_Complete', 'Number_of_Errors', 'Error_Log', 'Model_Settings']

    # Get current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Calculate time to complete
    time_to_complete = str(end_time - start_time)

    # Count number of errors
    num_errors = len(error_log)

    # Extract model settings
    model_settings = json.dumps({
        'n_estimators': parameters['Win_1d'].get('n_estimators', ''),
        'max_depth': parameters['Win_1d'].get('max_depth', ''),
        'min_samples_split': parameters['Win_1d'].get('min_samples_split', ''),
        'min_samples_leaf': parameters['Win_1d'].get('min_samples_leaf', ''),
        'max_features': parameters['Win_1d'].get('max_features', '')
    })

    # Prepare new row data
    new_row = {
        'Iteration': 1,  # This will be updated if the file already exists and has data
        'Date': date,
        'Time': time,
        'Number_of_Models': len(models),
        'Number_of_Features': len(features),
        'Time_to_Complete': time_to_complete,
        'Number_of_Errors': num_errors,
        'Error_Log': '; '.join(error_log) if error_log else 'None',
        'Model_Settings': model_settings
    }

    # Check if file exists and read existing data
    if os.path.exists(library_file):
        with open(library_file, 'r') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        
        # Check for duplicates and update iteration number
        is_duplicate = False
        if existing_data:  # Only process if there's existing data
            for row in existing_data:
                if (row['Date'] == new_row['Date'] and 
                    row['Time'] == new_row['Time'] and 
                    row['Number_of_Models'] == str(new_row['Number_of_Models']) and 
                    row['Number_of_Features'] == str(new_row['Number_of_Features'])):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                new_row['Iteration'] = str(int(existing_data[-1]['Iteration']) + 1)
                existing_data.append(new_row)
        else:
            existing_data.append(new_row)
    else:
        existing_data = [new_row]

    # Write updated data to CSV
    with open(library_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)

    print(f"Model library updated: {library_file}")
    
    return int(new_row['Iteration'])  # Return the iteration number

def create_feature_parameter_table(iteration, features, models, X):
    table_file = r'C:\Users\apod7\StockPicker\models\feature_parameter_table.csv'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(table_file), exist_ok=True)

    # Read existing data if file exists
    if os.path.exists(table_file):
        with open(table_file, 'r') as f:
            reader = csv.reader(f)
            existing_data = list(reader)
        existing_features = [row[0] for row in existing_data[1:]]  # Skip header
    else:
        existing_data = [['Feature']]
        existing_features = []

    # Prepare new data
    new_data = []
    for feature in features:
        if feature not in existing_features:
            new_data.append([feature])

    print(f"Total number of models: {len(models)}")
    
    # Count feature usage across all models
    feature_usage = {feature: 0 for feature in features}
    for model_name, model in models.items():
        print(f"\nInspecting model: {model_name}")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for feature, importance in zip(features, importances):
                if importance > 0:
                    feature_usage[feature] += 1
        else:
            print(f"Warning: Model {model_name} does not have feature_importances_ attribute")

    # Add new iteration column
    iteration_column = [f'Iteration_{iteration}']
    for row in existing_data[1:]:
        feature = row[0]
        usage = feature_usage.get(feature, 0) if feature in features else ''
        iteration_column.append(str(usage))
    for row in new_data:
        feature = row[0]
        usage = feature_usage.get(feature, 0)
        row.append(str(usage))

    # Update existing data with new column
    existing_data[0].append(f'Iteration_{iteration}')
    for i in range(1, len(existing_data)):
        existing_data[i].append(iteration_column[i])

    # Combine existing and new data
    updated_data = existing_data + new_data

    # Write updated data to CSV
    with open(table_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_data)

    print(f"Feature and Parameter table updated: {table_file}")
    
    # Print out the feature usage for debugging
    print("\nFeature usage summary:")
    for feature, usage in feature_usage.items():
        print(f"{feature}: {usage}")


def create_feature_parameter_table_with_average(iteration, features, models, X):
    table_file = r'C:\Users\apod7\StockPicker\models\feature_parameter_table_average.csv'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(table_file), exist_ok=True)

    # Read existing data if file exists
    if os.path.exists(table_file):
        with open(table_file, 'r') as f:
            reader = csv.reader(f)
            existing_data = list(reader)
        existing_features = [row[0] for row in existing_data[1:]]  # Skip header
    else:
        existing_data = [['Feature']]
        existing_features = []

    # Prepare new data
    new_data = []
    for feature in features:
        if feature not in existing_features:
            new_data.append([feature])

    print(f"Total number of models: {len(models)}")
    
    # Calculate average feature importance across all models
    feature_importance_sum = {feature: 0 for feature in features}
    feature_importance_count = {feature: 0 for feature in features}
    for model_name, model in models.items():
        print(f"\nInspecting model: {model_name}")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for feature, importance in zip(features, importances):
                feature_importance_sum[feature] += importance
                feature_importance_count[feature] += 1
        else:
            print(f"Warning: Model {model_name} does not have feature_importances_ attribute")

    # Calculate average feature importance
    feature_importance_avg = {feature: feature_importance_sum[feature] / feature_importance_count[feature] 
                              if feature_importance_count[feature] > 0 else 0 
                              for feature in features}

    # Add new iteration column
    iteration_column = [f'Iteration_{iteration}']
    for row in existing_data[1:]:
        feature = row[0]
        avg_importance = feature_importance_avg.get(feature, 0) if feature in features else ''
        iteration_column.append(str(avg_importance))
    for row in new_data:
        feature = row[0]
        avg_importance = feature_importance_avg.get(feature, 0)
        row.append(str(avg_importance))

    # Update existing data with new column
    existing_data[0].append(f'Iteration_{iteration}')
    for i in range(1, len(existing_data)):
        existing_data[i].append(iteration_column[i])

    # Combine existing and new data
    updated_data = existing_data + new_data

    # Write updated data to CSV
    with open(table_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_data)

    print(f"Feature and Parameter table (with average importance) updated: {table_file}")
    
    # Print out the average feature importance for debugging
    print("\nAverage feature importance summary:")
    for feature, avg_importance in feature_importance_avg.items():
        print(f"{feature}: {avg_importance:.6f}")



def calculate_reflection_angle(row):
    best_period = 0
    best_er = float('-inf')
    for period in range(1, 15):
        p_win = row.get(f'P_Win_{period}d', 0)
        p_return = row.get(f'P_Return_{period}d', 0)
        er = p_win * p_return
        if er > best_er:
            best_er = er
            best_period = period
    
    current_price = row['Close Price']
    ma_14 = row['MA_14']
    end_price_1 = current_price * (1 + best_er)
    slope_1 = (end_price_1 - current_price) / best_period if best_period != 0 else 0
    
    slope_to_ma = row.get('Slope_to_MA', 0)
    reflection_slope = -slope_to_ma
    end_price_2 = current_price + reflection_slope * best_period
    
    end_price_2 = np.clip(end_price_2, min(current_price, ma_14), max(current_price, ma_14))
    slope_2 = (end_price_2 - current_price) / best_period if best_period != 0 else 0
    
    angle = np.arctan((slope_2 - slope_1) / (1 + slope_1 * slope_2)) if (1 + slope_1 * slope_2) != 0 else 0
    direction = 1 if angle >= 0 else -1
    
    return np.degrees(angle), direction, best_period, best_er


def calculate_reflection_features(df):
    # Calculate ER for all periods
    er_columns = [f'P_Win_{period}d' for period in range(1, 15)]
    return_columns = [f'P_Return_{period}d' for period in range(1, 15)]
    
    er_matrix = df[er_columns].values * df[return_columns].values
    
    # Find best period and best ER
    best_er = np.max(er_matrix, axis=1)
    best_period = np.argmax(er_matrix, axis=1) + 1  # +1 because periods start from 1
    
    # Calculate slopes
    current_price = df['Close Price'].values
    ma_14 = df['MA_14'].values
    end_price_1 = current_price * (1 + best_er)
    slope_1 = np.where(best_period != 0, (end_price_1 - current_price) / best_period, 0)
    
    slope_to_ma = df['Slope_to_MA'].values
    reflection_slope = -slope_to_ma
    end_price_2 = current_price + reflection_slope * best_period
    
    end_price_2 = np.clip(end_price_2, np.minimum(current_price, ma_14), np.maximum(current_price, ma_14))
    slope_2 = np.where(best_period != 0, (end_price_2 - current_price) / best_period, 0)
    
    # Calculate angle and direction
    denominator = 1 + slope_1 * slope_2
    angle = np.where(denominator != 0, np.arctan((slope_2 - slope_1) / denominator), 0)
    direction = np.where(angle >= 0, 1, -1)
    
    # Create a DataFrame with the new features
    new_features = pd.DataFrame({
        'Reflection_Angle': np.degrees(angle),
        'Reflection_Direction': direction,
        'Best_Period': best_period,
        'Best_ER': best_er
    }, index=df.index)
    
    return new_features

def calculate_ma_variables(df, ma_period):
    ma_col = f'MA_{ma_period}'
    deviation_col = f'Deviation_from_MA_{ma_period}'
    veered_col = f'Veered_Away_{ma_period}'
    days_since_veer_col = f'Days_Since_Veer_{ma_period}'
    slope_to_ma_col = f'Slope_to_MA_{ma_period}'

    df[ma_col] = df.groupby('Symbol')['Close Price'].transform(lambda x: x.rolling(window=ma_period).mean())
    df[deviation_col] = (df['Close Price'] - df[ma_col]) / df[ma_col]
    df[veered_col] = (df[deviation_col].abs() > 0.05).astype(int)
    df[days_since_veer_col] = df.groupby('Symbol')[veered_col].transform(lambda x: x.cumsum() - x.cumsum().where(x == 1).ffill().fillna(0))
    df[slope_to_ma_col] = df.groupby('Symbol').apply(lambda x: (x['Close Price'].iloc[-1] - x['Close Price'].iloc[-x[days_since_veer_col].iloc[-1]]) / x[days_since_veer_col].iloc[-1] if x[days_since_veer_col].iloc[-1] > 0 else 0).reset_index(level=0, drop=True)

    return df


def select_top_stocks(df, top_fraction=1/3):
    # Get the latest week
    latest_week = df['Week'].max()
    
    # Filter for the latest week
    latest_data = df[df['Week'] == latest_week]
    
    # Calculate the best score for each stock
    best_scores = []
    for _, row in latest_data.iterrows():
        best_score = max(row[f'P_Win_{i}d'] * row[f'P_Return_{i}d'] for i in range(1, 15))
        best_scores.append((row['Symbol'], best_score))
    
    # Sort stocks by their best score
    sorted_stocks = sorted(best_scores, key=lambda x: x[1], reverse=True)
    
    # Select top fraction of stocks
    num_top_stocks = int(len(sorted_stocks) * top_fraction)
    top_stocks = [stock[0] for stock in sorted_stocks[:num_top_stocks]]
    
    return top_stocks

def prepare_additional_features(df):
    new_columns = {}

    # Calculate MA variables for different periods
    ma_periods = [7, 14, 30]
    for period in ma_periods:
        df = calculate_ma_variables(df, period)
    
    # Calculate angle variables
    for period in ma_periods:
        slope_col = f'Slope_to_MA_{period}'
        angle_col = f'Angle_{period}'
        direction_col = f'Direction_{period}'
        if slope_col in df.columns:
            if angle_col not in df.columns:
                new_columns[angle_col] = np.arctan(df[slope_col])
            if direction_col not in df.columns:
                new_columns[direction_col] = np.where(np.isfinite(df[slope_col]), np.sign(df[slope_col]), 0).astype(int)
    
    # Calculate volume spike variables
    if 'volume' in df.columns:
        new_columns['Volume_Change'] = df.groupby('Symbol')['volume'].pct_change()
        new_columns['Positive_Volume_Spike'] = ((new_columns['Volume_Change'] > 2) & (df['Price Change'] > 0)).astype(int)
        new_columns['Negative_Volume_Spike'] = ((new_columns['Volume_Change'] > 2) & (df['Price Change'] < 0)).astype(int)
    
        # Add these columns to df temporarily for the next calculations
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
        new_columns['Days_Since_Positive_Spike'] = df.groupby('Symbol')['Positive_Volume_Spike'].transform(lambda x: x.cumsum() - x.cumsum().where(x == 1).ffill().fillna(0))
        new_columns['Days_Since_Negative_Spike'] = df.groupby('Symbol')['Negative_Volume_Spike'].transform(lambda x: x.cumsum() - x.cumsum().where(x == 1).ffill().fillna(0))
        new_columns['Positive_Spike_Magnitude'] = new_columns['Positive_Volume_Spike'] * df['Price Change']
        new_columns['Negative_Spike_Magnitude'] = new_columns['Negative_Volume_Spike'] * df['Price Change']
    
    # Calculate change in slope variables
    slope_columns = [col for col in df.columns if col.startswith('Slope_') and not col.endswith('_Change')]
    for col in slope_columns:
        new_columns[f'{col}_Change'] = df.groupby('Symbol')[col].diff()
    
    # Add all new columns at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    # Create binary variables
    binary_columns = ['SkewIndex_1d', 'SkewIndex_3d', 'SkewIndex_7d', 'SkewIndex_15d', 'SkewIndex_30d'] + slope_columns
    for col in binary_columns:
        if col in df.columns and df[col].nunique() > 1:
            try:
                df[f'{col}_binary'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
            except ValueError as e:
                print(f"Warning: Could not create binary variable for {col}. Error: {str(e)}")
    
    # Create WOE variables
    for col in binary_columns:
        binary_col = f'{col}_binary'
        if binary_col in df.columns:
            try:
                df[f'{col}_woe'] = calculate_woe(df, binary_col, 'Win_14d')
            except ValueError as e:
                print(f"Warning: Could not create WOE variable for {col}. Error: {str(e)}")
    
    # Create binary and WOE variables for expected_volume_change_{sector}
    if 'sector' in df.columns:
        sectors = df['sector'].unique()
        for sector in sectors:
            col = f'expected_volume_change_{sector}'
            if col in df.columns and df[col].nunique() > 1:
                try:
                    df[f'{col}_binary'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
                    df[f'{col}_woe'] = calculate_woe(df, f'{col}_binary', 'Win_14d')
                except ValueError as e:
                    print(f"Warning: Could not create binary or WOE variables for {col}. Error: {str(e)}")
    
    return df



def generate_smoothed_predictions(train_df, validate_df):
    for period in range(1, 15):
        required_columns = [f'P_Win_{period}d', f'P_Return_{period}d']
        if not all(col in train_df.columns for col in required_columns):
            print(f"Error: Required columns for period {period} not found in train_df")
            continue

        # Apply exponential smoothing to P_Win and P_Return
        win_model = ExponentialSmoothing(train_df[f'P_Win_{period}d'], trend='add', seasonal='add', seasonal_periods=5).fit()
        return_model = ExponentialSmoothing(train_df[f'P_Return_{period}d'], trend='add', seasonal='add', seasonal_periods=5).fit()

        # Generate smoothed predictions for train data (1, 5, and 10 steps ahead)
        smoothed_wins_train_1 = win_model.forecast(1)
        smoothed_returns_train_1 = return_model.forecast(1)
        smoothed_wins_train_5 = win_model.forecast(5)
        smoothed_returns_train_5 = return_model.forecast(5)
        smoothed_wins_train_10 = win_model.forecast(10)
        smoothed_returns_train_10 = return_model.forecast(10)
        
        # Generate smoothed predictions for validate data (1, 5, and 10 steps ahead)
        smoothed_wins_validate_1 = win_model.forecast(1)
        smoothed_returns_validate_1 = return_model.forecast(1)
        smoothed_wins_validate_5 = win_model.forecast(5)
        smoothed_returns_validate_5 = return_model.forecast(5)
        smoothed_wins_validate_10 = win_model.forecast(10)
        smoothed_returns_validate_10 = return_model.forecast(10)
        
       
        # Add smoothed predictions to train_df and validate_df
        train_df.loc[:, f'Smoothed_P_Win_{period}d_1step'] = smoothed_wins_train_1.shift(1)
        train_df.loc[:, f'Smoothed_P_Return_{period}d_1step'] = smoothed_returns_train_1.shift(1)
        train_df.loc[:, f'Smoothed_P_Win_{period}d_5step'] = smoothed_wins_train_5.shift(5)
        train_df.loc[:, f'Smoothed_P_Return_{period}d_5step'] = smoothed_returns_train_5.shift(5)
        train_df.loc[:, f'Smoothed_P_Win_{period}d_10step'] = smoothed_wins_train_10.shift(10)
        train_df.loc[:, f'Smoothed_P_Return_{period}d_10step'] = smoothed_returns_train_10.shift(10)
        
        validate_df.loc[:, f'Smoothed_P_Win_{period}d_1step'] = smoothed_wins_validate_1.shift(1)
        validate_df.loc[:, f'Smoothed_P_Return_{period}d_1step'] = smoothed_returns_validate_1.shift(1)
        validate_df.loc[:, f'Smoothed_P_Win_{period}d_5step'] = smoothed_wins_validate_5.shift(5)
        validate_df.loc[:, f'Smoothed_P_Return_{period}d_5step'] = smoothed_returns_validate_5.shift(5)
        validate_df.loc[:, f'Smoothed_P_Win_{period}d_10step'] = smoothed_wins_validate_10.shift(10)
        validate_df.loc[:, f'Smoothed_P_Return_{period}d_10step'] = smoothed_returns_validate_10.shift(10)
    
    return train_df, validate_df



def normalize_data(df, columns_to_normalize):
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler

def handle_outliers(df, columns_to_check, method='clip'):
    for col in columns_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'clip':
            df[col] = df[col].clip(lower_bound, upper_bound)
        elif method == 'remove':
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

# new for 7.7.24
def handle_outliers(df, columns_to_check, method='IQR', threshold=1.5):
    for col in columns_to_check:
        if method=='IQR':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        else:
        
            raise  ValueError("Method must be either IQR or 'zscore'")
                        
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


# 7.7.24 - ok
def add_missing_columns(train_df, validate_df):
    missing_columns = set(train_df.columns) - set(validate_df.columns)
    
    for col in missing_columns:
        if col.endswith('_binary'):
            # For binary columns, fill with the most common value from train_df
            most_common_value = train_df[col].mode().iloc[0]
            validate_df[col] = most_common_value
        elif col.endswith('_woe'):
            # For WOE columns, fill with the mean value from train_df
            mean_value = train_df[col].mean()
            validate_df[col] = mean_value
        else:
            # For other columns, fill with zeros
            validate_df[col] = 0
    
    return validate_df

def preprocess_data(train_df, validate_df):
    # Columns to normalize and check for outliers
    columns_to_process = [f'P_Win_{period}d' for period in range(1, 15)] + [f'P_Return_{period}d' for period in range(1, 15)]
    
    # Handle infinite values
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    validate_df = validate_df.replace([np.inf, -np.inf], np.nan)
    
    # Handle outliers
    train_df = handle_outliers(train_df, columns_to_process)
    validate_df = handle_outliers(validate_df, columns_to_process)
    
    # Separate numeric and non-numeric columns
    numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = train_df.select_dtypes(exclude=[np.number]).columns
    
    # Fill NaN values with median for numeric columns and mode for non-numeric columns
    train_df[numeric_columns] = train_df[numeric_columns].fillna(train_df[numeric_columns].median())
    train_df[non_numeric_columns] = train_df[non_numeric_columns].fillna(train_df[non_numeric_columns].mode().iloc[0])
    
    validate_df[numeric_columns] = validate_df[numeric_columns].fillna(validate_df[numeric_columns].median())
    validate_df[non_numeric_columns] = validate_df[non_numeric_columns].fillna(validate_df[non_numeric_columns].mode().iloc[0])
    
    # Normalize data
    scaler = MinMaxScaler()
    train_df[columns_to_process] = scaler.fit_transform(train_df[columns_to_process])
    validate_df[columns_to_process] = scaler.transform(validate_df[columns_to_process])
    
    return train_df, validate_df, scaler



def refit_models(train_df, validate_df, models):
    train_df = train_df.copy()
    validate_df = validate_df.copy()

    # Define the columns to check for outliers
    columns_to_check = [col for col in train_df.columns if col.startswith(('SkewIndex_', 'Slope_', 'expected_volume_change_', 'expected_price_change_'))]
    
    # Handle outliers
    train_df = handle_outliers(train_df, columns_to_check, method='clip')
    validate_df = handle_outliers(validate_df, columns_to_check, method='clip')

    updated_models = {}
    
    for period in range(1, 15):
        # Apply exponential smoothing to P_Win and P_Return
        win_model = ExponentialSmoothing(train_df[f'P_Win_{period}d'], trend='add', seasonal='add', seasonal_periods=30).fit()
        return_model = ExponentialSmoothing(train_df[f'P_Return_{period}d'], trend='add', seasonal='add', seasonal_periods=30).fit()
        
        # Generate smoothed predictions for all periods (1-14 days)
        smoothed_wins = win_model.forecast(steps=len(validate_df))
        smoothed_returns = return_model.forecast(steps=len(validate_df))
        
        # Add smoothed predictions to validate_df
        validate_df[f'Smoothed_P_Win_{period}d'] = smoothed_wins
        validate_df[f'Smoothed_P_Return_{period}d'] = smoothed_returns
        
        # Prepare features for Random Forest model
        features = [f'P_Win_{period}d', f'P_Return_{period}d', 
                    f'Smoothed_P_Win_{period}d', f'Smoothed_P_Return_{period}d',
                    'Slope_to_MA_7', 'Slope_to_MA_14', 'Slope_to_MA_30',
                    'Angle_7', 'Angle_14', 'Angle_30',
                    'Direction_7', 'Direction_14', 'Direction_30',
                    'Volume_Change', 'Positive_Volume_Spike', 'Negative_Volume_Spike',
                    'Days_Since_Positive_Spike', 'Days_Since_Negative_Spike',
                    'Positive_Spike_Magnitude', 'Negative_Spike_Magnitude',
                    'Reflection_Angle', 'Reflection_Direction']
        
        features += [col for col in validate_df.columns if col.endswith('_Change')]

        # Add binary and WOE features
        binary_woe_features = [col for col in validate_df.columns if col.endswith('_binary') or col.endswith('_woe')]
        features.extend(binary_woe_features)
       
        # Ensure all features exist in the DataFrame
        existing_features = [f for f in features if f in validate_df.columns]
        
        X = validate_df[existing_features]
        y_win = validate_df[f'Win_{period}d']
        y_return = validate_df[f'Return_{period}d']
        
        rf_win = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        rf_return = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        
        rf_win.fit(X, y_win)
        rf_return.fit(X, y_return)
        
        updated_models[period] = {'win_model': rf_win, 'return_model': rf_return, 'features': existing_features}
    
    return updated_models, train_df, validate_df

def predict_outcomes_refitted(df, updated_models):
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    new_columns = {}
    df = df.reset_index(drop=True)
    for period in range(1, 15):
        if period not in updated_models:
            print(f"Warning: Models for period {period} not found. Skipping predictions for this period.")
            continue
        
        model_info = updated_models[period]
        features = model_info['features']
        
        # Ensure all required features are present in df
        missing_features = set(features) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing features for period {period}: {missing_features}")
            continue
        
        X = df[features].copy()
        
        # Replace inf and -inf with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with the mean of the column
        X = X.fillna(X.mean())
        
        win_key = f'P_Win_{period}d_refitted'
        return_key = f'P_Return_{period}d_refitted'
        
        new_columns[win_key] = model_info['win_model'].predict(X)
        new_columns[return_key] = model_info['return_model'].predict(X)
    
    # Add all new columns at once
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    
    return df




# 7.15 - try to deal with spy data again
# 7.21 - back to dealing with spy again

def calculate_roi_score(historical_data, validation_data, symbol, spy_returns, models, updated_models=None, risk_level='High', min_beta=0.1):
    print(f"Calculating ROI score for {symbol}")
    print(f"spy_returns type in calculate_roi_score: {type(spy_returns)}")
    print(f"spy_returns shape in calculate_roi_score: {spy_returns.shape}")
    print(f"First few values of spy_returns in calculate_roi_score:\n{spy_returns.head()}")
    try:
        print(f"Processing symbol: {symbol}")
        symbol_data = historical_data[historical_data['Symbol'] == symbol]
        validation_symbol_data = validation_data[validation_data['Symbol'] == symbol]
        
        print(f"Symbol data length: {len(symbol_data)}, Validation data length: {len(validation_symbol_data)}")
        
        if len(symbol_data) < 5 or len(validation_symbol_data) < 1:
            print(f"Insufficient data for {symbol}")
            return 0, 0, 0, 0, {}, 0, 0, 0, {}

        # Original calculations
        best_period_original = 0
        best_er_original = float('-inf')
        original_scores = {}
        
        for period in range(1, 15):
            if f'P_Win_{period}d' not in validation_symbol_data.columns or f'P_Return_{period}d' not in validation_symbol_data.columns:
                print(f"Missing P_Win_{period}d or P_Return_{period}d for {symbol}")
                continue
            p_win = validation_symbol_data[f'P_Win_{period}d'].iloc[-1]
            p_return = validation_symbol_data[f'P_Return_{period}d'].iloc[-1]
            er = p_win * p_return
            original_scores[period] = {'p_win': p_win, 'p_return': p_return, 'er': er}
            if er > best_er_original:
                best_er_original = er
                best_period_original = period

        print(f"Best original ER for {symbol}: {best_er_original}         Best Period: {best_period_original}  ")

        # Updated calculations
        best_period_updated = 0
        best_er_updated = float('-inf')
        updated_scores = {}
        
        if updated_models:
            for period in range(1, 15):
                if period not in updated_models:
                    print(f"No updated model for period {period}")
                    continue
                
                features = updated_models[period]['features']
                
                if not all(feature in validation_symbol_data.columns for feature in features):
                    print(f"Missing features for period {period} for {symbol}")
                    continue
                
                X = validation_symbol_data[features].iloc[-1:].copy()
                
                p_win = updated_models[period]['win_model'].predict(X)[0]
                p_return = updated_models[period]['return_model'].predict(X)[0]
                er = p_win * p_return
                updated_scores[period] = {'p_win': p_win, 'p_return': p_return, 'er': er}
                
                if er > best_er_updated:
                    best_er_updated = er
                    best_period_updated = period

            print(f"Best updated ER for {symbol}: {best_er_updated}         Best Period: {best_period_updated}  ")
        else:
            print("No updated models provided.")
            best_er_updated = 0
            score_updated = 0
            alpha_updated = 0

        # Calculate other metrics
        symbol_returns = symbol_data['Close Price'].pct_change().dropna()
        
        # Ensure spy_returns is aligned with symbol_returns
        if isinstance(spy_returns, pd.Series):
            market_returns = spy_returns.reindex(symbol_returns.index)
        else:
            print(f"Warning: spy_returns is not a pandas Series. Type: {type(spy_returns)}")
            market_returns = pd.Series(spy_returns, index=symbol_returns.index)
        
        aligned_returns = pd.concat([symbol_returns, market_returns], axis=1, join='inner')
        aligned_returns.columns = ['symbol_returns', 'market_returns']
        
        if len(aligned_returns) < 5:
            print(f"Insufficient aligned returns for {symbol}")
            return 0, 0, 0, 0, {}, 0, 0, 0, {}
        
        std_dev = aligned_returns['symbol_returns'].std()
        
        # Handle case where all market returns are identical
        if aligned_returns['market_returns'].nunique() == 1:
            print(f"Warning: All market returns are identical for {symbol}. Using default beta.")
            beta = 1.0
        else:
            try:
                slope, _, _, _, _ = stats.linregress(aligned_returns['market_returns'], aligned_returns['symbol_returns'])
                if np.isnan(slope) or np.isinf(slope):
                    print(f"Warning: Invalid slope for {symbol}. Using default beta.")
                    beta = 1.0
                else:
                    beta = max(abs(slope), min_beta) * np.sign(slope)
            except Exception as e:
                print(f"Error calculating beta for {symbol}: {str(e)}. Using default beta.")
                beta = 1.0

        risk_free_rate = 0.03 / 252
        
        if risk_level == 'Low':
            risk_factor = 2
        elif risk_level == 'Medium':
            risk_factor = 1
        else:  # High risk
            risk_factor = 0.5
        
        # Calculate scores for both original and updated
        epsilon = 1e-8  # Small constant to avoid division by zero
        sharpe_ratio_original = (best_er_original - risk_free_rate) / (std_dev * risk_factor + epsilon)
        treynor_ratio_original = (best_er_original - risk_free_rate) / (beta * risk_factor + epsilon)
        score_original = (sharpe_ratio_original + treynor_ratio_original) * (1 + best_er_original)
        alpha_original = best_er_original - (risk_free_rate + beta * (aligned_returns['market_returns'].mean() - risk_free_rate))
        
        if updated_models:
            sharpe_ratio_updated = (best_er_updated - risk_free_rate) / (std_dev * risk_factor + epsilon)
            treynor_ratio_updated = (best_er_updated - risk_free_rate) / (beta * risk_factor + epsilon)
            score_updated = (sharpe_ratio_updated + treynor_ratio_updated) * (1 + best_er_updated)
            alpha_updated = best_er_updated - (risk_free_rate + beta * (aligned_returns['market_returns'].mean() - risk_free_rate))
        else:
            sharpe_ratio_updated = treynor_ratio_updated = score_updated = alpha_updated = 0
        
        print(f"Debug for {symbol}:")
        print(f"best_er_original: {best_er_original}, best_er_updated: {best_er_updated}")
        print(f"std_dev: {std_dev}, beta: {beta}")
        print(f"sharpe_ratio_original: {sharpe_ratio_original}, treynor_ratio_original: {treynor_ratio_original}")
        print(f"sharpe_ratio_updated: {sharpe_ratio_updated}, treynor_ratio_updated: {treynor_ratio_updated}")
        print(f"score_original: {score_original}, score_updated: {score_updated}")
        print(f"alpha_original: {alpha_original}, alpha_updated: {alpha_updated}")
        
        if np.isnan(score_original) or np.isinf(score_original) or np.isnan(score_updated) or np.isinf(score_updated):
            print(f"Invalid score for {symbol}")
            return 0, 0, 0, 0, {}, 0, 0, 0, {}
        
        return score_original, best_er_original, beta, alpha_original, original_scores, score_updated, best_er_updated, alpha_updated, updated_scores
    
    except Exception as e:
        print(f"Error calculating ROI score for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, {}, 0, 0, 0, {}
    



def compare_actual_returns(validation_data, symbol, best_period_original, best_period_updated):
    symbol_data = validation_data[validation_data['Symbol'] == symbol]
    
    actual_return_original = symbol_data[f'Actual_Return_{best_period_original}d'].sum()
    actual_win_original = symbol_data[f'Actual_Win_{best_period_original}d'].sum()
    
    actual_return_updated = symbol_data[f'Actual_Return_{best_period_updated}d'].sum()
    actual_win_updated = symbol_data[f'Actual_Win_{best_period_updated}d'].sum()
    
    return {
        'original': {'return': actual_return_original, 'win': actual_win_original},
        'updated': {'return': actual_return_updated, 'win': actual_win_updated}
    }
    
def generate_comparison_report(validation_data, symbols, original_results, updated_results):
    report = []
    for symbol in symbols:
        symbol_data = validation_data[validation_data['Symbol'] == symbol]
        original = original_results[symbol]
        updated = updated_results[symbol]
        
        actual_results = compare_actual_returns(validation_data, symbol, original['best_period'], updated['best_period'])
        
        report.append({
            'Symbol': symbol,
            'Original_Score': original['score'],
            'Updated_Score': updated['score'],
            'Original_Best_Period': original['best_period'],
            'Updated_Best_Period': updated['best_period'],
            'Original_Expected_Return': original['best_er'],
            'Updated_Expected_Return': updated['best_er'],
            'Original_Actual_Return': actual_results['original']['return'],
            'Updated_Actual_Return': actual_results['updated']['return'],
            'Original_Actual_Win': actual_results['original']['win'],
            'Updated_Actual_Win': actual_results['updated']['win']
        })
    
    report_df = pd.DataFrame(report)
    report_df['Date'] = validation_data['Date'].min()
    return report_df
def plot_comparison(report_df):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(report_df['Original_Score'], report_df['Updated_Score'])
    plt.xlabel('Original Score')
    plt.ylabel('Updated Score')
    plt.title('Original vs Updated Scores')
    plt.plot([0, 1], [0, 1], 'r--')
    
    plt.subplot(2, 2, 2)
    plt.scatter(report_df['Original_Expected_Return'], report_df['Updated_Expected_Return'])
    plt.xlabel('Original Expected Return')
    plt.ylabel('Updated Expected Return')
    plt.title('Original vs Updated Expected Returns')
    plt.plot([0, 1], [0, 1], 'r--')
    
    plt.subplot(2, 2, 3)
    plt.scatter(report_df['Original_Actual_Return'], report_df['Updated_Actual_Return'])
    plt.xlabel('Original Actual Return')
    plt.ylabel('Updated Actual Return')
    plt.title('Original vs Updated Actual Returns')
    plt.plot([0, 1], [0, 1], 'r--')
    
    plt.subplot(2, 2, 4)
    plt.scatter(report_df['Original_Actual_Win'], report_df['Updated_Actual_Win'])
    plt.xlabel('Original Actual Win')
    plt.ylabel('Updated Actual Win')
    plt.title('Original vs Updated Actual Wins')
    plt.plot([0, 1], [0, 1], 'r--')
    
    plt.tight_layout()
    plt.show()



# original version  from before 7.6.24
def estimate_time_until_sale(stock_data):
    best_period = 0
    best_score = float('-inf')
    
    for period in range(1, 15):
        win_col = f'P_Win_{period}d'
        return_col = f'P_Return_{period}d'
        
        if win_col in stock_data and return_col in stock_data:
            score = stock_data[win_col] * stock_data[return_col]
            if score > best_score:
                best_score = score
                best_period = period
    
    return best_period, best_score

def apply_estimate_time_until_sale(row):
    return pd.Series(estimate_time_until_sale(row), index=['Estimated_Hold_Time', 'Best_Score'])



def calculate_avg_actual_win_rate(validation_data, symbol):
    stock_data = validation_data[validation_data['Symbol'] == symbol]
    win_rates = [stock_data[f'Win_{i}d'].mean() for i in range(1, 15)]
    return np.mean(win_rates)

def determine_best_strategy(validation_data, symbol):
    stock_data = validation_data[validation_data['Symbol'] == symbol]
    best_er = 0
    best_period = 0
    for i in range(1, 15):
        er = stock_data[f'P_Win_{i}d'].iloc[-1] * stock_data[f'P_Return_{i}d'].iloc[-1]
        if er > best_er:
            best_er = er
            best_period = i
    return f"{best_period}d"



# 7.5.24 - this version handles both Alpha and Beta versions of the Models
def select_portfolio(stocks, historical_data, spy_return, validation_data, models, updated_models=None, capital=100, min_stocks=10, max_stocks=30):
    scored_stocks_original = []
    scored_stocks_updated = []
    
    total_stocks = len(stocks)
    for i, stock in enumerate(stocks):
        if i % 10 == 0:  # Print progress every 10 stocks
            print(f"Processing stock {i+1}/{total_stocks}")
        
        symbol = stock['Symbol']
        if symbol == 'SPY':
            continue
        
        score_original, er_original, beta, alpha_original, original_scores, score_updated, er_updated, alpha_updated, updated_scores = calculate_roi_score(
            historical_data, validation_data, symbol, spy_return, models, updated_models
        )
        
        if score_original > 0 or score_updated > 0:
            stock_data = validation_data[validation_data['Symbol'] == symbol].iloc[-1]
            estimated_hold_time, best_score = estimate_time_until_sale(stock_data)
            
            stock_info = {
                'symbol': symbol,
                'beta': beta,
                'Estimated_Hold_Time': estimated_hold_time,
                'Best_Score': best_score
            }
            
            if score_original > 0:
                scored_stocks_original.append({
                    **stock_info,
                    'score': score_original,
                    'expected_return': er_original,
                    'alpha': alpha_original
                })
            
            if score_updated > 0:
                scored_stocks_updated.append({
                    **stock_info,
                    'score': score_updated,
                    'expected_return': er_updated,
                    'alpha': alpha_updated
                })
    
    print(f"Number of stocks with valid original scores: {len(scored_stocks_original)}")
    print(f"Number of stocks with valid updated scores: {len(scored_stocks_updated)}")
    
    if not scored_stocks_original and not scored_stocks_updated:
        print("No stocks with valid scores were found. Cannot create portfolios.")
        return None, None
    
    def create_portfolio(scored_stocks):
        if not scored_stocks:
            return None
        
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        selected_stocks = []
        selected_symbols = set()
        sectors_selected = set()
        
        for stock in scored_stocks:
            if len(selected_stocks) >= max_stocks:
                break
            
            if stock['symbol'] in selected_symbols:
                continue  # Skip already selected stocks
            
            stock_data = historical_data[historical_data['Symbol'] == stock['symbol']]
            if stock_data.empty:
                continue
            
            stock_sector = stock_data['sector'].iloc[0]
            
            if stock_sector not in sectors_selected or len(selected_stocks) < min_stocks:
                # Calculate additional metrics
                target_annual_return = (1 + stock['expected_return']) ** (252 / stock['Estimated_Hold_Time']) - 1
                avg_actual_win_rate = calculate_avg_actual_win_rate(validation_data, stock['symbol'])
                best_strategy = determine_best_strategy(validation_data, stock['symbol'])
                
                selected_stocks.append({
                    **stock,
                    'Target_Annual_Return': target_annual_return,
                    'Avg_Actual_Win_Rate': avg_actual_win_rate,
                    'Best_Strategy': best_strategy
                })
                selected_symbols.add(stock['symbol'])
                sectors_selected.add(stock_sector)
        
        if not selected_stocks:
            return None
        
        num_stocks = len(selected_stocks)
        stock_weight = capital / num_stocks
        
        for stock in selected_stocks:
            stock['weight'] = stock_weight / capital
        
        portfolio_er = np.mean([s['expected_return'] for s in selected_stocks])
        portfolio_beta = np.mean([s['beta'] for s in selected_stocks])
        portfolio_alpha = np.mean([s['alpha'] for s in selected_stocks])
        portfolio_estimated_hold_time = np.mean([s['Estimated_Hold_Time'] for s in selected_stocks])
        portfolio_target_annual_return = np.mean([s['Target_Annual_Return'] for s in selected_stocks])
        portfolio_avg_actual_win_rate = np.mean([s['Avg_Actual_Win_Rate'] for s in selected_stocks])
        
        return {
            'selected_stocks': selected_stocks,
            'portfolio_er': portfolio_er,
            'portfolio_beta': portfolio_beta,
            'portfolio_alpha': portfolio_alpha,
            'Estimated_Hold_Time': portfolio_estimated_hold_time,
            'portfolio_target_annual_return': portfolio_target_annual_return,
            'portfolio_avg_actual_win_rate': portfolio_avg_actual_win_rate
        }
    
    portfolio_original = create_portfolio(scored_stocks_original)
    portfolio_updated = create_portfolio(scored_stocks_updated) if updated_models else None
    
    return portfolio_original, portfolio_updated




def calculate_angle(slope1, slope2):
    angle_rad = np.arctan((slope2 - slope1) / (1 + slope1 * slope2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg




def plot_expected_returns_path(portfolio, output_dir, current_time):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    
    plt.figure(figsize=(12, 6))
    
    for stock in portfolio['selected_stocks']:
        symbol = stock['symbol']
        hold_time = stock['Estimated_Hold_Time']
        expected_return = stock['expected_return']
        
        x = [0, hold_time]
        y = [0, expected_return]
        
        plt.plot(x, y, label=f"{symbol} ({hold_time:.0f} days)")
    
    plt.xlabel('Days from Today')
    plt.ylabel('Expected Return')
    plt.title('Expected Returns Path to Exit for Portfolio Stocks')
    plt.legend()
    plt.grid(True)
    
    # Save the plot with timestamp
    filename = f"expected_returns_path_{current_time}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    # plt.close()
    
    return filepath

def plot_all_selected_stocks(portfolio, df, output_dir, current_time, days_of_history=90):
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import timedelta
    import os

    selected_stocks = portfolio['selected_stocks']
    num_stocks = len(selected_stocks)
    
    num_cols = 3
    num_rows = math.ceil(num_stocks / num_cols)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows))
    fig.suptitle("Predicted Paths for Selected Stocks", fontsize=16)
    
    angles = {}  # Dictionary to store angles for each stock
    
    for idx, stock in enumerate(selected_stocks):
        symbol = stock['symbol']
        row = idx // num_cols
        col = idx % num_cols
        
        ax = axs[row, col] if num_rows > 1 else axs[col]
        
        symbol_data = df[df['Symbol'] == symbol].sort_values('Week')
        
        if symbol_data.empty:
            ax.text(0.5, 0.5, f"No data for {symbol}", ha='center', va='center')
            continue
        
        # Convert 'Week' to datetime if it's not already
        symbol_data['Week'] = pd.to_datetime(symbol_data['Week'])
        
        last_row = symbol_data.iloc[-1]
        start_date = last_row['Week'] - timedelta(days=days_of_history)
        historical_data = symbol_data[symbol_data['Week'] > start_date]
        
        best_period = stock['Estimated_Hold_Time']
        current_price = last_row['Close Price']
        expected_return = stock['expected_return']
        ma_14 = last_row['MA_14']
        
        # Calculate additional MAs if not already in the dataframe
        if 'MA_7' not in historical_data.columns:
            historical_data['MA_7'] = historical_data['Close Price'].rolling(window=7).mean()
        if 'MA_30' not in historical_data.columns:
            historical_data['MA_30'] = historical_data['Close Price'].rolling(window=30).mean()
        
        # Path 1: Expected Return
        end_price_1 = current_price * (1 + expected_return)
        slope_1 = (end_price_1 - current_price) / best_period
        
        # Path 2: Symmetrical reflection towards MA_14
        below_ma_data = historical_data[historical_data['Close Price'] < historical_data['MA_14']]
        if not below_ma_data.empty:
            last_below_ma = below_ma_data.iloc[-1]
            days_since_below = (last_row['Week'] - last_below_ma['Week']).days
            if days_since_below > 0:
                slope_to_ma = (current_price - last_below_ma['Close Price']) / days_since_below
                reflection_slope = -slope_to_ma
                end_price_2 = current_price + reflection_slope * best_period
            else:
                end_price_2 = ma_14
        else:
            end_price_2 = ma_14
        
        end_price_2 = np.clip(end_price_2, min(current_price, ma_14), max(current_price, ma_14))
        slope_2 = (end_price_2 - current_price) / best_period
        
        # Calculate angle between the two predicted paths
        angle = calculate_angle(slope_1, slope_2)
        if isinstance(angle, tuple):
            angle = angle[0]  # Take the first value if it's a tuple
        angles[symbol] = -angle  #7.18 pm changed to minus to show up properly (positive to the upside)
        
        prediction_days = range(best_period + 1)
        prediction_dates = [last_row['Week'] + timedelta(days=day) for day in prediction_days]
        
        # Plot historical data and MAs
        ax.plot(historical_data['Week'], historical_data['Close Price'], label='Historical', color='blue')
        ax.plot(historical_data['Week'], historical_data['MA_7'], label='7-day MA', color='green', linestyle='--')
        ax.plot(historical_data['Week'], historical_data['MA_14'], label='14-day MA', color='red', linestyle='--')
        ax.plot(historical_data['Week'], historical_data['MA_30'], label='30-day MA', color='purple', linestyle='--')
        
        # Plot predicted paths
        ax.plot(prediction_dates, np.linspace(current_price, end_price_1, len(prediction_dates)), 
                label='Expected Return', color='orange', linestyle='--', marker='o')
        ax.plot(prediction_dates, np.linspace(current_price, end_price_2, len(prediction_dates)), 
                label='MA Reflection', color='cyan', linestyle=':', marker='s')
        
        ax.set_title(f"{symbol} (Hold: {best_period}d, ER: {expected_return:.2%}, Angle: {angle:.2f}°)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(num_stocks, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        fig.delaxes(axs[row, col] if num_rows > 1 else axs[col])
    
    plt.tight_layout()
    
    # Save the plot with timestamp
    filename = f"selected_stocks_performance_{current_time}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.show()

    # plt.close(fig)  # Close the figure to free up memory
    
    return filepath, angles  # Return both the filepath and the angles dictionary     


def backtest_strategy_1(validate_df, select_portfolio_func, models, initial_capital=20, top_n=20, use_updated_score=False):
    performance = []
    current_capital = initial_capital
    current_date = validate_df['Week'].min()
    end_date = validate_df['Week'].max()
    
    # Get SPY data
    spy_data = validate_df[validate_df['Symbol'] == 'SPY'].copy()
    spy_data['Return'] = spy_data['Close Price'].pct_change()
    spy_data = spy_data.set_index('Week')
    
    while current_date <= end_date:
        # Select top stocks for the current date
        current_data = validate_df[validate_df['Week'] == current_date]
        latest_stocks = current_data.to_dict('records')
        
        # Calculate SPY return
        spy_return = spy_data.loc[current_date:, 'Return'].iloc[0] if current_date in spy_data.index else 0
        
        # Call select_portfolio function
        portfolio, _ = select_portfolio_func(latest_stocks, validate_df, spy_return, current_data, models)
        
        if portfolio is None:
            break
        
        selected_stocks = portfolio['selected_stocks'][:top_n]
        
        for stock in selected_stocks:
            symbol = stock['symbol']
            stock_data = current_data[current_data['Symbol'] == symbol]
            if stock_data.empty:
                continue
            
            buy_price = stock_data['Open Price'].iloc[0]
            shares = 1 / buy_price  # Buy $1 worth of stock
            
            hold_time = int(stock['Estimated_Hold_Time'])
            sell_date = current_date + timedelta(days=hold_time)
            
            if sell_date > end_date:
                sell_date = end_date
            
            sell_data = validate_df[(validate_df['Week'] == sell_date) & (validate_df['Symbol'] == symbol)]
            
            if not sell_data.empty:
                sell_price = sell_data['Close Price'].iloc[0]
                gain_loss = (sell_price - buy_price) * shares
                current_capital += gain_loss
            
            performance.append({
                'Date': current_date,
                'Symbol': symbol,
                'Buy Price': buy_price,
                'Sell Date': sell_date,
                'Sell Price': sell_price if not sell_data.empty else None,
                'Gain/Loss': gain_loss if not sell_data.empty else None,
                'Portfolio Value': current_capital,
                'SPY Return': spy_return
            })
        
        current_date = sell_date
    
    performance_df = pd.DataFrame(performance)
    
    # Calculate performance metrics
    total_return = (current_capital - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (365 / (end_date - validate_df['Week'].min()).days) - 1
    
    # Calculate SPY total return for the same period
    spy_total_return = (spy_data.loc[end_date, 'Close Price'] / spy_data.loc[validate_df['Week'].min(), 'Close Price']) - 1
    spy_annualized_return = (1 + spy_total_return) ** (365 / (end_date - validate_df['Week'].min()).days) - 1
    
    # Create performance chart
    plt.figure(figsize=(12, 6))
    plt.plot(performance_df['Date'], performance_df['Portfolio Value'], label='Strategy 1')
    plt.plot(spy_data.index, spy_data['Close Price'] / spy_data['Close Price'].iloc[0] * initial_capital, label='SPY')
    plt.title('Strategy 1 Performance vs SPY')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Strategy 1 Total Return: {total_return:.2%}")
    print(f"Strategy 1 Annualized Return: {annualized_return:.2%}")
    print(f"SPY Total Return: {spy_total_return:.2%}")
    print(f"SPY Annualized Return: {spy_annualized_return:.2%}")
    
    return performance_df



# 7.21 - new function in place of 7.15 version to create spy returns

def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None,
                                       initial_investment=20000,
                                       strategy_1_annualized_gain=0.7, strategy_1_loss_threshold=-0.07,
                                       strategy_2_gain_threshold=0.025, strategy_2_loss_threshold=-0.07,
                                       strategy_3_gain_threshold=0.04, strategy_3_loss_threshold=-0.07,
                                       skip_top_n=2, depth=20):
    if start_date is None:
        start_date = validate_df['Week'].min()
    if stop_date is None:
        stop_date = validate_df['Week'].max()
    
    start_date = pd.to_datetime(start_date)
    stop_date = pd.to_datetime(stop_date)
    
    # Initialize SPY data
    spy_data = validate_df[validate_df['Symbol'] == 'SPY'].copy()
    
    if spy_data.empty:
        print("Error: No SPY data found in validate_df")
        return None, None, None
    
    print(f"SPY data shape: {spy_data.shape}")
    print(f"SPY data columns: {spy_data.columns}")
    
    spy_data['Return'] = spy_data['Close Price'].pct_change()
    spy_data = spy_data.set_index('Week')
    
    # Create a Series of SPY returns for the entire date range
    date_range = pd.date_range(start=start_date, end=stop_date)
    spy_returns = spy_data['Return'].reindex(date_range).fillna(0)
    
    print(f"spy_returns type: {type(spy_returns)}")
    print(f"spy_returns shape: {spy_returns.shape}")
    print(f"First few values of spy_returns:\n{spy_returns.head()}")
    
    if not isinstance(spy_returns, pd.Series):
        print("Error: spy_returns is not a pandas Series")
        return None, None, None
    
    # Initialize DataFrames to store rankings and daily gains/losses
    rankings_df = pd.DataFrame(columns=['Symbol'])
    
    # Initialize strategy tracking
    strategy_results = {
        'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment}
    }
    
    for strategy in strategy_results:
        if 'Cash' not in strategy_results[strategy]:
            print(f"Warning: 'Cash' key not found in {strategy}. Initializing to {initial_investment}")
            strategy_results[strategy]['Cash'] = initial_investment

    previous_date = None
    
    for current_date in date_range:
        current_data = validate_df[validate_df['Week'] == current_date]
        if current_data.empty:
            print(f"No data available for date: {current_date}")
            continue
        
        print(f"Processing date: {current_date}")
        
        # Calculate rankings for the day
        daily_rankings = []
        for _, stock in current_data.iterrows():
            symbol = stock['Symbol']
            try:
                score_original, _, _, _, _, _, _, _, _ = calculate_roi_score(
                    validate_df, current_data, symbol, spy_returns, models, updated_models
                )
                daily_rankings.append({'Symbol': symbol, 'Score': score_original, 'Close_Price': stock['Close Price']})
            except Exception as e:
                print(f"Error calculating ROI score for {symbol}: {str(e)}")
                continue
        
        daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('Score', ascending=False)
        daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
        daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

        # Implement strategies
        if current_date == start_date:
            print(f"Initializing strategies on start date: {current_date}")
            top_stocks = daily_rankings_df.iloc[skip_top_n:skip_top_n + depth]['Symbol'].tolist()  # Use variable skip and depth
            
            for strategy in strategy_results:
                invest_amount = strategy_results[strategy]['Cash']
                investment_per_stock = invest_amount / len(top_stocks)
                
                for stock in top_stocks:
                    stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
                    shares = investment_per_stock / stock_price
                    strategy_results[strategy]['Book'].append({
                        'Symbol': stock, 
                        'Buy_Date': current_date, 
                        'Buy_Price': stock_price, 
                        'Shares': shares
                    })
                
                strategy_results[strategy]['Cash'] = 0  # All cash is invested
                strategy_results[strategy]['Daily_Value'].append({'Date': current_date, 'Value': invest_amount})
        
        # Update strategies
        for strategy, data in strategy_results.items():
            total_value = data['Cash']
            new_book = []
            for holding in data['Book']:
                symbol = holding['Symbol']
                buy_price = holding['Buy_Price']
                shares = holding['Shares']
                
                current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
                gain_loss = (current_price - buy_price) / buy_price
                
                holding_value = shares * current_price
                total_value += holding_value
                
                # Strategy-specific sell conditions
                if strategy == 'Strategy_1':
                    days_held = (current_date - holding['Buy_Date']).days
                    if days_held > 0:
                        annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
                        if annualized_gain > strategy_1_annualized_gain or gain_loss < strategy_1_loss_threshold:
                            # Sell
                            holding['Sell_Date'] = current_date
                            holding['Sell_Price'] = current_price
                            holding['Gain_Loss'] = gain_loss
                            data['Transactions'].append(holding)
                            data['Cash'] += holding_value
                        else:
                            new_book.append(holding)
                    else:
                        new_book.append(holding)
                elif strategy == 'Strategy_2':
                    if gain_loss > strategy_2_gain_threshold or gain_loss < strategy_2_loss_threshold:
                        # Sell
                        holding['Sell_Date'] = current_date
                        holding['Sell_Price'] = current_price
                        holding['Gain_Loss'] = gain_loss
                        data['Transactions'].append(holding)
                        data['Cash'] += holding_value
                    else:
                        new_book.append(holding)
                elif strategy == 'Strategy_3':
                    if gain_loss > strategy_3_gain_threshold or gain_loss < strategy_3_loss_threshold:
                        # Sell
                        holding['Sell_Date'] = current_date
                        holding['Sell_Price'] = current_price
                        holding['Gain_Loss'] = gain_loss
                        data['Transactions'].append(holding)
                        data['Cash'] += holding_value
                    else:
                        new_book.append(holding)
            
            data['Book'] = new_book
            
            # Reinvestment logic (updated to invest all available cash)
            if data['Cash'] > 0:
                top_stocks = daily_rankings_df.iloc[skip_top_n:skip_top_n + depth]['Symbol'].tolist()  # Use variable skip and depth
                investment_per_stock = data['Cash'] / len(top_stocks)
                
                for stock in top_stocks:
                    stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
                    shares = investment_per_stock / stock_price
                    data['Book'].append({
                        'Symbol': stock, 
                        'Buy_Date': current_date, 
                        'Buy_Price': stock_price,
                        'Shares': shares
                    })
                data['Cash'] = 0  # All cash is reinvested
            
            data['Daily_Value'].append({'Date': current_date, 'Value': total_value})
    
    # Calculate SPY returns for the same period
    spy_initial_value = initial_investment
    spy_values = [spy_initial_value]
    
    for ret in spy_returns:
        spy_values.append(spy_values[-1] * (1 + ret))
    
    spy_values = spy_values[1:]  # Remove the initial value
    
    # Add SPY to strategy results
    strategy_results['SPY'] = {'Daily_Value': [{'Date': date, 'Value': value} for date, value in zip(date_range, spy_values)]}
    
    print("Debug: Strategy results before final report generation:")
    for strategy, data in strategy_results.items():
        print(f"{strategy}: {data.keys()}")
    
    # Generate final report
    strategy_summaries = {}
    for strategy, data in strategy_results.items():
        try:
            if not data['Daily_Value']:
                print(f"Warning: No daily values for {strategy}")
                continue
            
            final_value = data['Daily_Value'][-1]['Value']
            total_return = (final_value - initial_investment) / initial_investment
            
            strategy_summaries[strategy] = {
                'Starting Value': initial_investment,
                'Final Value': final_value,
                'Total Return': total_return,
                'Number of Transactions': len(data.get('Transactions', [])),
                'Current Holdings': len(data.get('Book', [])),
                'Cash Balance': data.get('Cash', 0)  # Use get() with a default value
            }
        except KeyError as e:
            print(f"Error processing {strategy}: Missing key {e}")
        except IndexError:
            print(f"Error processing {strategy}: No daily values")
        except Exception as e:
            print(f"Unexpected error processing {strategy}: {e}")
    
    return strategy_results, rankings_df, strategy_summaries



# 7.15 - this works (required some changes on ROI function to handle spy again though)
def generate_daily_rankings(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None):
    if start_date is None:
        start_date = validate_df['Week'].min()
    if stop_date is None:
        stop_date = validate_df['Week'].max()
    
    start_date = pd.to_datetime(start_date)
    stop_date = pd.to_datetime(stop_date)
    
    # Get SPY data
    spy_data = validate_df[validate_df['Symbol'] == 'SPY'].copy()
    spy_data['Return'] = spy_data['Close Price'].pct_change()
    spy_data = spy_data.set_index('Week')
    
    # Initialize DataFrames to store rankings
    best_er_rankings = pd.DataFrame(columns=['Symbol'])
    score_original_rankings = pd.DataFrame(columns=['Symbol'])
    
    for current_date in validate_df['Week'].sort_values().unique():
        if current_date < start_date or current_date > stop_date:
            continue
        
        print(f"Processing date: {current_date}")
        current_data = validate_df[validate_df['Week'] == current_date]
        latest_stocks = current_data.to_dict('records')
        
        spy_return = spy_data.loc[current_date, 'Return'] if current_date in spy_data.index else 0
        
        daily_best_er = []
        daily_score_original = []
        
        for stock in latest_stocks:
            symbol = stock['Symbol']
            if symbol == 'SPY':
                continue
            
            score_original, best_er, beta, alpha_original, original_scores, _, _, _, _ = calculate_roi_score(
                validate_df, current_data, symbol, spy_return, models, updated_models
            )
            
            daily_best_er.append({'Symbol': symbol, 'best_er': best_er})
            daily_score_original.append({'Symbol': symbol, 'score_original': score_original})
        
        # Sort and rank
        daily_best_er_df = pd.DataFrame(daily_best_er).sort_values('best_er', ascending=False)
        daily_best_er_df['Rank'] = daily_best_er_df['best_er'].rank(method='min', ascending=False)
        
        daily_score_original_df = pd.DataFrame(daily_score_original).sort_values('score_original', ascending=False)
        daily_score_original_df['Rank'] = daily_score_original_df['score_original'].rank(method='min', ascending=False)
        
        # Add to ranking DataFrames
        best_er_rankings = best_er_rankings.merge(daily_best_er_df[['Symbol', 'Rank']], on='Symbol', how='outer', suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}'))
        score_original_rankings = score_original_rankings.merge(daily_score_original_df[['Symbol', 'Rank']], on='Symbol', how='outer', suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}'))
    
    return best_er_rankings, score_original_rankings


def generate_html_report(report):
    html = "<h2>Daily Rankings Report</h2>"
    
    html += "<h3>Best ER Rankings</h3>"
    html += report['best_er_rankings'].to_html(index=False)
    
    html += "<h3>Score Original Rankings</h3>"
    html += report['score_original_rankings'].to_html(index=False)
    
    html += "<h3>Strategy Results</h3>"
    for strategy, data in report['strategy_results'].items():
        html += f"<h4>{strategy}</h4>"
        html += "<h5>Book</h5>"
        book_df = pd.DataFrame(data['Book'])
        html += book_df.to_html(index=False)
        html += "<h5>Transactions</h5>"
        transactions_df = pd.DataFrame(data['Transactions'])
        html += transactions_df.to_html(index=False)
    
    return html









# 7.19.24 10:20pm -let's make it rain

# was the current for a bit

# def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None,
#                                        initial_investment=20000,
#                                        strategy_1_annualized_gain=.7, strategy_1_loss_threshold=-0.07,
#                                        strategy_2_gain_threshold=0.025, strategy_2_loss_threshold=-0.07,
#                                        strategy_3_gain_threshold=0.04, strategy_3_loss_threshold=-0.07):
#     if start_date is None:
#         start_date = validate_df['Week'].min()
#     if stop_date is None:
#         stop_date = validate_df['Week'].max()
    
#     start_date = pd.to_datetime(start_date)
#     stop_date = pd.to_datetime(stop_date)
    
#     # Initialize DataFrames to store rankings and daily gains/losses
#     rankings_df = pd.DataFrame(columns=['Symbol'])
    
#     # Initialize strategy tracking
#     strategy_results = {
#         'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
#         'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
#         'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment}
#     }

#     date_range = pd.date_range(start=start_date, end=stop_date)
#     previous_date = None
    
#     for current_date in date_range:
#         current_data = validate_df[validate_df['Week'] == current_date]
#         if current_data.empty:
#             print(f"No data available for date: {current_date}")
#             continue
        
#         print(f"Processing date: {current_date}")
        
#         # Calculate rankings for the day
#         daily_rankings = []
#         for _, stock in current_data.iterrows():
#             symbol = stock['Symbol']
#             score_original, _, _, _, _, _, _, _, _ = calculate_roi_score(
#                 validate_df, current_data, symbol, 0, models, updated_models
#             )
#             daily_rankings.append({'Symbol': symbol, 'Score': score_original, 'Close_Price': stock['Close Price']})
        
#         daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('Score', ascending=False)
#         daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
#         daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

#         # Implement strategies
#         if current_date == start_date:
#             print(f"Initializing strategies on start date: {current_date}")
#             top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
            
#             for strategy in strategy_results:
#                 invest_amount = strategy_results[strategy]['Cash']
#                 investment_per_stock = invest_amount / len(top_stocks)
                
#                 for stock in top_stocks:
#                     stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
#                     shares = investment_per_stock / stock_price
#                     strategy_results[strategy]['Book'].append({
#                         'Symbol': stock, 
#                         'Buy_Date': current_date, 
#                         'Buy_Price': stock_price, 
#                         'Shares': shares
#                     })
                
#                 strategy_results[strategy]['Cash'] = 0  # All cash is invested
#                 strategy_results[strategy]['Daily_Value'].append({'Date': current_date, 'Value': invest_amount})
                
#                 if strategy == 'Strategy_1':
#                     print(f"Strategy_1 initial buy: {len(strategy_results[strategy]['Book'])} stocks bought")

#         # Update strategies
#         for strategy, data in strategy_results.items():
#             total_value = data['Cash']
#             new_book = []
#             for holding in data['Book']:
#                 symbol = holding['Symbol']
#                 buy_price = holding['Buy_Price']
#                 shares = holding['Shares']
                
#                 current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
#                 gain_loss = (current_price - buy_price) / buy_price
                
#                 holding_value = shares * current_price
#                 total_value += holding_value
                
#                 # Strategy-specific sell conditions
#                 if strategy == 'Strategy_1':
#                     days_held = (current_date - holding['Buy_Date']).days
#                     if days_held > 0:
#                         annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
#                         if annualized_gain > strategy_1_annualized_gain or gain_loss < strategy_1_loss_threshold:
#                             # Sell
#                             holding['Sell_Date'] = current_date
#                             holding['Sell_Price'] = current_price
#                             holding['Gain_Loss'] = gain_loss
#                             data['Transactions'].append(holding)
#                             data['Cash'] += holding_value
#                             print(f"Strategy_1 sold {symbol} for a {gain_loss:.2%} gain/loss")
#                         else:
#                             new_book.append(holding)
#                     else:
#                         new_book.append(holding)
#                 elif strategy == 'Strategy_2':
#                     if gain_loss > strategy_2_gain_threshold or gain_loss < strategy_2_loss_threshold:
#                         # Sell
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Cash'] += holding_value
#                     else:
#                         new_book.append(holding)
#                 elif strategy == 'Strategy_3':
#                     if gain_loss > strategy_3_gain_threshold or gain_loss < strategy_3_loss_threshold:
#                         # Sell
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Cash'] += holding_value
#                     else:
#                         new_book.append(holding)
            
#             data['Book'] = new_book
            
#             # Reinvestment logic (updated to invest all available cash)
#             if data['Cash'] > 0:
#                 top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#                 investment_per_stock = data['Cash'] / len(top_stocks)
                
#                 for stock in top_stocks:
#                     stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
#                     shares = investment_per_stock / stock_price
#                     data['Book'].append({
#                         'Symbol': stock, 
#                         'Buy_Date': current_date, 
#                         'Buy_Price': stock_price,
#                         'Shares': shares
#                     })
#                 if strategy == 'Strategy_1':
#                     print(f"Strategy_1 reinvested in {len(top_stocks)} stocks")
#                 data['Cash'] = 0  # All cash is reinvested
            
#             data['Daily_Value'].append({'Date': current_date, 'Value': total_value})
            
#             if strategy == 'Strategy_1':
#                 print(f"Strategy_1 end of day: {len(data['Book'])} stocks held, Total Value: ${total_value:.2f}")
        
#         # Add rankings to the main DataFrame
#         rankings_df = rankings_df.merge(
#             daily_rankings_df[['Symbol', 'Rank', 'Close_Price']], 
#             on='Symbol', 
#             how='outer', 
#             suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}')
#         )
        
#         # Calculate gain/loss
#         if previous_date:
#             prev_close_col = f'Close_Price_{previous_date.strftime("%Y-%m-%d")}'
#             curr_close_col = f'Close_Price_{current_date.strftime("%Y-%m-%d")}'
#             gain_loss_col = f'Gain_Loss_{current_date.strftime("%Y-%m-%d")}'
            
#             if prev_close_col in rankings_df.columns and curr_close_col in rankings_df.columns:
#                 rankings_df[prev_close_col] = rankings_df[prev_close_col].astype(float)
#                 rankings_df[curr_close_col] = rankings_df[curr_close_col].astype(float)
                
#                 rankings_df[gain_loss_col] = (rankings_df[curr_close_col] - rankings_df[prev_close_col]) / rankings_df[prev_close_col]
#             else:
#                 print(f"Warning: Missing close price data for {previous_date} or {current_date}")
        
#         previous_date = current_date
    
#     # Generate final report
#     strategy_summaries = {}
#     for strategy, data in strategy_results.items():
#         final_value = data['Daily_Value'][-1]['Value']
#         total_return = (final_value - initial_investment) / initial_investment
        
#         strategy_summaries[strategy] = {
#             'Starting Value': initial_investment,
#             'Final Value': final_value,
#             'Total Return': total_return,
#             'Number of Transactions': len(data['Transactions']),
#             'Current Holdings': len(data['Book']),
#             'Cash Balance': data['Cash']
#         }

#     # Create HTML report
#     html_report = "<h2>Strategy Performance Report</h2>"
    
#     # Strategy Summaries
#     html_report += "<h3>Strategy Summaries</h3>"
#     summary_df = pd.DataFrame(strategy_summaries).T
    
#     # Custom formatter function for strategy summaries
#     def strategy_summary_formatter(val, column_name):
#         if column_name == 'Total Return':
#             return f"{val:.2%}"
#         elif isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             return f"${val:.2f}"
#         else:
#             return str(val)
    
#     # Apply the formatter to the summary DataFrame
#     formatted_summary = summary_df.applymap(lambda x: strategy_summary_formatter(x, summary_df.columns[summary_df.eq(x).any()].tolist()[0]))
#     html_report += formatted_summary.to_html()
    
#     # Daily Strategy Values
#     html_report += "<h3>Daily Strategy Values</h3>"
    
#     # Create a DataFrame with all strategy values
#     all_strategy_values = []
#     for strategy, data in strategy_results.items():
#         for daily_value in data['Daily_Value']:
#             all_strategy_values.append({
#                 'Date': daily_value['Date'],
#                 strategy: daily_value['Value']
#             })
    
#     daily_values_df = pd.DataFrame(all_strategy_values)
#     daily_values_df = daily_values_df.groupby('Date').first().reset_index()
    
#     # Format the daily values as currency
#     for strategy in strategy_results.keys():
#         daily_values_df[strategy] = daily_values_df[strategy].apply(lambda x: f"${x:.2f}")
    
#     html_report += daily_values_df.to_html(index=False)
    
#     # Rankings and Daily Gains/Losses
#     html_report += "<h3>Rankings and Daily Gains/Losses</h3>"
    
#     # Custom formatter function for rankings
#     def rankings_formatter(val):
#         if isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             if 'Gain_Loss' in str(val):
#                 return f"{val:.2%}"
#             else:
#                 return f"${val:.2f}"
#         else:
#             return str(val)
    
#     html_report += rankings_df.to_html(index=False, formatters={col: rankings_formatter for col in rankings_df.columns})

#     # Save transaction details to separate files
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     for strategy, data in strategy_results.items():
#         transactions_df = pd.DataFrame(data['Transactions'])
#         if not transactions_df.empty:
#             filename = f"C:\\Users\\apod7\\StockPicker\\strategy\\{strategy}_transactions_{timestamp}.csv"
#             transactions_df.to_csv(filename, index=False)
#             print(f"Transaction details for {strategy} saved to {filename}")

#     return strategy_results, rankings_df, html_report




# 7.19.24 6:37pm- almost there.. works now to get the right numbers in there still need to work on the algorithm execution a bit :)

# def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None,
#                                        initial_investment=20000,
#                                        strategy_1_annualized_gain=0.6, strategy_1_loss_threshold=-0.07,
#                                        strategy_2_gain_threshold=0.035, strategy_2_loss_threshold=-0.07,
#                                        strategy_3_gain_threshold=0.05, strategy_3_loss_threshold=-0.07):
#     if start_date is None:
#         start_date = validate_df['Week'].min()
#     if stop_date is None:
#         stop_date = validate_df['Week'].max()
    
#     start_date = pd.to_datetime(start_date)
#     stop_date = pd.to_datetime(stop_date)
    
#     # Initialize DataFrames to store rankings and daily gains/losses
#     rankings_df = pd.DataFrame(columns=['Symbol'])
    
#     # Initialize strategy tracking
#     strategy_results = {
#         'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': []}
#     }
    
#     date_range = pd.date_range(start=start_date, end=stop_date)
#     previous_date = None
    
#     for current_date in date_range:
#         current_data = validate_df[validate_df['Week'] == current_date]
#         if current_data.empty:
#             print(f"No data available for date: {current_date}")
#             continue
        
#         print(f"Processing date: {current_date}")
        
#         # Calculate rankings for the day
#         daily_rankings = []
#         for _, stock in current_data.iterrows():
#             symbol = stock['Symbol']
#             score_original, _, _, _, _, _, _, _, _ = calculate_roi_score(
#                 validate_df, current_data, symbol, 0, models, updated_models
#             )
#             daily_rankings.append({'Symbol': symbol, 'Score': score_original, 'Close_Price': stock['Close Price']})
        
#         daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('Score', ascending=False)
#         daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
#         daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

#         # Implement strategies
#         if current_date == start_date:
#             print(f"Initializing strategies on start date: {current_date}")
#             top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#             initial_investment_per_stock = initial_investment / len(top_stocks)
            
#             print(f"Top stocks selected: {top_stocks}")
#             print(f"Initial investment per stock: ${initial_investment_per_stock:.2f}")
            
#             for strategy in strategy_results:
#                 strategy_results[strategy]['Book'] = [
#                     {'Symbol': stock, 
#                      'Buy_Date': current_date, 
#                      'Buy_Price': daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0], 
#                      'Shares': initial_investment_per_stock / daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]} 
#                     for stock in top_stocks
#                 ]
#                 strategy_results[strategy]['Daily_Value'].append({'Date': current_date, 'Value': initial_investment})
        
#         # Update strategies
#         for strategy, data in strategy_results.items():
#             total_value = 0
#             new_book = []
#             for holding in data['Book']:
#                 symbol = holding['Symbol']
#                 buy_price = holding['Buy_Price']
#                 shares = holding['Shares']
                
#                 current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
#                 gain_loss = (current_price - buy_price) / buy_price
                
#                 holding_value = shares * current_price
#                 total_value += holding_value
                
#                 # Strategy-specific sell conditions
#                 if strategy == 'Strategy_1':
#                     days_held = (current_date - holding['Buy_Date']).days
#                     if days_held > 0:
#                         annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
#                         if annualized_gain > strategy_1_annualized_gain or gain_loss < strategy_1_loss_threshold:
#                             # Sell
#                             holding['Sell_Date'] = current_date
#                             holding['Sell_Price'] = current_price
#                             holding['Gain_Loss'] = gain_loss
#                             data['Transactions'].append(holding)
#                         else:
#                             new_book.append(holding)
                
#                 elif strategy == 'Strategy_2':
#                     if gain_loss > strategy_2_gain_threshold or gain_loss < strategy_2_loss_threshold:
#                         # Sell
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                     else:
#                         new_book.append(holding)
                
#                 elif strategy == 'Strategy_3':
#                     if gain_loss > strategy_3_gain_threshold or gain_loss < strategy_3_loss_threshold:
#                         # Sell
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                     else:
#                         new_book.append(holding)
            
#             data['Book'] = new_book
            
#             # Reinvestment logic
#             cash_to_invest = total_value - sum(holding['Shares'] * holding['Buy_Price'] for holding in data['Book'])
#             if cash_to_invest > 0:
#                 top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#                 investment_per_stock = cash_to_invest / len(top_stocks)
                
#                 for stock in top_stocks:
#                     stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
#                     shares = investment_per_stock / stock_price
#                     data['Book'].append({
#                         'Symbol': stock, 
#                         'Buy_Date': current_date, 
#                         'Buy_Price': stock_price,
#                         'Shares': shares
#                     })
            
#             data['Daily_Value'].append({'Date': current_date, 'Value': total_value})
        
#         # Add rankings to the main DataFrame
#         rankings_df = rankings_df.merge(
#             daily_rankings_df[['Symbol', 'Rank', 'Close_Price']], 
#             on='Symbol', 
#             how='outer', 
#             suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}')
#         )
        
#         # Calculate gain/loss
#         if previous_date:
#             prev_close_col = f'Close_Price_{previous_date.strftime("%Y-%m-%d")}'
#             curr_close_col = f'Close_Price_{current_date.strftime("%Y-%m-%d")}'
#             gain_loss_col = f'Gain_Loss_{current_date.strftime("%Y-%m-%d")}'
            
#             if prev_close_col in rankings_df.columns and curr_close_col in rankings_df.columns:
#                 rankings_df[prev_close_col] = rankings_df[prev_close_col].astype(float)
#                 rankings_df[curr_close_col] = rankings_df[curr_close_col].astype(float)
                
#                 rankings_df[gain_loss_col] = (rankings_df[curr_close_col] - rankings_df[prev_close_col]) / rankings_df[prev_close_col]
#             else:
#                 print(f"Warning: Missing close price data for {previous_date} or {current_date}")
        
#         previous_date = current_date
    
#     # Generate final report
#     strategy_summaries = {}
#     for strategy, data in strategy_results.items():
#         final_value = data['Daily_Value'][-1]['Value']
#         total_return = (final_value - initial_investment) / initial_investment
        
#         strategy_summaries[strategy] = {
#             'Starting Value': initial_investment,
#             'Final Value': final_value,
#             'Total Return': total_return,
#             'Number of Transactions': len(data['Transactions']),
#             'Current Holdings': len(data['Book'])
#         }

#     # Create HTML report
#     html_report = "<h2>Strategy Performance Report</h2>"
    
#     # Strategy Summaries
#     html_report += "<h3>Strategy Summaries</h3>"
#     summary_df = pd.DataFrame(strategy_summaries).T
    
#     # Custom formatter function for strategy summaries
#     def strategy_summary_formatter(val, column_name):
#         if column_name == 'Total Return':
#             return f"{val:.2%}"
#         elif isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             return f"${val:.2f}"
#         else:
#             return str(val)
    
#     # Apply the formatter to the summary DataFrame
#     formatted_summary = summary_df.applymap(lambda x: strategy_summary_formatter(x, summary_df.columns[summary_df.eq(x).any()].tolist()[0]))
#     html_report += formatted_summary.to_html()
    
#     # Daily Strategy Values
#     html_report += "<h3>Daily Strategy Values</h3>"
    
#     # Create a DataFrame with all strategy values
#     all_strategy_values = []
#     for strategy, data in strategy_results.items():
#         for daily_value in data['Daily_Value']:
#             all_strategy_values.append({
#                 'Date': daily_value['Date'],
#                 strategy: daily_value['Value']
#             })
    
#     daily_values_df = pd.DataFrame(all_strategy_values)
#     daily_values_df = daily_values_df.groupby('Date').first().reset_index()
    
#     # Format the daily values as currency
#     for strategy in strategy_results.keys():
#         daily_values_df[strategy] = daily_values_df[strategy].apply(lambda x: f"${x:.2f}")
    
#     html_report += daily_values_df.to_html(index=False)
    
#     # Rankings and Daily Gains/Losses
#     html_report += "<h3>Rankings and Daily Gains/Losses</h3>"
    
#     # Custom formatter function for rankings
#     def rankings_formatter(val):
#         if isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             if 'Gain_Loss' in str(val):
#                 return f"{val:.2%}"
#             else:
#                 return f"${val:.2f}"
#         else:
#             return str(val)
    
#     html_report += rankings_df.to_html(index=False, formatters={col: rankings_formatter for col in rankings_df.columns})

#     # Save transaction details to separate files
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     for strategy, data in strategy_results.items():
#         transactions_df = pd.DataFrame(data['Transactions'])
#         if not transactions_df.empty:
#             filename = f"C:\\Users\\apod7\\StockPicker\\strategy\\{strategy}_transactions_{timestamp}.csv"
#             transactions_df.to_csv(filename, index=False)
#             print(f"Transaction details for {strategy} saved to {filename}")

#     return strategy_results, rankings_df, html_report


# 7.19.24 pm - getting there...

# def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None,
#                                        initial_investment=20000,
#                                        strategy_1_annualized_gain=0.9, strategy_1_loss_threshold=-0.07,
#                                        strategy_2_gain_threshold=0.035, strategy_2_loss_threshold=-0.07,
#                                        strategy_3_gain_threshold=0.05, strategy_3_loss_threshold=-0.07):
#     if start_date is None:
#         start_date = validate_df['Week'].min()
#     if stop_date is None:
#         stop_date = validate_df['Week'].max()
    
#     start_date = pd.to_datetime(start_date)
#     stop_date = pd.to_datetime(stop_date)
    
#     # Initialize DataFrames to store rankings and daily gains/losses
#     rankings_df = pd.DataFrame(columns=['Symbol'])
    
#     # Initialize strategy tracking
#     strategy_results = {
#         'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': []}
#     }
    
#     date_range = pd.date_range(start=start_date, end=stop_date)
#     previous_date = None
    
#     for current_date in date_range:
#         current_data = validate_df[validate_df['Week'] == current_date]
#         if current_data.empty:
#             print(f"No data available for date: {current_date}")
#             continue
        
#         print(f"Processing date: {current_date}")
        
#         # Calculate rankings for the day
#         daily_rankings = []
#         for _, stock in current_data.iterrows():
#             symbol = stock['Symbol']
#             score_original, _, _, _, _, _, _, _, _ = calculate_roi_score(
#                 validate_df, current_data, symbol, 0, models, updated_models
#             )
#             daily_rankings.append({'Symbol': symbol, 'Score': score_original, 'Close_Price': stock['Close Price']})
        
#         daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('Score', ascending=False)
#         daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
#         daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

#         # Implement strategies
#         if current_date == start_date:
#             print(f"Initializing strategies on start date: {current_date}")
#             top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#             initial_investment_per_stock = initial_investment / len(top_stocks)
            
#             for strategy in strategy_results:
#                 strategy_results[strategy]['Book'] = [
#                     {'Symbol': stock, 
#                      'Buy_Date': current_date, 
#                      'Buy_Price': daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0], 
#                      'Shares': initial_investment_per_stock / daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]} 
#                     for stock in top_stocks
#                 ]
#                 strategy_results[strategy]['Daily_Value'].append({'Date': current_date, 'Value': initial_investment})
        
#         # Update strategies
#         for strategy, data in strategy_results.items():
#             total_value = 0
#             for holding in data['Book']:
#                 symbol = holding['Symbol']
#                 buy_price = holding['Buy_Price']
#                 shares = holding['Shares']
                
#                 current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
#                 gain_loss = (current_price - buy_price) / buy_price
                
#                 holding_value = shares * current_price
#                 total_value += holding_value
                
#                 # Strategy-specific sell conditions
#                 if strategy == 'Strategy_1':
#                     days_held = (current_date - holding['Buy_Date']).days
#                     if days_held > 0:
#                         annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
#                         if annualized_gain > strategy_1_annualized_gain or gain_loss < strategy_1_loss_threshold:
#                             # Sell
#                             holding['Sell_Date'] = current_date
#                             holding['Sell_Price'] = current_price
#                             holding['Gain_Loss'] = gain_loss
#                             data['Transactions'].append(holding)
#                             data['Book'].remove(holding)
                
#                 elif strategy == 'Strategy_2':
#                     if gain_loss > strategy_2_gain_threshold or gain_loss < strategy_2_loss_threshold:
#                         # Sell
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Book'].remove(holding)
                
#                 elif strategy == 'Strategy_3':
#                     if gain_loss > strategy_3_gain_threshold or gain_loss < strategy_3_loss_threshold:
#                         # Sell and reinvest
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Book'].remove(holding)
                        
#                         # Reinvest
#                         reinvestment_amount = shares * current_price
#                         top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#                         reinvestment_per_stock = reinvestment_amount / len(top_stocks)
                        
#                         for stock in top_stocks:
#                             stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
#                             data['Book'].append({
#                                 'Symbol': stock, 
#                                 'Buy_Date': current_date, 
#                                 'Buy_Price': stock_price,
#                                 'Shares': reinvestment_per_stock / stock_price
#                             })
            
#             data['Daily_Value'].append({'Date': current_date, 'Value': total_value})
        
#         # Add rankings to the main DataFrame
#         rankings_df = rankings_df.merge(
#             daily_rankings_df[['Symbol', 'Rank', 'Close_Price']], 
#             on='Symbol', 
#             how='outer', 
#             suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}')
#         )
        
#         # Calculate gain/loss
#         if previous_date:
#             prev_close_col = f'Close_Price_{previous_date.strftime("%Y-%m-%d")}'
#             curr_close_col = f'Close_Price_{current_date.strftime("%Y-%m-%d")}'
#             gain_loss_col = f'Gain_Loss_{current_date.strftime("%Y-%m-%d")}'
            
#             if prev_close_col in rankings_df.columns and curr_close_col in rankings_df.columns:
#                 rankings_df[prev_close_col] = rankings_df[prev_close_col].astype(float)
#                 rankings_df[curr_close_col] = rankings_df[curr_close_col].astype(float)
                
#                 rankings_df[gain_loss_col] = (rankings_df[curr_close_col] - rankings_df[prev_close_col]) / rankings_df[prev_close_col]
#             else:
#                 print(f"Warning: Missing close price data for {previous_date} or {current_date}")
        
#         previous_date = current_date
    
#     # Generate final report
#     strategy_summaries = {}
#     for strategy, data in strategy_results.items():
#         final_value = data['Daily_Value'][-1]['Value']
#         total_return = (final_value - initial_investment) / initial_investment
        
#         strategy_summaries[strategy] = {
#             'Starting Value': initial_investment,
#             'Final Value': final_value,
#             'Total Return': total_return,
#             'Number of Transactions': len(data['Transactions']),
#             'Current Holdings': len(data['Book'])
#         }

#     # Create HTML report
#     html_report = "<h2>Strategy Performance Report</h2>"
    
#     # Strategy Summaries
#     html_report += "<h3>Strategy Summaries</h3>"
#     summary_df = pd.DataFrame(strategy_summaries).T
    
#     # Custom formatter function for strategy summaries
#     def strategy_summary_formatter(val, column_name):
#         if column_name == 'Total Return':
#             return f"{val:.2%}"
#         elif isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             return f"${val:.2f}"
#         else:
#             return str(val)
    
#     # Apply the formatter to the summary DataFrame
#     formatted_summary = summary_df.applymap(lambda x: strategy_summary_formatter(x, summary_df.columns[summary_df.eq(x).any()].tolist()[0]))
#     html_report += formatted_summary.to_html()
    
#     # Daily Strategy Values
#     html_report += "<h3>Daily Strategy Values</h3>"
    
#     # Create a DataFrame with all strategy values
#     all_strategy_values = []
#     for strategy, data in strategy_results.items():
#         for daily_value in data['Daily_Value']:
#             all_strategy_values.append({
#                 'Date': daily_value['Date'],
#                 strategy: daily_value['Value']
#             })
    
#     daily_values_df = pd.DataFrame(all_strategy_values)
#     daily_values_df = daily_values_df.groupby('Date').first().reset_index()
    
#     # Format the daily values as currency
#     for strategy in strategy_results.keys():
#         daily_values_df[strategy] = daily_values_df[strategy].apply(lambda x: f"${x:.2f}")
    
#     html_report += daily_values_df.to_html(index=False)
    
#     # Rankings and Daily Gains/Losses
#     html_report += "<h3>Rankings and Daily Gains/Losses</h3>"
    
#     # Custom formatter function for rankings
#     def rankings_formatter(val):
#         if isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             if 'Gain_Loss' in str(val):
#                 return f"{val:.2%}"
#             else:
#                 return f"${val:.2f}"
#         else:
#             return str(val)
    
#     html_report += rankings_df.to_html(index=False, formatters={col: rankings_formatter for col in rankings_df.columns})

#     # Save transaction details to separate files
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     for strategy, data in strategy_results.items():
#         transactions_df = pd.DataFrame(data['Transactions'])
#         if not transactions_df.empty:
#             filename = f"C:\\Users\\apod7\\StockPicker\\strategy\\{strategy}_transactions_{timestamp}.csv"
#             transactions_df.to_csv(filename, index=False)
#             print(f"Transaction details for {strategy} saved to {filename}")

#     return strategy_results, rankings_df, html_report












# # 7.19 - just fixing formatting at this point - looks useful, but going to make it better above ;)

# def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None):
#     if start_date is None:
#         start_date = validate_df['Week'].min()
#     if stop_date is None:
#         stop_date = validate_df['Week'].max()
    
#     start_date = pd.to_datetime(start_date)
#     stop_date = pd.to_datetime(stop_date)
    
#     # Initialize DataFrames to store rankings and daily gains/losses
#     rankings_df = pd.DataFrame(columns=['Symbol'])
    
#     # Initialize strategy tracking
#     strategy_results = {
#         'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': []}
#     }
    
#     initial_investment = 20000  # $20,000 initial investment per strategy
    
#     date_range = pd.date_range(start=start_date, end=stop_date)
#     previous_date = None
    
#     for current_date in date_range:
#         current_data = validate_df[validate_df['Week'] == current_date]
#         if current_data.empty:
#             print(f"No data available for date: {current_date}")
#             continue
        
#         print(f"Processing date: {current_date}")
        
#         # Calculate rankings for the day
#         daily_rankings = []
#         for _, stock in current_data.iterrows():
#             symbol = stock['Symbol']
#             score_original, _, _, _, _, _, _, _, _ = calculate_roi_score(
#                 validate_df, current_data, symbol, 0, models, updated_models
#             )
#             daily_rankings.append({'Symbol': symbol, 'Score': score_original, 'Close_Price': stock['Close Price']})
        
#         daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('Score', ascending=False)
#         daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
#         daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

#         # Implement strategies
#         if current_date == start_date:
#             print(f"Initializing strategies on start date: {current_date}")
#             # Initial buy for all strategies
#             top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#             initial_investment_per_stock = initial_investment / len(top_stocks)
            
#             print(f"Top stocks selected: {top_stocks}")
#             print(f"Initial investment per stock: ${initial_investment_per_stock:.2f}")
            
#             for strategy in strategy_results:
#                 strategy_results[strategy]['Book'] = [
#                     {'Symbol': stock, 
#                      'Buy_Date': current_date, 
#                      'Buy_Price': daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0], 
#                      'Shares': initial_investment_per_stock / daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]} 
#                     for stock in top_stocks
#                 ]
#                 strategy_results[strategy]['Daily_Value'].append({'Date': current_date, 'Value': initial_investment})
            
#             print(f"Initial buys for strategies: {strategy_results}")
        
#         # Update strategies
#         for strategy, data in strategy_results.items():
#             total_value = 0
#             for holding in data['Book']:
#                 symbol = holding['Symbol']
#                 buy_price = holding['Buy_Price']
#                 shares = holding['Shares']
                
#                 current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
#                 gain_loss = (current_price - buy_price) / buy_price
                
#                 holding_value = shares * current_price
#                 total_value += holding_value
                
#                 # Strategy-specific sell conditions
#                 if strategy == 'Strategy_1':
#                     days_held = (current_date - holding['Buy_Date']).days
#                     if days_held > 0:
#                         annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
#                         if annualized_gain > 0.6 or gain_loss < -0.07:
#                             # Sell
#                             holding['Sell_Date'] = current_date
#                             holding['Sell_Price'] = current_price
#                             holding['Gain_Loss'] = gain_loss
#                             data['Transactions'].append(holding)
#                             data['Book'].remove(holding)
                
#                 elif strategy == 'Strategy_2':
#                     if gain_loss > 0.035 or gain_loss < -0.07:
#                         # Sell
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Book'].remove(holding)
                
#                 elif strategy == 'Strategy_3':
#                     if gain_loss > 0.035 or gain_loss < -0.07:
#                         # Sell and reinvest
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Book'].remove(holding)
                        
#                         # Reinvest
#                         reinvestment_amount = shares * current_price
#                         top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#                         reinvestment_per_stock = reinvestment_amount / len(top_stocks)
                        
#                         for stock in top_stocks:
#                             stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
#                             data['Book'].append({
#                                 'Symbol': stock, 
#                                 'Buy_Date': current_date, 
#                                 'Buy_Price': stock_price,
#                                 'Shares': reinvestment_per_stock / stock_price
#                             })
            
#             data['Daily_Value'].append({'Date': current_date, 'Value': total_value})
        
#         # Add rankings to the main DataFrame
#         rankings_df = rankings_df.merge(
#             daily_rankings_df[['Symbol', 'Rank', 'Close_Price']], 
#             on='Symbol', 
#             how='outer', 
#             suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}')
#         )
        
#         # Calculate gain/loss
#         if previous_date:
#             prev_close_col = f'Close_Price_{previous_date.strftime("%Y-%m-%d")}'
#             curr_close_col = f'Close_Price_{current_date.strftime("%Y-%m-%d")}'
#             gain_loss_col = f'Gain_Loss_{current_date.strftime("%Y-%m-%d")}'
            
#             print(f"Columns in rankings_df: {rankings_df.columns}")
#             print(f"Checking for columns: {prev_close_col}, {curr_close_col}")
            
#             if prev_close_col in rankings_df.columns and curr_close_col in rankings_df.columns:
#                 # Ensure both columns are float type
#                 rankings_df[prev_close_col] = rankings_df[prev_close_col].astype(float)
#                 rankings_df[curr_close_col] = rankings_df[curr_close_col].astype(float)
                
#                 # Calculate gain/loss
#                 rankings_df[gain_loss_col] = (rankings_df[curr_close_col] - rankings_df[prev_close_col]) / rankings_df[prev_close_col]
#             else:
#                 print(f"Warning: Missing close price data for {previous_date} or {current_date}")
        
#         previous_date = current_date
    
#     # Generate final report
#     strategy_summaries = {}
#     for strategy, data in strategy_results.items():
#         final_value = data['Daily_Value'][-1]['Value']
#         total_return = (final_value - initial_investment) / initial_investment
        
#         strategy_summaries[strategy] = {
#             'Starting Value': initial_investment,
#             'Final Value': final_value,
#             'Total Return': total_return,
#             'Number of Transactions': len(data['Transactions']),
#             'Current Holdings': len(data['Book'])
#         }

#     # Create HTML report
#     html_report = "<h2>Strategy Performance Report</h2>"
    
#     # Strategy Summaries
#     html_report += "<h3>Strategy Summaries</h3>"
#     summary_df = pd.DataFrame(strategy_summaries).T
    
#     # Custom formatter function for strategy summaries
#     def strategy_summary_formatter(val, column_name):
#         if column_name == 'Total Return':
#             return f"{val:.2%}"
#         elif isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             return f"${val:.2f}"
#         else:
#             return str(val)
    
#     # Apply the formatter to the summary DataFrame
#     formatted_summary = summary_df.applymap(lambda x: strategy_summary_formatter(x, summary_df.columns[summary_df.eq(x).any()].tolist()[0]))
#     html_report += formatted_summary.to_html()
    
#     # Daily Strategy Values
#     html_report += "<h3>Daily Strategy Values</h3>"
    
#     # Create a DataFrame with all strategy values
#     all_strategy_values = []
#     for strategy, data in strategy_results.items():
#         for daily_value in data['Daily_Value']:
#             all_strategy_values.append({
#                 'Date': daily_value['Date'],
#                 strategy: daily_value['Value']
#             })
    
#     daily_values_df = pd.DataFrame(all_strategy_values)
#     daily_values_df = daily_values_df.groupby('Date').first().reset_index()
    
#     # Format the daily values as currency
#     for strategy in strategy_results.keys():
#         daily_values_df[strategy] = daily_values_df[strategy].apply(lambda x: f"${x:.2f}")
    
#     html_report += daily_values_df.to_html(index=False)
    
#     # Rankings and Daily Gains/Losses
#     html_report += "<h3>Rankings and Daily Gains/Losses</h3>"
    
#     # Custom formatter function for rankings
#     def rankings_formatter(val):
#         if isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             if 'Gain_Loss' in str(val):
#                 return f"{val:.2%}"
#             else:
#                 return f"${val:.2f}"
#         else:
#             return str(val)
    
#     html_report += rankings_df.to_html(index=False, formatters={col: rankings_formatter for col in rankings_df.columns})

#     return strategy_results, rankings_df, html_report




# # 7.19.24 - strategy simulation ;)
#     # first working version !!!!
# def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None):
#     if start_date is None:
#         start_date = validate_df['Week'].min()
#     if stop_date is None:
#         stop_date = validate_df['Week'].max()
    
#     start_date = pd.to_datetime(start_date)
#     stop_date = pd.to_datetime(stop_date)
    
#     # Initialize DataFrames to store rankings and daily gains/losses
#     rankings_df = pd.DataFrame(columns=['Symbol'])
    
#     # Initialize strategy tracking
#     strategy_results = {
#         'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': []},
#         'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': []}
#     }
    
#     initial_investment = 20000  # $20,000 initial investment per strategy
    
#     date_range = pd.date_range(start=start_date, end=stop_date)
#     previous_date = None
    
#     for current_date in date_range:
#         current_data = validate_df[validate_df['Week'] == current_date]
#         if current_data.empty:
#             print(f"No data available for date: {current_date}")
#             continue
        
#         print(f"Processing date: {current_date}")
        
#         # Calculate rankings for the day
#         daily_rankings = []
#         for _, stock in current_data.iterrows():
#             symbol = stock['Symbol']
#             score_original, _, _, _, _, _, _, _, _ = calculate_roi_score(
#                 validate_df, current_data, symbol, 0, models, updated_models
#             )
#             daily_rankings.append({'Symbol': symbol, 'Score': score_original, 'Close_Price': stock['Close Price']})
        
#         daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('Score', ascending=False)
#         daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
#         daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

#         # Implement strategies
#         if current_date == start_date:
#             print(f"Initializing strategies on start date: {current_date}")
#             # Initial buy for all strategies
#             top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#             initial_investment_per_stock = initial_investment / len(top_stocks)
            
#             print(f"Top stocks selected: {top_stocks}")
#             print(f"Initial investment per stock: ${initial_investment_per_stock:.2f}")
            
#             for strategy in strategy_results:
#                 strategy_results[strategy]['Book'] = [
#                     {'Symbol': stock, 
#                      'Buy_Date': current_date, 
#                      'Buy_Price': daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0], 
#                      'Shares': initial_investment_per_stock / daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]} 
#                     for stock in top_stocks
#                 ]
#                 strategy_results[strategy]['Daily_Value'].append({'Date': current_date, 'Value': initial_investment})
            
#             print(f"Initial buys for strategies: {strategy_results}")
        
#         # Update strategies
#         for strategy, data in strategy_results.items():
#             total_value = 0
#             for holding in data['Book']:
#                 symbol = holding['Symbol']
#                 buy_price = holding['Buy_Price']
#                 shares = holding['Shares']
                
#                 current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
#                 gain_loss = (current_price - buy_price) / buy_price
                
#                 holding_value = shares * current_price
#                 total_value += holding_value
                
#                 # Strategy-specific sell conditions
#                 if strategy == 'Strategy_1':
#                     days_held = (current_date - holding['Buy_Date']).days
#                     if days_held > 0:
#                         annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
#                         if annualized_gain > 1.6 or gain_loss < -0.07:
#                             # Sell
#                             holding['Sell_Date'] = current_date
#                             holding['Sell_Price'] = current_price
#                             holding['Gain_Loss'] = gain_loss
#                             data['Transactions'].append(holding)
#                             data['Book'].remove(holding)
                
#                 elif strategy == 'Strategy_2':
#                     if gain_loss > 0.035 or gain_loss < -0.07:
#                         # Sell
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Book'].remove(holding)
                
#                 elif strategy == 'Strategy_3':
#                     if gain_loss > 0.035 or gain_loss < -0.07:
#                         # Sell and reinvest
#                         holding['Sell_Date'] = current_date
#                         holding['Sell_Price'] = current_price
#                         holding['Gain_Loss'] = gain_loss
#                         data['Transactions'].append(holding)
#                         data['Book'].remove(holding)
                        
#                         # Reinvest
#                         reinvestment_amount = shares * current_price
#                         top_stocks = daily_rankings_df.iloc[2:22]['Symbol'].tolist()  # Skip top 2, select next 20
#                         reinvestment_per_stock = reinvestment_amount / len(top_stocks)
                        
#                         for stock in top_stocks:
#                             stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
#                             data['Book'].append({
#                                 'Symbol': stock, 
#                                 'Buy_Date': current_date, 
#                                 'Buy_Price': stock_price,
#                                 'Shares': reinvestment_per_stock / stock_price
#                             })
            
#             data['Daily_Value'].append({'Date': current_date, 'Value': total_value})
        
#         # Add rankings to the main DataFrame
#         rankings_df = rankings_df.merge(
#             daily_rankings_df[['Symbol', 'Rank', 'Close_Price']], 
#             on='Symbol', 
#             how='outer', 
#             suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}')
#         )
        
#         # Calculate gain/loss
#         if previous_date:
#             prev_close_col = f'Close_Price_{previous_date.strftime("%Y-%m-%d")}'
#             curr_close_col = f'Close_Price_{current_date.strftime("%Y-%m-%d")}'
#             gain_loss_col = f'Gain_Loss_{current_date.strftime("%Y-%m-%d")}'
            
#             print(f"Columns in rankings_df: {rankings_df.columns}")
#             print(f"Checking for columns: {prev_close_col}, {curr_close_col}")
            
#             if prev_close_col in rankings_df.columns and curr_close_col in rankings_df.columns:
#                 # Ensure both columns are float type
#                 rankings_df[prev_close_col] = rankings_df[prev_close_col].astype(float)
#                 rankings_df[curr_close_col] = rankings_df[curr_close_col].astype(float)
                
#                 # Calculate gain/loss
#                 rankings_df[gain_loss_col] = (rankings_df[curr_close_col] - rankings_df[prev_close_col]) / rankings_df[prev_close_col]
#             else:
#                 print(f"Warning: Missing close price data for {previous_date} or {current_date}")
        
#         previous_date = current_date
    
#     # Generate final report
#     strategy_summaries = {}
#     for strategy, data in strategy_results.items():
#         final_value = data['Daily_Value'][-1]['Value']
#         total_return = (final_value - initial_investment) / initial_investment
        
#         strategy_summaries[strategy] = {
#             'Starting Value': initial_investment,
#             'Final Value': final_value,
#             'Total Return': total_return,
#             'Number of Transactions': len(data['Transactions']),
#             'Current Holdings': len(data['Book'])
#         }

#     # Create HTML report
#     html_report = "<h2>Strategy Performance Report</h2>"
    
#     # Strategy Summaries
#     html_report += "<h3>Strategy Summaries</h3>"
#     summary_df = pd.DataFrame(strategy_summaries).T
#     html_report += summary_df.to_html(float_format=lambda x: f'${x:.2f}')
    
#     # Daily Strategy Values
#     html_report += "<h3>Daily Strategy Values</h3>"
#     for strategy, data in strategy_results.items():
#         df = pd.DataFrame(data['Daily_Value'])
#         html_report += f"<h4>{strategy}</h4>"
#         html_report += df.to_html(index=False, float_format=lambda x: f'${x:.2f}')
    
#     # Rankings and Daily Gains/Losses
#     html_report += "<h3>Rankings and Daily Gains/Losses</h3>"
    
#     # Custom formatter function
#     def custom_formatter(val):
#         if isinstance(val, (int, np.integer)):
#             return f"{val:d}"
#         elif isinstance(val, (float, np.float64)):
#             if 'Gain_Loss' in str(val):
#                 return f"{val:.2%}"
#             else:
#                 return f"${val:.2f}"
#         else:
#             return str(val)
    
#     html_report += rankings_df.to_html(index=False, formatters={col: custom_formatter for col in rankings_df.columns})

#     return strategy_results, rankings_df, html_report




#7.15 -  Now create some plots 
def create_ranking_report(best_er_rankings, score_original_rankings):
    # Create output directory if it doesn't exist
    output_dir = r'C:\Users\apod7\StockPicker\daily_rank'
    os.makedirs(output_dir, exist_ok=True)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dates = [col.split('_')[-1] for col in best_er_rankings.columns if col.startswith('Rank_')]
    
    # Convert ranks to integers
    for date in dates:
        best_er_rankings[f'Rank_{date}'] = best_er_rankings[f'Rank_{date}'].astype(int)
        score_original_rankings[f'Rank_{date}'] = score_original_rankings[f'Rank_{date}'].astype(int)
    
    # Calculate rank changes
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        best_er_rankings[f'Rank_change_{curr_date}'] = best_er_rankings[f'Rank_{prev_date}'] - best_er_rankings[f'Rank_{curr_date}']
        score_original_rankings[f'Rank_change_{curr_date}'] = score_original_rankings[f'Rank_{prev_date}'] - score_original_rankings[f'Rank_{curr_date}']
    
    # Sort the DataFrames by the most recent date's rank
    most_recent_date = dates[-1]
    best_er_rankings = best_er_rankings.sort_values(by=f'Rank_{most_recent_date}')
    score_original_rankings = score_original_rankings.sort_values(by=f'Rank_{most_recent_date}')
    
    # Create separate DataFrames for ranks and rank changes
    er_rank_df = best_er_rankings[['Symbol'] + [col for col in best_er_rankings.columns if col.startswith('Rank_') and not col.startswith('Rank_change_')]]
    er_rank_change_df = best_er_rankings[['Symbol'] + [col for col in best_er_rankings.columns if col.startswith('Rank_change_')]]
    score_rank_df = score_original_rankings[['Symbol'] + [col for col in score_original_rankings.columns if col.startswith('Rank_') and not col.startswith('Rank_change_')]]
    score_rank_change_df = score_original_rankings[['Symbol'] + [col for col in score_original_rankings.columns if col.startswith('Rank_change_')]]
    
    # Plot top 20 stocks based on the most recent score rank
    top_20_symbols = score_original_rankings.nsmallest(20, f'Rank_{most_recent_date}')['Symbol'].tolist()
    
    # Create a function to plot and add statistics table
    def plot_with_stats(rankings, rank_changes, title, filename):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20), gridspec_kw={'height_ratios': [3, 1]})
        for symbol in top_20_symbols:
            ax1.plot(dates, [min(rankings.loc[rankings['Symbol'] == symbol, f'Rank_{date}'].values[0], 30) for date in dates], label=symbol)
        ax1.set_title(title)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Rank')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(30, 1)  # Set y-axis limits from 30 to 1
        ax1.set_yticks(range(1, 31, 5))  # Set y-axis ticks from 1 to 30 with step 5
        
        # Create statistics table
        stats_data = []
        for symbol in top_20_symbols:
            stats_data.append([
                symbol,
                rankings.loc[rankings['Symbol'] == symbol, f'Rank_{most_recent_date}'].values[0],
                rank_changes.loc[rank_changes['Symbol'] == symbol, f'Rank_change_{most_recent_date}'].values[0],
                score_rank_df.loc[score_rank_df['Symbol'] == symbol, f'Rank_{most_recent_date}'].values[0],
                score_rank_change_df.loc[score_rank_change_df['Symbol'] == symbol, f'Rank_change_{most_recent_date}'].values[0]
            ])
        
        ax2.axis('off')
        table = ax2.table(cellText=stats_data,
                          colLabels=['Symbol', 'ER_Rank', 'ER_Rank_Change', 'Score_Rank', 'Score_Rank_Change'],
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}_{timestamp}.png'))
        plt.close()
    
    # Plot for Score Rank
    plot_with_stats(score_rank_df, score_rank_change_df, 'Daily Rankings - Score Rank Over Time (Top 20 Stocks)', 'daily_rankings_score_rank')
    
    # Plot for ER Rank
    plot_with_stats(er_rank_df, er_rank_change_df, 'Daily Rankings - ER Rank Over Time (Top 20 Stocks)', 'daily_rankings_er_rank')
    
    # Save DataFrames to CSV
    er_rank_df.to_csv(os.path.join(output_dir, f'er_rank_{timestamp}.csv'), index=False)
    er_rank_change_df.to_csv(os.path.join(output_dir, f'er_rank_change_{timestamp}.csv'), index=False)
    score_rank_df.to_csv(os.path.join(output_dir, f'score_rank_{timestamp}.csv'), index=False)
    score_rank_change_df.to_csv(os.path.join(output_dir, f'score_rank_change_{timestamp}.csv'), index=False)
    
    return er_rank_df, er_rank_change_df, score_rank_df, score_rank_change_df



# 7.15 - create ranking report for current portfolio

def create_holdings_ranking_report(er_rank_df, score_rank_df, output_dir):
    # Get existing portfolio holdings
    holdings_data = r.robinhood.account.build_holdings()
    
    # Get the most recent date in the ranking data
    most_recent_date = max([col for col in er_rank_df.columns if col.startswith('Rank_')])
    
    # Create a DataFrame for holdings that are also in the rankings
    holdings_report = []
    
    for symbol, holding in holdings_data.items():
        if symbol in er_rank_df['Symbol'].values:
            er_rank = er_rank_df.loc[er_rank_df['Symbol'] == symbol, most_recent_date].values[0]
            score_rank = score_rank_df.loc[score_rank_df['Symbol'] == symbol, most_recent_date].values[0]
            
            # Extract relevant information from the holding data
            bought_price = float(holding['average_buy_price'])
            current_price = float(holding['price'])
            quantity = float(holding['quantity'])
            equity = float(holding['equity'])
            percent_change = float(holding['percent_change'])
            equity_change = float(holding['equity_change'])
            type_of_holding = holding['type']
            name = holding['name']
            pe_ratio = float(holding['pe_ratio']) if holding['pe_ratio'] != 'None' else None
            portfolio_percentage = float(holding['percentage'])

            # Calculate returns
            total_return = (current_price - bought_price) / bought_price
            
            holdings_report.append({
                'Symbol': symbol,
                'ER_Rank': er_rank,
                'Score_Rank': score_rank,
                'Name': name,
                'Type': type_of_holding,
                'Quantity': quantity,
                'Bought_Price': bought_price,
                'Current_Price': current_price,
                'Equity': equity,
                'Percent_Change': percent_change,
                'Equity_Change': equity_change,
                'Total_Return': total_return,
                'PE_Ratio': pe_ratio,
                'Portfolio_Percentage': portfolio_percentage
            })
    
    holdings_df = pd.DataFrame(holdings_report)
    
    # Sort by Score_Rank
    holdings_df = holdings_df.sort_values('Score_Rank')
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"holdings_ranking_report_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    holdings_df.to_csv(csv_filepath, index=False)
    
    # Create a plot
    plt.figure(figsize=(12, 6))
    plt.scatter(holdings_df['Portfolio_Percentage'], holdings_df['Total_Return'], 
                c=holdings_df['Score_Rank'], cmap='viridis', s=50)
    plt.colorbar(label='Score Rank')
    plt.xlabel('Portfolio Percentage')
    plt.ylabel('Total Return')
    plt.title('Holdings Performance vs. Portfolio Percentage')
    for i, row in holdings_df.iterrows():
        plt.annotate(row['Symbol'], (row['Portfolio_Percentage'], row['Total_Return']))
    plt.tight_layout()
    
    plot_filename = f"holdings_performance_plot_{timestamp}.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()
    
    return holdings_df, csv_filepath, plot_filepath




# 7.21.24 - User dictated settings for what to use for ranking and how deep to go


def create_rankings_df(validate_df, start_date, end_date, ranking_metric, skip, depth):
    date_range = pd.date_range(start=start_date, end=end_date)
    symbols = validate_df['Symbol'].unique()
    
    # Calculate SPY returns for the entire date range
    spy_data = validate_df[validate_df['Symbol'] == 'SPY'].copy()
    spy_data['Return'] = spy_data['Close Price'].pct_change()
    spy_data = spy_data.set_index('Week')
    spy_returns = spy_data['Return'].reindex(date_range).fillna(0)
    
    if spy_returns.empty:
        print("Error: No SPY data found in validate_df")
        return None
    
    print(f"SPY data shape: {spy_data.shape}")
    print(f"SPY data columns: {spy_data.columns}")
    print(f"spy_returns type: {type(spy_returns)}")
    print(f"spy_returns shape: {spy_returns.shape}")
    print(f"First few values of spy_returns:\n{spy_returns.head()}")
    
    rankings_data = []
    for date in date_range:
        daily_data = validate_df[validate_df['Week'] == date]
        daily_rankings = []
        
        for symbol in symbols:
            stock_data = daily_data[daily_data['Symbol'] == symbol]
            if stock_data.empty:
                continue
            
            # Calculate returns for the last 30 days (or available data)
            symbol_history = validate_df[(validate_df['Symbol'] == symbol) & (validate_df['Week'] <= date)].tail(30)
            returns = symbol_history['Close Price'].pct_change().dropna()
            models = []
            updated_models = []
            
            score_original, best_er_original, _, _, _, score_updated, best_er_updated, _, _ = calculate_roi_score(
                validate_df, daily_data, symbol, spy_returns, models, updated_models
            )
            
            expected_return = sum([stock_data[f'P_Win_{i}d'].iloc[0] * stock_data[f'P_Return_{i}d'].iloc[0] for i in range(1, 15)]) / 14
            
            # Calculate Sharpe ratio using the standard deviation of returns
            sharpe_ratio_original = (best_er_original - 0.03 / 252) / (returns.std() * 0.5 + 1e-8)
            treynor_ratio_original = (best_er_original - 0.03 / 252) / (1.0 * 0.5 + 1e-8)  # Assuming beta = 1.0 for simplicity
            
            score = 0
            if ranking_metric == 'score_original':
                score = score_original
            elif ranking_metric == 'score_updated':
                score = score_updated
            elif ranking_metric == 'expected_return':
                score = expected_return
            elif ranking_metric == 'best_er_original':
                score = best_er_original
            elif ranking_metric == 'sharpe_ratio_original':
                score = sharpe_ratio_original
            elif ranking_metric == 'treynor_ratio_original':
                score = treynor_ratio_original
                
            daily_rankings.append({'Symbol': symbol, 'Score': score, 'Close_Price': stock_data['Close Price'].iloc[0]})
        
        daily_rankings_df = pd.DataFrame(daily_rankings)
        if daily_rankings_df.empty:
            continue
        
        daily_rankings_df = daily_rankings_df.sort_values('Score', ascending=False)
        daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
        daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)
        
        for symbol in symbols:
            symbol_data = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]
            if not symbol_data.empty:
                rankings_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Rank': symbol_data['Rank'].values[0],
                    'Close_Price': symbol_data['Close_Price'].values[0],
                    'Gain_Loss': symbol_data['Gain_Loss'].values[0] if 'Gain_Loss' in symbol_data.columns else np.nan
                })
    
    rankings_df = pd.DataFrame(rankings_data)
    return rankings_df



# 7.19.24 - new section to have report data structured more efficiently and get ready for streamlit ingestion (in /app/Strategy_Play.py)


# import pandas as pd
# import numpy as np

# def create_rankings_df(validate_df, start_date, end_date):
#     date_range = pd.date_range(start=start_date, end=end_date)
#     symbols = validate_df['Symbol'].unique()
    
#     rankings_data = []
#     for date in date_range:
#         daily_data = validate_df[validate_df['Week'] == date]
        
#         # Calculate 'Score' if not present
#         if 'Score' not in daily_data.columns:
#             daily_data['Score'] = daily_data['Close Price']  # Placeholder logic for score calculation
        
#         # Create 'Rank' based on 'Score'
#         daily_data['Rank'] = daily_data['Score'].rank(ascending=False, method='dense').astype(int)
        
#         for symbol in symbols:
#             symbol_data = daily_data[daily_data['Symbol'] == symbol]
#             if not symbol_data.empty:
#                 rankings_data.append({
#                     'Date': date,
#                     'Symbol': symbol,
#                     'Rank': symbol_data['Rank'].values[0],
#                     'Close_Price': symbol_data['Close Price'].values[0],
#                     'Gain_Loss': symbol_data['Gain_Loss'].values[0] if 'Gain_Loss' in symbol_data.columns else np.nan
#                 })
    
#     rankings_df = pd.DataFrame(rankings_data)
#     return rankings_df

def update_strategy_results(rankings_df, initial_investment, strategy_params):
    strategy_results = {
        'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment}
    }
    
    date_range = rankings_df['Date'].unique()
    
    for current_date in date_range:
        daily_rankings_df = rankings_df[rankings_df['Date'] == current_date]
        
        # Implement strategies
        if current_date == date_range[0]:
            top_stocks = daily_rankings_df.nsmallest(20, 'Rank')['Symbol'].tolist()
            
            for strategy in strategy_results:
                invest_amount = strategy_results[strategy]['Cash']
                investment_per_stock = invest_amount / len(top_stocks)
                
                for stock in top_stocks:
                    stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
                    shares = investment_per_stock / stock_price
                    strategy_results[strategy]['Book'].append({
                        'Symbol': stock, 
                        'Buy_Date': current_date, 
                        'Buy_Price': stock_price, 
                        'Shares': shares
                    })
                
                strategy_results[strategy]['Cash'] = 0  # All cash is invested
                strategy_results[strategy]['Daily_Value'].append({'Date': current_date, 'Value': invest_amount})
        
        # Update strategies
        for strategy, data in strategy_results.items():
            total_value = data['Cash']
            new_book = []
            for holding in data['Book']:
                symbol = holding['Symbol']
                buy_price = holding['Buy_Price']
                shares = holding['Shares']
                
                current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
                gain_loss = (current_price - buy_price) / buy_price
                
                holding_value = shares * current_price
                total_value += holding_value
                
                # Strategy-specific sell conditions
                if strategy == 'Strategy_1':
                    days_held = (current_date - holding['Buy_Date']).days
                    if days_held > 0:
                        annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
                        if annualized_gain > strategy_params['Strategy_1']['annualized_gain_threshold'] or gain_loss < strategy_params['Strategy_1']['loss_threshold']:
                            # Sell
                            holding['Sell_Date'] = current_date
                            holding['Sell_Price'] = current_price
                            holding['Gain_Loss'] = gain_loss
                            data['Transactions'].append(holding)
                            data['Cash'] += holding_value
                        else:
                            new_book.append(holding)
                    else:
                        new_book.append(holding)
                elif strategy == 'Strategy_2':
                    if gain_loss > strategy_params['Strategy_2']['gain_threshold'] or gain_loss < strategy_params['Strategy_2']['loss_threshold']:
                        # Sell
                        holding['Sell_Date'] = current_date
                        holding['Sell_Price'] = current_price
                        holding['Gain_Loss'] = gain_loss
                        data['Transactions'].append(holding)
                        data['Cash'] += holding_value
                    else:
                        new_book.append(holding)
                elif strategy == 'Strategy_3':
                    if gain_loss > strategy_params['Strategy_3']['gain_threshold'] or gain_loss < strategy_params['Strategy_3']['loss_threshold']:
                        # Sell
                        holding['Sell_Date'] = current_date
                        holding['Sell_Price'] = current_price
                        holding['Gain_Loss'] = gain_loss
                        data['Transactions'].append(holding)
                        data['Cash'] += holding_value
                    else:
                        new_book.append(holding)
            
            data['Book'] = new_book
            
            # Reinvestment logic (updated to invest all available cash)
            if data['Cash'] > 0:
                top_stocks = daily_rankings_df.nsmallest(20, 'Rank')['Symbol'].tolist()
                investment_per_stock = data['Cash'] / len(top_stocks)
                
                for stock in top_stocks:
                    stock_price = daily_rankings_df[daily_rankings_df['Symbol'] == stock]['Close_Price'].values[0]
                    shares = investment_per_stock / stock_price
                    data['Book'].append({
                        'Symbol': stock, 
                        'Buy_Date': current_date, 
                        'Buy_Price': stock_price,
                        'Shares': shares
                    })
                data['Cash'] = 0  # All cash is reinvested
            
            data['Daily_Value'].append({'Date': current_date, 'Value': total_value})
    
    return strategy_results

def create_strategy_values_df(strategy_results):
    strategy_values = []
    for strategy, data in strategy_results.items():
        for daily_value in data['Daily_Value']:
            strategy_values.append({
                'Date': daily_value['Date'],
                'Strategy': strategy,
                'Value': daily_value['Value']
            })
    
    strategy_values_df = pd.DataFrame(strategy_values)
    
    # Drop duplicate entries if any
    strategy_values_df = strategy_values_df.drop_duplicates(subset=['Date', 'Strategy'])
    
    # Pivot the DataFrame
    strategy_values_df = strategy_values_df.pivot(index='Date', columns='Strategy', values='Value').reset_index()
    
    return strategy_values_df




