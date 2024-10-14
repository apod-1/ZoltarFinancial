# -*- coding: utf-8 -*-
"""

Created on Fri Jul 19 17:18:26 2024
Create a dataframe access to create stratetegy based on prior rankings (and potentially current rankings $)

To kick off, run this: streamlit run Strategy_Play.py

requirements:
    python -m venv streamlit_env
    streamlit_env\Scripts\activate
    pip install streamlit
    import streamlit as st
    
  **  To Launch:  **
    activate myenv
    streamlit_env\Scripts\activate
    cd C:\ Users\apod7\Stockpicker\app    
    streamlit run Strategy_Play.py

@author: ZF
"""

# Standard library imports
import os
import sys
import csv
import json
import pickle
import smtplib
import math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta, date
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third-party library imports
import numpy as np
import pandas as pd
import polars as pl
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
from sqlalchemy import create_engine, select, column, case, func, text, desc, Integer, Float
from sqlalchemy.types import Numeric
# from sqlalchemy.sql import select, case, func
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from pmdarima import auto_arima
from joblib import dump, load
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go

# Local imports
import sys
# sys.path.append('C:/Users/apod7/StockPicker/scripts')
import robin_stocks as r
import os
import io
import base64
import openai
import streamlit as st
import altair as alt

# from plotly.io import to_image

# from openai import OpenAI

# import main_functions
# import prepare_data_functions

# Set random seed for reproducibility
np.random.seed(42)

# Load environment variables
# load_dotenv()
# RH_Login = os.getenv('RH_Login')
# RH_Pass = os.getenv('RH_Pass')
# GMAIL_ACCT = os.getenv('GMAIL_ACCT')
# GMAIL_PASS = os.getenv('GMAIL_PASS')
# OPENAI_API = os.getenv('API_KEY')

# Initialize session state
if 'show_confirmation' not in st.session_state:
    st.session_state.show_confirmation = False
    st.session_state.start_time = 0

# from main_functions import (
#     # create_rankings_df
#     # ,update_strategy_results
#     # ,create_strategy_values_df
#     # ,generate_daily_rankings_strategies
#     calculate_roi_score
    
#     )


# validate_df = pd.read_pickle('validate_df_072024.pkl')
# validate_oot_df = pd.read_pickle(r'C:\Users\apod7\StockPicker\validate_oot_df_072024.pkl')
# stop_date = validate_oot_df['Week'].max()
# end_date = stop_date- relativedelta(days=37) # IF THIS HITS A DATE WHEN THERE IS NO TRADING IT WILL BOMB CURRENTLY ;)


# current version 


# section for library setup - works on both pc and cloud
# Determine the base directory
# if os.path.exists('/mount/src/zoltarfinancial'):
#     # GitHub-Streamlit environment
#     BASE_DIR = '/mount/src/zoltarfinancial'
# else:
#     # Local environment
#     BASE_DIR = 'C:/Users/apod7/StockPicker/app/ZoltarFinancial'

# # Set the data directory
# DATA_DIR = os.path.join(BASE_DIR, 'data')



# 7.21 - back to dealing with spy again
# @st.cache_data(ttl=1*24*3600,persist="disk")
def calculate_roi_score(historical_data, validation_data, symbol, spy_returns, models, updated_models=None, risk_level='High', min_beta=0.1):
    print(f"Calculating ROI score for {symbol}")
    # print(f"spy_returns type in calculate_roi_score: {type(spy_returns)}")
    # print(f"spy_returns shape in calculate_roi_score: {spy_returns.shape}")
    # print(f"First few values of spy_returns in calculate_roi_score:\n{spy_returns.head()}")
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
        p_win_list = []
        p_return_list = []
        er_list = []        
        for period in range(1, 15):
            if f'P_Win_{period}d' not in validation_symbol_data.columns or f'P_Return_{period}d' not in validation_symbol_data.columns:
                print(f"Missing P_Win_{period}d or P_Return_{period}d for {symbol}")
                continue
            
            p_win = validation_symbol_data[f'P_Win_{period}d'].iloc[-1]
            p_return = validation_symbol_data[f'P_Return_{period}d'].iloc[-1]
            er = p_win * p_return
            
            p_win_list.append(p_win)
            p_return_list.append(p_return)
            er_list.append(er)
            
            original_scores[period] = {'p_win': p_win, 'p_return': p_return, 'er': er}
            
            if er > best_er_original:
                best_er_original = er
                best_period_original = period

        print(f"Best original ER for {symbol}: {best_er_original} Best Period: {best_period_original}")


        top_3_returns = sorted(enumerate(p_return_list), key=lambda x: x[1], reverse=True)[:3]
        TstScr6_Top3Return = np.mean([K for _, K in top_3_returns])
        best_period6 = np.mean([i+1 for i, _ in top_3_returns])
        
        top_3_er = sorted(enumerate(er_list), key=lambda x: x[1], reverse=True)[:3]
        TstScr7_Top3ER = np.mean([er for _, er in top_3_er])
        best_period7 = np.mean([i+1 for i, _ in top_3_er])

        best_er_original = TstScr7_Top3ER
        best_er_original = best_period7



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
        symbol_returns = symbol_data['Close_Price'].pct_change().dropna()
        
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
        # print(f"sharpe_ratio_updated: {sharpe_ratio_updated}, treynor_ratio_updated: {treynor_ratio_updated}")
        # print(f"score_original: {score_original}, score_updated: {score_updated}")
        # print(f"alpha_original: {alpha_original}, alpha_updated: {alpha_updated}")
        
        if np.isnan(score_original) or np.isinf(score_original) or np.isnan(score_updated) or np.isinf(score_updated):
            print(f"Invalid score for {symbol}")
            return 0, 0, 0, 0, {}, 0, 0, 0, {}
        
        return score_original, best_er_original, beta, alpha_original, original_scores, score_updated, best_er_updated, alpha_updated, updated_scores

    except Exception as e:
        print(f"Error calculating ROI score for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, {}, 0, 0, 0, {}
    
#8.28.24 not sure eif this original is the issue
# def select_portfolio_with_sectors(df, top_x, omit_first, use_sharpe, market_cap, sectors, industries, risk_level, show_industries, use_bullet_proof):
#     score_column = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
    
#     # Filter based on market cap, sectors, and industries
#     if market_cap != "All":
#         df = df[df['Cap_Size'] == market_cap]
#     if sectors:
#         df = df[df['Sector'].isin(sectors)]
#     if show_industries and industries:
#         df = df[df['Industry'].isin(industries)]
    
#     # Sort and select top stocks
#     df_sorted = df.sort_values(score_column, ascending=False)
#     top_stocks = df_sorted.iloc[omit_first:omit_first+top_x]
    
#     # Implement sector-based selection logic
#     if use_bullet_proof:
#         selected_stocks = []
#         selected_sectors = set()
#         for _, stock in top_stocks.iterrows():
#             if len(selected_stocks) >= top_x:
#                 break
#             if stock['Sector'] not in selected_sectors or len(selected_stocks) < len(sectors):
#                 selected_stocks.append(stock)
#                 selected_sectors.add(stock['Sector'])
#         return pd.DataFrame(selected_stocks)
#     else:
#         return top_stocks


def select_portfolio_with_sectors(df, top_x, omit_first, use_sharpe, market_cap, sectors, industries, risk_level, show_industries, use_bullet_proof):
    print(f"Columns in df: {df.columns}")  # Debug print
    
    score_column = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
    print(f"Looking for score column: {score_column}")  # Debug print
    
    if score_column not in df.columns:
        print(f"Warning: {score_column} not found in columns. Available columns: {df.columns}")
        # Try to find a suitable alternative column
        alternative_columns = [col for col in df.columns if 'Score' in col or 'Risk' in col]
        if alternative_columns:
            score_column = alternative_columns[0]
            print(f"Using alternative column: {score_column}")
        else:
            print("No suitable alternative column found. Using the first numeric column.")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                score_column = numeric_columns[0]
            else:
                raise ValueError("No numeric columns found for sorting.")
    
    # Filter based on market cap, sectors, and industries
    if market_cap != "All":
        df = df[df['Cap_Size'] == market_cap]
    if sectors:
        df = df[df['Sector'].isin(sectors)]
    if show_industries and industries:
        df = df[df['Industry'].isin(industries)]
    
    # Sort and select top stocks
    df_sorted = df.sort_values(score_column, ascending=False)
    
    # Handle None values for omit_first and top_x
    omit_first = 0 if omit_first is None else omit_first
    top_x = len(df_sorted) if top_x is None else top_x
    
    top_stocks = df_sorted.iloc[omit_first:omit_first+top_x]
    
    # Implement sector-based selection logic
    if use_bullet_proof:
        selected_stocks = []
        selected_sectors = set()
        for _, stock in top_stocks.iterrows():
            if len(selected_stocks) >= top_x:
                break
            if stock['Sector'] not in selected_sectors or len(selected_stocks) < len(sectors):
                selected_stocks.append(stock)
                selected_sectors.add(stock['Sector'])
        return pd.DataFrame(selected_stocks)
    else:
        return top_stocks


    
# def update_strategy(strategy, portfolio, current_data, current_date, gain_threshold, loss_threshold):
#     for symbol in list(strategy['Book']):  # Use list() to avoid modifying the list while iterating
#         stock_data = current_data[current_data['Symbol'] == symbol]
#         if not stock_data.empty:
#             current_price = stock_data['Close_Price'].iloc[0]
#             purchase_info = next((t for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy'), None)
#             if purchase_info:
#                 purchase_price = purchase_info['Price']
#                 purchase_date = purchase_info['Date']
#                 days_held = (current_date - purchase_date).days
                
#                 # Strategy 3: Sell if annualized gain is reached or loss threshold is hit
#                 annualized_return = (current_price / purchase_price) ** (365 / days_held) - 1 if days_held > 0 else 0
#                 if annualized_return >= gain_threshold or (current_price - purchase_price) / purchase_price <= loss_threshold:
#                     sell_stock(strategy, symbol, current_price, current_date, days_held)

#     # Buy new stocks
#     for _, stock in portfolio.iterrows():
#         if stock['Symbol'] not in strategy['Book'] and strategy['Cash'] > stock['Close_Price']:
#             shares_to_buy = math.floor(strategy['Cash'] / stock['Close_Price'])
#             if shares_to_buy > 0:
#                 strategy['Cash'] -= shares_to_buy * stock['Close_Price']
#                 strategy['Book'].append(stock['Symbol'])
#                 strategy['Transactions'].append({
#                     'Date': current_date,
#                     'Symbol': stock['Symbol'],
#                     'Action': 'Buy',
#                     'Price': stock['Close_Price'],
#                     'Shares': shares_to_buy,
#                     'Value': shares_to_buy * stock['Close_Price']
#                 })

#     # Calculate daily value
#     daily_value = strategy['Cash']
#     for symbol in strategy['Book']:
#         stock_data = current_data[current_data['Symbol'] == symbol]
#         if not stock_data.empty:
#             current_price = stock_data['Close_Price'].iloc[0]
#             shares = next(t['Shares'] for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy')
#             daily_value += current_price * shares

#     strategy['Daily_Value'].append(daily_value)


# sell stock and update strategy were moved into main run_streamlit_app 9.11.24

def sell_stock(strategy, symbol, current_price, current_date, days_held):
    purchase_info = next(t for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy')
    shares = purchase_info['Shares']
    purchase_price = purchase_info['Price']
    sell_value = current_price * shares
    strategy['Cash'] += sell_value
    strategy['Book'].remove(symbol)
    strategy['Transactions'].append({
        'Date': current_date,
        'Symbol': symbol,
        'Action': 'Sell',
        'Price': current_price,
        'Shares': shares,
        'Value': sell_value,
        'Gain': (current_price - purchase_price) / purchase_price,
        'Days_Held': days_held
    })

# 9.11.24 - ALTERNATE EXECUTION LOGIC
def update_strategy(strategy, portfolio, current_data, current_date, annualized_gain, loss_threshold, 
                    ranking_metric, top_x, omit_first, score_cutoff, enable_panic_sell, 
                    normalized_rank, gauge_trigger, bottom_z_percent):
    # Determine if we should apply panic sell rules
    apply_panic_sell = enable_panic_sell and normalized_rank is not None and gauge_trigger is not None and normalized_rank < gauge_trigger
    print(normalized_rank)
    print(gauge_trigger)
    print(enable_panic_sell)
    print(top_x)
    print(omit_first)
    print(enable_panic_sell and normalized_rank is not None and gauge_trigger is not None and normalized_rank < gauge_trigger)
    # Sell logic
    for symbol in list(strategy['Book']):
        stock_data = current_data[current_data['Symbol'] == symbol]
        if not stock_data.empty:
            current_price = stock_data['Close_Price'].iloc[0]
            purchase_info = next((t for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy'), None)
            if purchase_info:
                purchase_price = purchase_info['Price']
                purchase_date = purchase_info['Date']
                days_held = (current_date - purchase_date).days
                
                 # Calculate the percentile threshold for the current date
                current_date_data = current_data[current_data['Date'] == current_date]
                percentile_threshold = np.percentile(current_date_data[ranking_metric], bottom_z_percent)               
                for symbol in list(strategy['Book']):
                    stock_data = current_data[current_data['Symbol'] == symbol]
                    if not stock_data.empty:
                        current_price = stock_data['Close_Price'].iloc[0]
                        current_rank = stock_data[ranking_metric].iloc[0]
                        purchase_info = next((t for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy'), None)
                        if purchase_info:
                            purchase_price = purchase_info['Price']
                            purchase_date = purchase_info['Date']
                            days_held = (current_date - purchase_date).days
                            
                            # Check if the symbol's ranking is in the bottom Z%
                            is_bottom_z_percent = current_rank <= percentile_threshold
                
                            if apply_panic_sell and is_bottom_z_percent:
                                # Sell if in panic sell mode or if the symbol is in the bottom 50%
                                sell_stock(strategy, symbol, current_price, current_date, days_held)
                            else:
                                annualized_return = (current_price / purchase_price) ** (365 / days_held) - 1 if days_held > 0 else 0
                                if annualized_return >= annualized_gain or (current_price - purchase_price) / purchase_price <= loss_threshold:
                                    sell_stock(strategy, symbol, current_price, current_date, days_held)
    # Buy logic (only if not in panic sell mode)
    if not apply_panic_sell:
        available_cash = strategy['Cash']
        
        # Apply portfolio selection criteria
        if score_cutoff is not None:
            qualified_stocks = portfolio[portfolio[ranking_metric] >= score_cutoff]
        else:
            if ranking_metric in portfolio.columns:
                qualified_stocks = portfolio.sort_values(ranking_metric, ascending=False).iloc[omit_first:omit_first+top_x]
            else:
                print(f"Warning: {ranking_metric} not found in portfolio. Using 'Close_Price' for sorting.")
                qualified_stocks = portfolio.sort_values('Close_Price', ascending=False).iloc[omit_first:omit_first+top_x]
        
        num_stocks_to_buy = len(qualified_stocks)
        
        if num_stocks_to_buy > 0:
            cash_per_stock = available_cash / num_stocks_to_buy
            
            for _, stock in qualified_stocks.iterrows():
                symbol = stock['Symbol']
                if symbol not in strategy['Book']:
                    current_price = stock['Close_Price']
                    shares_to_buy = cash_per_stock / current_price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        strategy['Cash'] -= cost
                        strategy['Book'].append(symbol)
                        strategy['Transactions'].append({
                            'Date': current_date,
                            'Symbol': symbol,
                            'Action': 'Buy',
                            'Price': current_price,
                            'Shares': shares_to_buy,
                            'Value': cost
                        })

    # Calculate daily value
    daily_value = strategy['Cash']
    for symbol in strategy['Book']:
        stock_data = current_data[current_data['Symbol'] == symbol]
        if not stock_data.empty:
            current_price = stock_data['Close_Price'].iloc[0]
            shares = next(t['Shares'] for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy')
            daily_value += current_price * shares

    # Always append the date and daily value, even if no transaction occurred
    if 'Date' not in strategy:
        strategy['Date'] = []
    strategy['Date'].append(current_date)
    strategy['Daily_Value'].append(daily_value)

    
# DEPRECIATED 9.11.24 TO GET MORE GRANULAR ALTERNATE EXECUTION
# def update_strategy(strategy, portfolio, current_data, current_date, annualized_gain, loss_threshold, ranking_metric, top_x, omit_first, score_cutoff):
#     # Sell logic
#     for symbol in list(strategy['Book']):
#         stock_data = current_data[current_data['Symbol'] == symbol]
#         if not stock_data.empty:
#             current_price = stock_data['Close_Price'].iloc[0]
#             purchase_info = next((t for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy'), None)
#             if purchase_info:
#                 purchase_price = purchase_info['Price']
#                 purchase_date = purchase_info['Date']
#                 days_held = (current_date - purchase_date).days
                
#                 annualized_return = (current_price / purchase_price) ** (365 / days_held) - 1 if days_held > 0 else 0
#                 if annualized_return >= annualized_gain or (current_price - purchase_price) / purchase_price <= loss_threshold:
#                     sell_stock(strategy, symbol, current_price, current_date, days_held)

#     # Buy logic
#     available_cash = strategy['Cash']
    
#     # Apply portfolio selection criteria
#     if score_cutoff is not None:
#         qualified_stocks = portfolio[portfolio[ranking_metric] >= score_cutoff]
#     else:
#         qualified_stocks = portfolio.sort_values(ranking_metric, ascending=False).iloc[omit_first:omit_first+top_x]
    
#     num_stocks_to_buy = len(qualified_stocks)
    
#     if num_stocks_to_buy > 0:
#         cash_per_stock = available_cash / num_stocks_to_buy
        
#         for _, stock in qualified_stocks.iterrows():
#             symbol = stock['Symbol']
#             if symbol not in strategy['Book']:
#                 current_price = stock['Close_Price']
#                 shares_to_buy = cash_per_stock / current_price
#                 if shares_to_buy > 0:
#                     cost = shares_to_buy * current_price
#                     strategy['Cash'] -= cost
#                     strategy['Book'].append(symbol)
#                     strategy['Transactions'].append({
#                         'Date': current_date,
#                         'Symbol': symbol,
#                         'Action': 'Buy',
#                         'Price': current_price,
#                         'Shares': shares_to_buy,
#                         'Value': cost
#                     })

#     # Calculate daily value
#     daily_value = strategy['Cash']
#     for symbol in strategy['Book']:
#         stock_data = current_data[current_data['Symbol'] == symbol]
#         if not stock_data.empty:
#             current_price = stock_data['Close_Price'].iloc[0]
#             shares = next(t['Shares'] for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy')
#             daily_value += current_price * shares

#     # Always append the date and daily value, even if no transaction occurred
#     if 'Date' not in strategy:
#         strategy['Date'] = []
#     strategy['Date'].append(current_date)
#     strategy['Daily_Value'].append(daily_value)



# 9.10.24 - replace 8.28 late night version

# generate_daily_rankings was here
    
# 9.17.24 - work on both c and cloud
def get_latest_files(data_dir=None):
    if data_dir is None:
        # Determine the environment and set the appropriate data directory
        if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
            # Cloud environment
            data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
        else:
            # Local environment
            data_dir = '/mount/src/zoltarfinancial/daily_ranks'

    latest_files = {}
    for category in ['high_risk', 'low_risk']:
        files = [f for f in os.listdir(data_dir) if f.startswith(f"{category}_rankings_") and f.endswith(".pkl")]
        if files:
            # Use the file's modification time to determine the latest file
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
            latest_files[category] = latest_file
        else:
            latest_files[category] = None

    return latest_files
    
# 8.28.24
# ydef get_latest_files(data_dir):
#     latest_files = {}
#     for category in ['high_risk', 'low_risk']:
#         files = [f for f in os.listdir(data_dir) if f.startswith(f"{category}_rankings_") and f.endswith(".pkl")]
#         if files:
#             # Use the file's modification time to determine the latest file
#             latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
#             latest_files[category] = latest_file
#         else:
#             latest_files[category] = None
#     return latest_files

# 9.3.24 - need new one for fundamentals_df 
def find_most_recent_file(directory, prefix):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter files that start with the given prefix
    files = [f for f in files if f.startswith(prefix)]
    # Sort files by modification time in descending order
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    # Return the most recent file
    return os.path.join(directory, files[0]) if files else None
   
   # 8.5.24 version  
# @st.cache_data(ttl=1*24*3600, persist="disk")

# # 8.28.24 version 
# def generate_daily_rankings_strategies(selected_df, select_portfolio_func, start_date=None, end_date=None, 
#                                        initial_investment=20000,
#                                        strategy_3_annualized_gain=0.4, strategy_3_loss_threshold=-0.07,
#                                        skip=2, depth=20, ranking_metric='High_Risk_Score',
#                                        use_sharpe=False, use_bullet_proof=False,
#                                        market_cap="All", sectors=None, industries=None):
#     if start_date is None:
#         start_date = selected_df['Date'].min()
#     if end_date is None:
#         end_date = selected_df['Date'].max()

#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)
#     date_range = pd.date_range(start=start_date, end=end_date)

#     # Initialize SPY data
#     spy_data = selected_df[selected_df['Symbol'] == 'SPY'].copy()
#     spy_data['Return'] = spy_data['Close_Price'].pct_change()
#     spy_data = spy_data.set_index('Date')

#     # Create a Series of SPY returns for the entire date range
#     spy_returns = spy_data['Return'].reindex(date_range).fillna(0)

#     if spy_returns.empty:
#         print("Error: No SPY data found in selected_df")
#         return None, None, None, None, None

#     # Initialize rankings DataFrame
#     rankings = pd.DataFrame(columns=['Date', 'Symbol', ranking_metric])

#     # Initialize strategy tracking
#     strategy_results = {
#         'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment}
#     }

#     # Calculate total number of days
#     total_days = len(date_range)

#     # Create a progress bar and progress text
#     progress_bar = st.progress(0)
#     progress_text = st.empty()

#     # Initialize top_ranked_symbols_last_day
#     top_ranked_symbols_last_day = []

#     for i, current_date in enumerate(date_range):
#         # Update progress
#         progress = (i + 1) / total_days
#         progress_bar.progress(progress)
#         progress_text.text(f"Progress: {progress:.2%}")

#         current_data = selected_df[selected_df['Date'] == current_date]
        
#         if current_data.empty:
#             print(f"No data available for date: {current_date}")
#             continue

#         print(f"Processing date: {current_date}")

#         # Calculate rankings for the day
#         daily_rankings = []
#         for _, stock in current_data.iterrows():
#             symbol = stock['Symbol']
#             daily_rankings.append({
#                 'Symbol': symbol,
#                 ranking_metric: stock[ranking_metric],
#                 'Close_Price': stock['Close_Price']
#             })

#         # Sort and update rankings
#         daily_rankings_df = pd.DataFrame(daily_rankings)
#         daily_rankings_df['Date'] = current_date
#         daily_rankings_df = daily_rankings_df.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
#         rankings = pd.concat([rankings, daily_rankings_df], ignore_index=True)

#         # Apply portfolio selection
#         portfolio = select_portfolio_func(daily_rankings_df, depth, skip, use_sharpe, market_cap, sectors, industries)

#         # Update strategy results
#         update_strategy(strategy_results['Strategy_3'], portfolio, current_data, current_date, 
#                         strategy_3_annualized_gain, strategy_3_loss_threshold)

#         # Store top ranked symbols for the last day
#         if i == total_days - 1:
#             top_ranked_symbols_last_day = daily_rankings_df['Symbol'].tolist()[:depth]

#     # Remove progress bar and text after completion
#     progress_bar.empty()
#     progress_text.empty()

#     # Calculate final strategy value
#     strategy_values = {'Strategy_3': strategy_results['Strategy_3']['Daily_Value'][-1] if strategy_results['Strategy_3']['Daily_Value'] else initial_investment}

#     # Prepare summary statistics
#     summary = {
#         'Strategy_3': {
#             'Starting Value': initial_investment,
#             'Final Value': strategy_values['Strategy_3'],
#             'Total Return': (strategy_values['Strategy_3'] - initial_investment) / initial_investment,
#             'Number of Transactions': len(strategy_results['Strategy_3']['Transactions']),
#             'Average Days Held': np.mean([t['Days_Held'] for t in strategy_results['Strategy_3']['Transactions'] if 'Days_Held' in t]) if strategy_results['Strategy_3']['Transactions'] else 0
#         }
#     }

#     return rankings, strategy_results, strategy_values, summary, top_ranked_symbols_last_day

# def update_strategy(strategy, portfolio, current_data, current_date, gain_threshold, loss_threshold):
#     for symbol in list(strategy['Book']):  # Use list() to avoid modifying the list while iterating
#         stock_data = current_data[current_data['Symbol'] == symbol]
#         if not stock_data.empty:
#             current_price = stock_data['Close_Price'].iloc[0]
#             purchase_info = next((t for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy'), None)
#             if purchase_info:
#                 purchase_price = purchase_info['Price']
#                 purchase_date = purchase_info['Date']
#                 days_held = (current_date - purchase_date).days
                
#                 # Strategy 3: Sell if annualized gain is reached or loss threshold is hit
#                 annualized_return = (current_price / purchase_price) ** (365 / days_held) - 1 if days_held > 0 else 0
#                 if annualized_return >= gain_threshold or (current_price - purchase_price) / purchase_price <= loss_threshold:
#                     sell_stock(strategy, symbol, current_price, current_date, days_held)

#     # Buy new stocks
#     for _, stock in portfolio.iterrows():
#         if stock['Symbol'] not in strategy['Book'] and strategy['Cash'] > stock['Close_Price']:
#             shares_to_buy = math.floor(strategy['Cash'] / stock['Close_Price'])
#             if shares_to_buy > 0:
#                 strategy['Cash'] -= shares_to_buy * stock['Close_Price']
#                 strategy['Book'].append(stock['Symbol'])
#                 strategy['Transactions'].append({
#                     'Date': current_date,
#                     'Symbol': stock['Symbol'],
#                     'Action': 'Buy',
#                     'Price': stock['Close_Price'],
#                     'Shares': shares_to_buy,
#                     'Value': shares_to_buy * stock['Close_Price']
#                 })

#     # Calculate daily value
#     daily_value = strategy['Cash']
#     for symbol in strategy['Book']:
#         stock_data = current_data[current_data['Symbol'] == symbol]
#         if not stock_data.empty:
#             current_price = stock_data['Close_Price'].iloc[0]
#             shares = next(t['Shares'] for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy')
#             daily_value += current_price * shares

#     strategy['Daily_Value'].append(daily_value)
#     strategy['Date'].append(current_date)

# def sell_stock(strategy, symbol, current_price, current_date, days_held):
#     shares = next(t['Shares'] for t in reversed(strategy['Transactions']) if t['Symbol'] == symbol and t['Action'] == 'Buy')
#     strategy['Cash'] += current_price * shares
#     strategy['Book'].remove(symbol)
#     strategy['Transactions'].append({
#         'Date': current_date,
#         'Symbol': symbol,
#         'Action': 'Sell',
#         'Price': current_price,
#         'Shares': shares,
#         'Value': current_price * shares,
#         'Days_Held': days_held
#     })


    
    
# 7.31.24 - new version to take advantage of all 28 models in some shape (7 new scores, and 2 new best periods added)

# @st.cache_data(ttl=1*24*3600,persist="disk")
def calculate_multi_roi_score(historical_data, validation_data, symbol, spy_returns, models, updated_models=None, risk_level='High', min_beta=0.1):
    print(f"Calculating multi ROI score for {symbol}")
    try:
        print(f"Processing symbol: {symbol}")
        symbol_data = historical_data[historical_data['Symbol'] == symbol]
        validation_symbol_data = validation_data[validation_data['Symbol'] == symbol]
        print(f"Symbol data length: {len(symbol_data)}, Validation data length: {len(validation_symbol_data)}")

        if len(symbol_data) < 5 or len(validation_symbol_data) < 1:
            print(f"Insufficient data for {symbol}")
            return 0, 0, 0, 0, {}, 0, 0, 0, {}, [0]*7, [0, 0]

        # Original calculations
        best_period_original = 0
        best_er_original = float('-inf')
        original_scores = {}
        p_win_list = []
        p_return_list = []
        er_list = []

        for period in range(1, 15):
            if f'P_Win_{period}d' not in validation_symbol_data.columns or f'P_Return_{period}d' not in validation_symbol_data.columns:
                print(f"Missing P_Win_{period}d or P_Return_{period}d for {symbol}")
                continue
            
            p_win = validation_symbol_data[f'P_Win_{period}d'].iloc[-1]
            p_return = validation_symbol_data[f'P_Return_{period}d'].iloc[-1]
            er = p_win * p_return
            
            p_win_list.append(p_win)
            p_return_list.append(p_return)
            er_list.append(er)
            
            original_scores[period] = {'p_win': p_win, 'p_return': p_return, 'er': er}
            
            if er > best_er_original:
                best_er_original = er
                best_period_original = period

        print(f"Best original ER for {symbol}: {best_er_original} Best Period: {best_period_original}")

        # Calculate additional scores
        TstScr1_AvgWin = np.mean(p_win_list)
        TstScr2_AvgReturn = np.mean(p_return_list)
        TstScr3_AvgER = np.mean(er_list)
        TstScr4_OlympER = np.mean(sorted(er_list)[1:-1])
        TstScr5_Top3Win = np.mean(sorted(p_win_list, reverse=True)[:3])
        
        top_3_returns = sorted(enumerate(p_return_list), key=lambda x: x[1], reverse=True)[:3]
        TstScr6_Top3Return = np.mean([K for _, K in top_3_returns])
        best_period6 = np.mean([i+1 for i, _ in top_3_returns])
        
        top_3_er = sorted(enumerate(er_list), key=lambda x: x[1], reverse=True)[:3]
        TstScr7_Top3ER = np.mean([er for _, er in top_3_er])
        best_period7 = np.mean([i+1 for i, _ in top_3_er])

        best_er_original = TstScr7_Top3ER
        best_period_original = best_period7

        # Updated calculations (unchanged)
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
            print(f"Best updated ER for {symbol}: {best_er_updated} Best Period: {best_period_updated}")
        else:
            print("No updated models provided.")
            best_er_updated = 0
            score_updated = 0
            alpha_updated = 0

        # Calculate other metrics (unchanged)
        symbol_returns = symbol_data['Close_Price'].pct_change().dropna()
        if isinstance(spy_returns, pd.Series):
            market_returns = spy_returns.reindex(symbol_returns.index)
        else:
            print(f"Warning: spy_returns is not a pandas Series. Type: {type(spy_returns)}")
            market_returns = pd.Series(spy_returns, index=symbol_returns.index)
        
        aligned_returns = pd.concat([symbol_returns, market_returns], axis=1, join='inner')
        aligned_returns.columns = ['symbol_returns','market_returns']
        
        if len(aligned_returns) < 5:
            print(f"Insufficient aligned returns for {symbol}")
            return 0, 0, 0, 0, {}, 0, 0, 0, {}, [0]*7, [0, 0]
        
        std_dev = aligned_returns['symbol_returns'].std()
        
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

        if np.isnan(score_original) or np.isinf(score_original) or np.isnan(score_updated) or np.isinf(score_updated):
            print(f"Invalid score for {symbol}")
            return 0, 0, 0, 0, {}, 0, 0, 0, {}, [0]*7, [0, 0]

        additional_scores = [
            TstScr1_AvgWin, TstScr2_AvgReturn, TstScr3_AvgER, TstScr4_OlympER,
            TstScr5_Top3Win, TstScr6_Top3Return, TstScr7_Top3ER
        ]
        best_periods = [best_period6, best_period7]

        return (score_original, best_er_original, beta, alpha_original, original_scores,
                score_updated, best_er_updated, alpha_updated, updated_scores,
                additional_scores, best_periods)

    except Exception as e:
        print(f"Error calculating multi ROI score for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, {}, 0, 0, 0, {}, [0]*7, [0, 0]
    



# @st.cache_data
# def load_data(file_path):
#     return pd.read_pickle(file_path)


# @st.cache_data(ttl=1*24*3600, persist="disk")
def create_strategy_values_df(strategy_results):
    strategy_values = []
    for strategy, data in strategy_results.items():
        for daily_value in data['Daily_Value']:
            try:
                value = float(daily_value['Value'])
            except (ValueError, TypeError):
                print(f"Warning: Invalid value for {strategy} on {daily_value['Date']}: {daily_value['Value']}")
                value = 0.0  # or use the previous valid value if available
            strategy_values.append({
                'DAte': daily_value['Date'],
                'Strategy': strategy,
                'Value': value
            })
    
    strategy_values_df = pd.DataFrame(strategy_values)
    
    # Drop duplicate entries if any
    strategy_values_df = strategy_values_df.drop_duplicates(subset=['Date', 'Strategy'])
    
    # Pivot the DataFrame
    strategy_values_df = strategy_values_df.pivot(index='Date', columns='Strategy', values='Value').reset_index()
    
    return strategy_values_df

# @st.cache_data(ttl=1*24*3600, persist="disk")
def fill_missing_dates(strategy_values_df, _date_range):
    strategy_values_df = strategy_values_df.set_index('Date').reindex(_date_range, method='ffill').reset_index()
    strategy_values_df = strategy_values_df.rename(columns={'index': 'Date'})
    return strategy_values_df

# Set the page configuration at the very top
st.set_page_config(layout="wide")


# 7.26.24 - let user select which file to analyze - Large, Mid, or Small-caps - Streamlit can't handla all to be loaded (not sure about Small actually)
# @st.cache_data(ttl=1*24*3600,persist="disk")
# 8.28.24 - depreciated
# def get_latest_files(data_dir):
#     files = os.listdir(data_dir)
#     latest_files = {'Small': None, 'Mid': None, 'Large': None, 'Tot': None}
    
#     for file in files:
#         if file.startswith('combined_data_') and file.endswith('.pkl') and not file.startswith('spy_'):
#             for category in ['Small', 'Mid', 'Large', 'Tot']:
#                 if category in file:
#                     date_str = file.split('_')[-1].split('.')[0]
#                     date = datetime.strptime(date_str, '%Y%m%d')
#                     if latest_files[category] is None or date > datetime.strptime(latest_files[category].split('_')[-1].split('.')[0], '%Y%m%d'):
#                         latest_files[category] = file
    
#     return latest_files


# 7.26.24 - selection of small, mid, large
# Your existing load_data function
# @st.cache_data(ttl=1*24*3600,persist="disk")
# 8.28.24 - updating to new version of ranks
# def load_data(file_prefix):
    
#     if os.path.exists('/mount/src/zoltarfinancial'):
#         # GitHub-Streamlit environment
#         BASE_DIR1 = '/mount/src/zoltarfinancial'
#     else:
#         # Local environment
#         BASE_DIR1 = 'C:/Users/apod7/StockPicker/app/ZoltarFinancial'

#     # Set the data directory
#     base_dir = os.path.join(BASE_DIR1, 'data')
    
#     # base_dir = "data"
#     today = date.today()
    
#     # Try to load the file with today's date
#     for days_back in range(7):  # Try up to 7 days back
#         current_date = today - timedelta(days=days_back)
#         filename = f"{file_prefix}_{current_date.strftime('%Y%m%d')}.pkl"
#         file_path = os.path.join(base_dir, filename)
#         if os.path.exists(file_path):
#             return pd.read_pickle(file_path)
    
#     # If no file found in the last 7 days, list available files and let user choose
#     st.warning(f"No recent {file_prefix} file found. Please select a file manually.")
#     available_files = [f for f in os.listdir(base_dir) if f.startswith(file_prefix) and f.endswith('.pkl')]
#     if available_files:
#         selected_file = st.selectbox(f"Select a {file_prefix} file:", available_files)
#         return pd.read_pickle(os.path.join(base_dir, selected_file))
#     else:
#         st.error(f"No {file_prefix} files found in the data directory.")
#         return None




# @st.cache_data(persist="disk")
def add_email_to_list(email):
    # Set the path to the /email directory within the user's home directory
    home_dir = os.path.expanduser('~')
    email_dir = os.path.join(home_dir, 'email')
    email_csv_file = os.path.join(email_dir, 'subscribers.csv')
    
    # st.write(f"Current working directory: {os.getcwd()}")
    # st.write(f"Attempting to save to: {os.path.abspath(email_csv_file)}")
    
    try:
        # Validate email format
        if not email or '@' not in email or '.' not in email:
            st.error("Invalid email address format.")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(email_dir, exist_ok=True)
        # st.write(f"Directory created/checked: {email_dir}")
        
        # Initialize emails list
        emails = []
        
        # Check if file exists and read existing emails
        if os.path.exists(email_csv_file):
            # st.write(f"File {email_csv_file} exists. Reading existing emails.")
            with open(email_csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] != 'Email':  # Skip header if present
                        emails.append(row[0])
            # st.write(f"Existing emails: {emails}")
        else:
            st.write()
            # st.write(f"File {email_csv_file} does not exist. It will be created.")
        
        # Add new email if it doesn't exist
        if email not in emails:
            with open(email_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not emails:  # If the file was empty or didn't exist, write the header
                    writer.writerow(['Email'])
                writer.writerow([email])
            
            # st.write(f"Email written to file: {email}")
            
            # Verify file contents
            with open(email_csv_file, 'r') as f:
                contents = f.read()
            # st.write(f"File contents: {contents}")
            
            return True
        else:
            # st.info("Email already exists in the list.")
            return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def print_email_list():
    home_dir = os.path.expanduser('~')
    email_csv_file = os.path.join(home_dir, 'email', 'subscribers.csv')
    
    if os.path.exists(email_csv_file):
        with open(email_csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            emails = [row[0] for row in reader if row and row[0] != 'Email']
            st.write("Email List:")
            for email in emails:
                st.write(email)
    else:
        st.write("No email list found.")

 # os.remove(os.path.join(os.path.expanduser('~'), 'email', 'subscribers.csv'))   

# Define a function to create centered headers
def centered_header(text):
    st.sidebar.markdown(f"<h3 style='text-align: center;'>{text}</h3>", unsafe_allow_html=True)

# Centered header function
def centered_header_main(text):
    st.markdown(f"<h2 style='text-align: center;'>{text}</h2>", unsafe_allow_html=True)

def centered_header_main_small(text):
    st.markdown(f"<h2 style='text-align: center; font-size: 16px;font-weight: bold;'>{text}</h3>", unsafe_allow_html=True)

# Use the centered header function
# centered_header("Latest Iteration Ranks Research")

# @st.cache_data(persist="disk")
def get_image_urls(date):
    base_url = "https://github.com/apod-1/ZoltarFinancial/raw/main/daily_ranks/"
    return [
        f"{base_url}expected_returns_path_Small_{date}.png",
        f"{base_url}expected_returns_path_Mid_{date}.png",
        f"{base_url}expected_returns_path_Large_{date}.png"
    ]

# @st.cache_data(persist="disk")
def get_latest_file(prefix):
    import requests
    url = "https://api.github.com/repos/apod-1/ZoltarFinancial/contents/daily_ranks"
    response = requests.get(url)
    if response.status_code == 200:
        files = [file for file in response.json() if file['name'].startswith(prefix)]
        if files:
            latest_file = max(files, key=lambda x: x['name'])
            return f"https://github.com/apod-1/ZoltarFinancial/raw/main/daily_ranks/{latest_file['name']}"
    return None

# Function to toggle show_image state
def toggle_show_image():
    st.session_state.show_image = not st.session_state.show_image




# Function to generate rankings_df for the last day
# @st.cache_data(ttl=1*24*3600, persist="disk")
# def generate_last_day_rankings(selected_df, end_date, initial_investment, strategy_params, ranking_metric, risk_level, use_sharpe, use_bullet_proof, market_cap, sectors, industries):
#     start_date = end_date - timedelta(days=5)  # Get last 5 days of data

#     rankings, strategy_results, strategy_values, summary, top_ranked_symbols_last_day = generate_daily_rankings_strategies(
#         selected_df,
#         select_portfolio_with_sectors,
#         start_date=start_date,
#         end_date=end_date,
#         initial_investment=20000,
#         strategy_3_annualized_gain=strategy_3_annualized_gain,
#         strategy_3_loss_threshold=strategy_3_loss_threshold,
#         skip=omit_first,
#         depth=top_x,
#         ranking_metric=f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}",
#         use_sharpe=use_sharpe,
#         use_bullet_proof=use_bullet_proof,
#         market_cap=market_cap,
#         sectors=sectors,
#         industries=industries if show_industries else None,
#         risk_level=risk_level,
#         show_industries=show_industries
#     )
    
#     print(f"Generated rankings DataFrame columns: {rankings.columns}")
#     print(f"Generated rankings DataFrame shape: {rankings.shape}")
#     print(f"First few rows of generated rankings DataFrame:\n{rankings.head()}")
#     print(f"Data types of columns:\n{rankings.dtypes}")
    
#     return rankings, top_ranked_symbols_last_day


# @st.cache_data(ttl=1*24*3600, persist="disk")
def generate_last_week_rankings(high_risk_df, low_risk_df, end_date, risk_level='High', use_sharpe=False):
    start_date = end_date - timedelta(days=15)
    
    # Select the appropriate dataframe based on risk level
    selected_df = high_risk_df if risk_level == 'High' else low_risk_df
    
    
    # Get SPY data
    spy_data = selected_df[selected_df['Symbol'] == 'SPY'].copy()
    spy_data['Return'] = spy_data['Close_Price'].pct_change()
    spy_data = spy_data.set_index('Date')
    
    # Create a Series of SPY returns for the last 3 days
    date_range = pd.date_range(start=start_date, end=end_date)
    spy_returns = spy_data['Return'].reindex(date_range).fillna(0)
    
    # Initialize DataFrame to store rankings
    ranking_metric = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
    rankings_df = pd.DataFrame(columns=['Symbol', 'Date', ranking_metric])
    
    for current_date in date_range:
        print(f"Processing date: {current_date}")
        current_data = selected_df[selected_df['Date'] == current_date]
        
        daily_rankings = []
        
        for _, stock in current_data.iterrows():
            symbol = stock['Symbol']
            if symbol == 'SPY':
                continue
            
            daily_rankings.append({
                'Symbol': symbol,
                'Date': current_date,
                ranking_metric: stock[ranking_metric]
            })
        
        # Add daily rankings to the main DataFrame
        rankings_df = pd.concat([rankings_df, pd.DataFrame(daily_rankings)], ignore_index=True)
    
    return rankings_df


# 9.3.24
def convert_to_ranking_format(df, ranking_metric):
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot the dataframe to have dates as columns and symbols as rows
    pivot_df = df.pivot(index='Symbol', columns='Date', values=ranking_metric)
    
    # Reset index to have 'Symbol' as a column
    pivot_df.reset_index(inplace=True)
    
    return pivot_df



# 8.2.24 - late night: use only top x to define strength of portfolio potential
# @st.cache_data(persist="disk")
# depreciated 8.28.24
# def calculate_market_rank_metrics(rankings_df):
#     # Calculate the average TstScr7_Top3ER for each day
#     daily_avg_metric = rankings_df.groupby('Date')['TstScr7_Top3ER'].mean()

#     # Sort the daily average metrics
#     sorted_metrics = daily_avg_metric.sort_values(ascending=False)

#     # Calculate the mean of the top 20 values after omitting the top 2
#     if len(sorted_metrics) > 22:
#         avg_market_rank = sorted_metrics.iloc[2:22].mean()
#     else:
#         avg_market_rank = sorted_metrics.mean()  # Fallback if there are not enough values

#     latest_market_rank = daily_avg_metric.iloc[-1]

#     # Calculate standard deviation
#     std_dev = sorted_metrics.iloc[2:22].std() if len(sorted_metrics) > 22 else sorted_metrics.std()

#     # Calculate low and high settings
#     low_setting = avg_market_rank - 2 * std_dev
#     high_setting = avg_market_rank + 2 * std_dev

#     return avg_market_rank, std_dev, latest_market_rank, low_setting, high_setting


# 8.28.24 - switching to new version



# 8.5 addition
# @st.cache_data(ttl=1*24*3600, persist="disk")
# def prepare_rankings_data(rankings_df, ranking_type):
#     # Rename the 'Rank' column to the date
#     if 'Rank' in rankings_df.columns:
#         date_col = rankings_df.columns[-1]  # Assuming the last column is the date
#         rankings_df = rankings_df.rename(columns={'Rank': date_col})
    
#     # Get the top N stocks based on the last day's ranking
#     last_day = rankings_df.columns[-1]
#     top_stocks = rankings_df.sort_values(by=last_day).head(25)['Symbol'].tolist()
    
#     # Filter the dataframe for these stocks
#     filtered_df = rankings_df[rankings_df['Symbol'].isin(top_stocks)]
    
#     return filtered_df, top_stocks


# 8.5.24 pm - new version to align with main section
# @st.cache_data(ttl=1*24*3600, persist="disk")
def prepare_rankings_data(rankings_df, ranking_type):
    # Get the last date column
    last_date = rankings_df.columns[-1]
    
    # Sort the DataFrame by the last date's ranking, in descending order
    sorted_df = rankings_df.sort_values(by=last_date, ascending=False)
    
    # Get the top 25 stocks based on the last day's ranking
    top_stocks = sorted_df['Symbol'].head(25).tolist()
    
    # Filter the dataframe for these stocks
    filtered_df = rankings_df[rankings_df['Symbol'].isin(top_stocks)]
    
    return filtered_df, top_stocks


# @st.cache_data(ttl=1*24*3600, persist="disk")
# def prepare_rankings_data(rankings_df, ranking_type):
#     # Get the top N stocks based on the last day's ranking
#     last_day = rankings_df.columns[-1]
#     top_stocks = rankings_df.sort_values(by=last_day).head(25)['Symbol'].tolist()
    
#     # Filter the dataframe for these stocks
#     filtered_df = rankings_df[rankings_df['Symbol'].isin(top_stocks)]
    
#     return filtered_df, top_stocks

# def prepare_rankings_data(rankings_df, ranking_type):
#     # Get the top N stocks based on the last day's ranking
#     last_day = rankings_df.columns[-1]
#     top_stocks = rankings_df.sort_values(by=last_day).head(25)['Symbol'].tolist()
    
#     # Filter the dataframe for these stocks
#     filtered_df = rankings_df[rankings_df['Symbol'].isin(top_stocks)]
    
#     return filtered_df, top_stocks

# depreciated 9.3.24
# def display_interactive_rankings(rankings_df, ranking_type):
#     # Prepare data
#     filtered_df, top_stocks = prepare_rankings_data(rankings_df, ranking_type)
    
#     # Dropdown for selecting number of top stocks to display
#     top_n = st.selectbox(f"Select number of top stocks ({ranking_type})", [5, 10, 15, 20, 25], key=f"{ranking_type}_top_n")
    
#     # Create multiselect for choosing which stocks to display
#     selected_stocks = st.multiselect(f"Select stocks to display ({ranking_type})", top_stocks, default=top_stocks[:top_n], key=f"{ranking_type}_stocks")
    
#     # Filter based on user selection
#     display_df = filtered_df[filtered_df['Symbol'].isin(selected_stocks)]
    
#     # Melt the dataframe to long format for plotting
#     try:
#         date_columns = [col for col in display_df.columns if col != 'Symbol']
#         melted_df = display_df.melt(id_vars=['Symbol'], value_vars=date_columns, var_name='Date', value_name='Score')
#         melted_df['Date'] = pd.to_datetime(melted_df['Date']).dt.date  # Convert to date without time

#         # Create the plot
#         fig = go.Figure()
#         for stock in selected_stocks:
#             stock_data = melted_df[melted_df['Symbol'] == stock]
#             fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Score'], mode='lines', name=stock))

#         fig.update_layout(
#             title=f'Top {top_n} Stocks Score Over Time ({ranking_type})',
#             xaxis_title='Date',
#             yaxis_title='Score',
#             xaxis=dict(
#                 tickformat='%Y-%m-%d',  # Format x-axis ticks as YYYY-MM-DD
#                 tickmode='auto',
#                 nticks=10  # Adjust this number to control the density of x-axis labels
#             )
#         )

#         st.plotly_chart(fig)

#         # Display the dataframe
#         st.dataframe(display_df)
#     except Exception as e:
#         st.error(f"Error processing data: {str(e)}")
#         st.write("DataFrame structure:")
#         st.write(display_df.head())
#         st.write("DataFrame columns:")
#         st.write(display_df.columns)

# 9.3.24 - even later - trying to streamline use of fundamentals data and make the whole section persistent
# Define fine-tuning filters outside of any function
# Define fine-tuning filters outside of any function
def create_fine_tuning_filters(merged_df):
    st.sidebar.subheader("Fine-Tuning Parameters")
    
    analyst_rating = st.sidebar.slider("Analyst Rating", 
                               min_value=float(merged_df['Fundamentals_OverallRating'].min()), 
                               max_value=float(merged_df['Fundamentals_OverallRating'].max()), 
                               value=(float(merged_df['Fundamentals_OverallRating'].min()), float(merged_df['Fundamentals_OverallRating'].max())), 
                               key="analyst_rating")
    
    dividend_yield = st.sidebar.slider("Dividend Yield (%)", 
                               min_value=float(merged_df['Fundamentals_Dividends'].min()), 
                               max_value=float(merged_df['Fundamentals_Dividends'].max()), 
                               value=(float(merged_df['Fundamentals_Dividends'].min()), float(merged_df['Fundamentals_Dividends'].max())), 
                               key="dividend_yield")
    
    pe_ratio = st.sidebar.slider("PE Ratio", 
                         min_value=float(merged_df['Fundamentals_PE'].min()), 
                         max_value=float(merged_df['Fundamentals_PE'].max()), 
                         value=(float(merged_df['Fundamentals_PE'].min()), float(merged_df['Fundamentals_PE'].max())), 
                         key="pe_ratio")
    
    market_cap_billions = merged_df['Fundamentals_MarketCap'] / 1e9
    market_cap = st.sidebar.slider("Market Cap (Bn)", 
                           min_value=float(market_cap_billions.min()), 
                           max_value=float(market_cap_billions.max()), 
                           value=(float(market_cap_billions.min()), float(market_cap_billions.max())), 
                           step=float((market_cap_billions.max() - market_cap_billions.min()) / 5), 
                           key="market_cap")
    
    ex_dividend_options = ["All", "Within 2 days","Within 1 week", "Within 1 momnth"]
    ex_dividend_choice = st.sidebar.radio("Dividend", ex_dividend_options, key="ex_dividend")
    
    return analyst_rating, dividend_yield, pe_ratio, market_cap, ex_dividend_choice



# 9.5.24 - new version with limits for user specified dates (not the full thing)

def display_interactive_rankings(rankings_df, ranking_type, fundamentals_df, filters, top_x, date_range, unique_prefix):
    start_date, end_date = date_range
    
    # Merge rankings with fundamentals
    merged_df = rankings_df.merge(fundamentals_df, on='Symbol', how='left')
    
    # Get all date columns
    date_columns = [col for col in merged_df.columns if isinstance(col, pd.Timestamp)]
    
    # Filter date columns based on the selected date range
    date_columns = [col for col in date_columns if start_date <= col <= end_date]
    
    if not date_columns:
        st.error(f"No data available for the selected date range for {ranking_type} rankings.")
        return
    
    # Use the latest date column in the selected range for ranking
    latest_date = max(date_columns)
    ranking_column = latest_date
    
    # Unpack filters
    analyst_rating, dividend_yield, pe_ratio, market_cap, ex_dividend_choice = filters
    
    # Filter the DataFrame based on user selections
    filtered_df = merged_df[
        (merged_df['Fundamentals_OverallRating'].between(*analyst_rating)) &
        ((merged_df['Fundamentals_Dividends'].between(*dividend_yield)) | 
         ((merged_df['Fundamentals_Dividends'].isnull()) & (dividend_yield[0] == 0))) &
        (merged_df['Fundamentals_PE'].between(*pe_ratio)) &
        (merged_df['Fundamentals_MarketCap'].between(*[x * 1e9 for x in market_cap]))
    ]
    
    # Apply Ex-Dividend Date filter
    if ex_dividend_choice != "All":
        today = pd.Timestamp.now().date()
        day_before_exdiv_shft=today + BDay(1)
        two_days_later = day_before_exdiv_shft + pd.Timedelta(days=2)
        one_week_later = day_before_exdiv_shft + pd.Timedelta(days=7)
        one_month_later = day_before_exdiv_shft + pd.Timedelta(days=30)
        
        # Convert today and one_week_later to datetime64[ns]
        today = pd.Timestamp(today)
        day_before_exdiv_shft = pd.Timestamp(day_before_exdiv_shft)
        two_days_later = pd.Timestamp(two_days_later)
        one_week_later = pd.Timestamp(one_week_later)
        one_month_later = pd.Timestamp(one_month_later)
        
        if ex_dividend_choice == "Within 2 days":
            filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'].between(day_before_exdiv_shft, two_days_later)]
        elif ex_dividend_choice ==  "Within 1 week":
            filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'].between(day_before_exdiv_shft, one_week_later)]
        elif ex_dividend_choice ==  "Within 1 month":
            filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'].between(day_before_exdiv_shft, one_month_later)]
    # Sort the filtered DataFrame
    sorted_df = filtered_df.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
    
    # Use top_x to limit the number of stocks displayed
    display_df = sorted_df.head(top_x)
    
    # Multi-select for stocks
    default_stocks = display_df['Symbol'].tolist()
    selected_stocks = st.multiselect(
        f"Select stocks to display ({ranking_type})",
        options=sorted_df['Symbol'].tolist(),
        default=default_stocks,
        key=f"{ranking_type}_stock_multiselect"
    )
    st.session_state[f'{ranking_type}_selected_stocks'] = selected_stocks
    
    # Add custom stock input
    custom_stock = st.text_input(f"Add a custom stock symbol ({ranking_type})", key=f"{ranking_type}_custom_stock")
    if custom_stock and custom_stock in rankings_df['Symbol'].values:
        if custom_stock not in selected_stocks:
            selected_stocks.append(custom_stock)
    elif custom_stock:
        st.warning(f"Symbol '{custom_stock}' not found in the data.")
    
    
    # Plot selected stocks
    fig = go.Figure()
    for symbol in selected_stocks:
        stock_data = rankings_df[rankings_df['Symbol'] == symbol]
        if symbol == custom_stock:
            # Use a dashed line for the custom stock
            fig.add_trace(go.Scatter(x=date_columns, y=stock_data[date_columns].values[0], 
                                     mode='lines', name=symbol, line=dict(dash='dash')))
        else:
            fig.add_trace(go.Scatter(x=date_columns, y=stock_data[date_columns].values[0], 
                                     mode='lines', name=symbol))
    
    fig.update_layout(title=f'Selected Stocks Ranking Over Time ({ranking_type})',
                      xaxis_title='Date',
                      yaxis_title='Ranking',
                      legend_title='Symbols')
    st.plotly_chart(fig)

    # Display the selected stocks
    # 9.15.25 - REMOVED TO INCLUDE FUNDAMENTALS
    # st.write("Last Day Rankings:")
    # display_df = sorted_df[sorted_df['Symbol'].isin(selected_stocks)]
    # st.dataframe(display_df[['Symbol', ranking_column]].style.format({ranking_column: "{:.4f}"}))
    # Display the selected stocks with additional information
    # Display the selected stocks with additional information

    # Get the maximum date from both dataframes
   
    # Calculate the next business day
    next_bd = (end_date + BDay(1)).strftime('%m-%d-%Y')

    st.markdown(f"<h3 style='text-align: center;'>Your {next_bd} Research Results</h3>", unsafe_allow_html=True)
    display_df = sorted_df[sorted_df['Symbol'].isin(selected_stocks)]
    #9.15.24 Email functionality
    #9.15.24 Email functionality
    # email_button_key = f"email_button_{ranking_type}"  # Create a unique key for each ranking type
    # if st.button("Email This Portfolio", key=email_button_key):
    #     email_input_key = f"email_input_{ranking_type}"  # Create a unique key for each email input
    #     user_email = st.text_input("Enter your email address:", key=email_input_key)
    #     if user_email:
    #         send_user_email(user_email, display_df, ranking_type)
    #     else:
    #         st.warning("Please enter your email address.")
    
    email_input_key = f"email_input_{ranking_type}"
    # user_email = st.text_input("Email This Portfolio:", key=email_input_key)
    user_email = st.text_input(
        "Send Your Research:",
        key=email_input_key,
        placeholder="Enter Your Email Here"
    )    
    email_button_key = f"email_button_{ranking_type}"
    # if st.button("Submit", key=email_button_key):
    #     if user_email:
    #         send_user_email(user_email, display_df, ranking_type)
    #     else:
    #         st.warning("Please enter your email address.")            
            
            
    columns_to_display = [
        'Symbol', 
        ranking_column, 
        'Fundamentals_Sector',
        'Fundamentals_Industry',
        'Fundamentals_MarketCap',
        'Fundamentals_PB',
        'Fundamentals_PE',
        'Fundamentals_Float',
        'Fundamentals_SharesOutstanding',
        'Fundamentals_Dividends',
        'Fundamentals_ExDividendDate',
        'Fundamentals_PayableDate'
    ]

    # Format the DataFrame
    formatted_df = display_df[columns_to_display].copy()
    formatted_df = formatted_df.rename(columns={
        ranking_column: 'Ranking',
        'Fundamentals_Sector': 'Sector',
        'Fundamentals_Industry': 'Industry',
        'Fundamentals_MarketCap': 'Market Cap',
        'Fundamentals_PB': 'P/B Ratio',
        'Fundamentals_PE': 'P/E Ratio',
        'Fundamentals_Float': 'Float',
        'Fundamentals_SharesOutstanding': 'Shares Outstanding',
        'Fundamentals_Dividends': 'Dividend Yield',
        'Fundamentals_ExDividendDate': 'Ex-Dividend Date',
        'Fundamentals_PayableDate': 'Payable Date'
    })

    # Format the values
    # formatted_df['Market Cap'] = formatted_df['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B")
    # Convert Market Cap to billions but keep as numeric
    formatted_df['Market Cap'] = formatted_df['Market Cap'] / 1e9
    formatted_df['Dividend Yield'] = formatted_df['Dividend Yield'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) and x != 0 else "-")
    formatted_df['Float'] = formatted_df['Float'].apply(lambda x: f"{x:,.0f}")
    formatted_df['Shares Outstanding'] = formatted_df['Shares Outstanding'].apply(lambda x: f"{x:,.0f}")
    formatted_df['Ex-Dividend Date'] = formatted_df['Ex-Dividend Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else "-")
    formatted_df['Payable Date'] = formatted_df['Payable Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else "-")
    
    if st.button("Email", key=email_button_key):
        if user_email:

            # Assuming you have your selected stocks in a list called 'selected_stocks'
            future_date = high_risk_df['Date'].max()
            future_date = pd.to_datetime(future_date)
            # Convert future_date to a string format suitable for directory naming
            future_date_str = (future_date+BDay(1)).strftime("%Y-%m-%d")

            # future_date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%Y%m%d")
         
            # cap_size = 'All'  # or whatever cap size you're using
            
            # Create selected_stocks list
            selected_stocks = formatted_df['Symbol'].unique().tolist()
            
            # Create a portfolio-like structure for all selected stocks
            portfolio = {
                'selected_stocks': []
            }
            
            for symbol in selected_stocks:
                stock_slice = high_risk_df[high_risk_df['Symbol'] == symbol]
                if not stock_slice.empty:
                    stock_info = stock_slice.iloc[0]
                    portfolio['selected_stocks'].append({
                        'symbol': symbol,
                        'Estimated_Hold_Time': stock_info.get('High_Risk_Score_HoldPeriod', 30),
                        'expected_return': stock_info.get('High_Risk_Score', 0.1)
                    })
            
            # Generate expected returns path
            expected_returns_path, expected_returns_plotly = plot_expected_returns_path(selected_stocks, high_risk_df, 'output_dir', future_date, market_cap)
            # st.image(expected_returns_path, caption="Expected Returns Path for Selected Stocks")
            
            # 9.14.24 - this portion actually works to generate all stocks on one sheet - may be better/more compact view for some pages
            # performance_plot, angles = plot_all_selected_stocks(selected_stocks, high_risk_df, future_date_str, current_time, market_cap)
            # st.image(performance_plot, caption="Performance Plot for Selected Stocks")
            
            # Display angles if needed
            # for symbol, angle in angles.items():
            #     st.write(f"{symbol}: Angle between Expected Return and MA Reflection: {angle:.2f}°")

            send_user_email(user_email, high_risk_df, formatted_df, ranking_type, display_df, market_cap)
        else:
            st.warning("Please enter your email address.")          
            
    # Display the formatted DataFrame
    st.dataframe(formatted_df.style.format({
        'Ranking': "{:.4f}",
        'P/B Ratio': "{:.2f}",
        'Market Cap': "${:.2f}B",  # Format as currency in billions
        'P/E Ratio': "{:.2f}"       
    }))
    display_df['Fundamentals_NumEmployees'] = display_df['Fundamentals_NumEmployees'].apply(lambda x: f"{x:,.0f}")
    display_df['Fundamentals_YearFounded'] = display_df['Fundamentals_YearFounded'].apply(lambda x: f"{x:,.0f}")

# 9.16.24 - later version
# Display additional information for each stock

    # Create selected_stocks list
    selected_stocks = formatted_df['Symbol'].unique().tolist()
    
    # Create a portfolio-like structure for all selected stocks
    portfolio = {
        'selected_stocks': []
    }
    
    for symbol in selected_stocks:
        stock_slice = high_risk_df[high_risk_df['Symbol'] == symbol]
        if not stock_slice.empty:
            stock_info = stock_slice.iloc[0]
            portfolio['selected_stocks'].append({
                'symbol': symbol,
                'Estimated_Hold_Time': stock_info.get('High_Risk_Score_HoldPeriod', 30),
                'expected_return': stock_info.get('High_Risk_Score', 0.1)
            })
    print(formatted_df.columns)

    # Assuming you have your selected stocks in a list called 'selected_stocks'
    future_date = high_risk_df['Date'].max()
    future_date = pd.to_datetime(future_date)
    # Convert future_date to a string format suitable for directory naming
    future_date_str = (future_date+BDay(1)).strftime("%Y-%m-%d")

    # Generate expected returns path
    expected_returns_path, expected_returns_plotly = plot_expected_returns_path(selected_stocks, high_risk_df, 'output_dir', future_date, market_cap) #changed from datetime.now().strftime("%Y%m%d_%H%M%S") 9.21.24
    # st.image(expected_returns_path, caption=f"Expected Returns Path for Selected Stocks")  #{symbol}
    # if isinstance(expected_returns_path, str):
    #     st.image(expected_returns_path, caption="Expected Returns Path for Selected Stocks")
    # elif isinstance(expected_returns_path, tuple) and len(expected_returns_path) > 0:
    #     st.image(expected_returns_path[0], caption="Expected Returns Path for Selected Stocks")
    # else:
    #     st.warning("Expected returns path image not available.")
    
    # Display the Plotly figure
    st.plotly_chart(expected_returns_plotly)
    

    # future_date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    for i, symbol in enumerate(selected_stocks):
        stock_slice = display_df[display_df['Symbol'] == symbol]
        formatted_slice = formatted_df[formatted_df['Symbol'] == symbol]
        high_risk_slice = high_risk_df[high_risk_df['Symbol'] == symbol]
        
        if not stock_slice.empty and not formatted_slice.empty and not high_risk_slice.empty:
            stock_info = stock_slice.iloc[0]
            formatted_info = formatted_slice.iloc[0]
            high_risk_info = high_risk_slice.iloc[0]
            centered_header_main(f"{symbol}")
            
            # Create gauge chart for Overall Rating
            if 'Fundamentals_OverallRating' in stock_info and 'total_ratings' in stock_info:
                overall_rating = stock_info['Fundamentals_OverallRating']
                total_ratings = stock_info['total_ratings']
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = overall_rating,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Rating"},
                    gauge = {
                        'axis': {'range': [0, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 1], 'color': 'red'},
                            {'range': [1, 2], 'color': 'yellow'},
                            {'range': [2, 3], 'color': 'green'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': overall_rating}}))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=50, b=10),
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"**Total Ratings:** {total_ratings}")
            
            col1, col2 = st.columns(2)
            
            
            with col1:
                if 'Fundamentals_CEO' in stock_info:
                    st.write(f"**CEO:** {stock_info['Fundamentals_CEO']}")
                if 'Sector' in formatted_info:
                    st.write(f"**Sector:** {formatted_info['Sector']}")
            
            with col2:
                if 'Fundamentals_NumEmployees' in stock_info:
                    st.write(f"**Employees:** {stock_info['Fundamentals_NumEmployees']}")
                if 'Industry' in formatted_info:
                    st.write(f"**Industry:** {formatted_info['Industry']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Fundamentals_YearFounded' in stock_info:
                    year_founded = stock_info['Fundamentals_YearFounded']
                    if isinstance(year_founded, str):
                        year_founded = year_founded.replace(',', '')
                    try:
                        year_founded = int(float(year_founded))
                        st.write(f"**Year Founded:** {year_founded}")
                    except ValueError:
                        st.write(f"**Year Founded:** {stock_info['Fundamentals_YearFounded']} (Unable to format)")
                if 'Market Cap' in formatted_info:
                    st.write(f"**Market Cap:** ${formatted_info['Market Cap']:.2f}B")
            
            with col2:
                if 'P/B Ratio' in formatted_info:
                    st.write(f"**P/B Ratio:** {formatted_info['P/B Ratio']}")
                if 'P/E Ratio' in formatted_info:
                    st.write(f"**P/E Ratio:** {formatted_info['P/E Ratio']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Float' in formatted_info:
                    st.write(f"**Float:** {formatted_info['Float']}")
                if 'Shares Outstanding' in formatted_info:
                    st.write(f"**Shares Outstanding:** {formatted_info['Shares Outstanding']}")
            
            with col2:
                if 'Dividend Yield' in formatted_info:
                    st.write(f"**Dividend Yield:** {formatted_info['Dividend Yield']}")
                if 'Ex-Dividend Date' in formatted_info:
                    st.write(f"**Ex-Dividend Date:** {formatted_info['Ex-Dividend Date']}")
            
            if 'Payable Date' in formatted_info:
                st.write(f"**Payable Date:** {formatted_info['Payable Date']}")
            
            # Add new information from high_risk_df
            col1, col2 = st.columns(2)
            
            with col1:
                estimated_hold_time = high_risk_info.get('High_Risk_Score_HoldPeriod', 30)
                st.write(f"**Estimated Hold Time:** {estimated_hold_time} days")
            
            with col2:
                expected_return = high_risk_info.get('High_Risk_Score', 0.1)
                st.write(f"**Expected Return:** {expected_return:.2%}")
            
            if 'Fundamentals_Description' in stock_info:
                st.write(f"**Description:** {stock_info['Fundamentals_Description']}")
            
            performance_plot, angle, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, datetime.now().strftime("%Y%m%d_%H%M%S"), market_cap)
            if performance_plot:
                # Generate a unique key for each plotly chart
                chart_key = f"{unique_prefix}_plotly_chart_{symbol}_{i}"
                st.plotly_chart(plotly_fig, key=chart_key)
            else:
                st.write("No performance plot available for this stock.")
            
            st.write("---")
        else:
            st.write(f"### {symbol}")
            st.write("No information available for this stock.")
            st.write("---")
# 10.14.24 - expanded version 2
    # for i, symbol in enumerate(selected_stocks):
    #     stock_slice = display_df[display_df['Symbol'] == symbol]
    #     formatted_slice = formatted_df[formatted_df['Symbol'] == symbol]
    #     if not stock_slice.empty and not formatted_slice.empty:
    #         stock_info = stock_slice.iloc[0]
    #         formatted_info = formatted_slice.iloc[0]
    #         centered_header_main(f"{symbol}")
            
    #         col1, col2 = st.columns(2)
            
    #         with col1:
    #             if 'Fundamentals_CEO' in stock_info:
    #                 st.write(f"**CEO:** {stock_info['Fundamentals_CEO']}")
    #             if 'Sector' in formatted_info:
    #                 st.write(f"**Sector:** {formatted_info['Sector']}")
            
    #         with col2:
    #             if 'Fundamentals_NumEmployees' in stock_info:
    #                 st.write(f"**Employees:** {stock_info['Fundamentals_NumEmployees']}")
    #             if 'Industry' in formatted_info:
    #                 st.write(f"**Industry:** {formatted_info['Industry']}")
            
    #         col1, col2 = st.columns(2)
            
    #         with col1:
    #             if 'Fundamentals_YearFounded' in stock_info:
    #                 year_founded = stock_info['Fundamentals_YearFounded']
    #                 if isinstance(year_founded, str):
    #                     year_founded = year_founded.replace(',', '')
    #                 try:
    #                     year_founded = int(float(year_founded))
    #                     st.write(f"**Year Founded:** {year_founded}")
    #                 except ValueError:
    #                     st.write(f"**Year Founded:** {stock_info['Fundamentals_YearFounded']} (Unable to format)")
    #             if 'Market Cap' in formatted_info:
    #                 st.write(f"**Market Cap:** ${formatted_info['Market Cap']:.2f}B")
            
    #         with col2:
    #             if 'P/B Ratio' in formatted_info:
    #                 st.write(f"**P/B Ratio:** {formatted_info['P/B Ratio']}")
    #             if 'P/E Ratio' in formatted_info:
    #                 st.write(f"**P/E Ratio:** {formatted_info['P/E Ratio']}")
            
    #         col1, col2 = st.columns(2)
            
    #         with col1:
    #             if 'Float' in formatted_info:
    #                 st.write(f"**Float:** {formatted_info['Float']}")
    #             if 'Shares Outstanding' in formatted_info:
    #                 st.write(f"**Shares Outstanding:** {formatted_info['Shares Outstanding']}")
            
    #         with col2:
    #             if 'Dividend Yield' in formatted_info:
    #                 st.write(f"**Dividend Yield:** {formatted_info['Dividend Yield']}")
    #             if 'Ex-Dividend Date' in formatted_info:
    #                 st.write(f"**Ex-Dividend Date:** {formatted_info['Ex-Dividend Date']}")
            
    #         if 'Payable Date' in formatted_info:
    #             st.write(f"**Payable Date:** {formatted_info['Payable Date']}")
            
    #         if 'Fundamentals_Description' in stock_info:
    #             st.write(f"**Description:** {stock_info['Fundamentals_Description']}")
            
    #         performance_plot, angle, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, datetime.now().strftime("%Y%m%d_%H%M%S"), market_cap)
    #         if performance_plot:
    #             # Generate a unique key for each plotly chart
    #             chart_key = f"{unique_prefix}_plotly_chart_{symbol}_{i}"
    #             st.plotly_chart(plotly_fig, key=chart_key)
    #         else:
    #             st.write("No performance plot available for this stock.")
            
    #         st.write("---")
    #     else:
    #         st.write(f"### {symbol}")
    #         st.write("No information available for this stock.")
    #         st.write("---")

# 10.14.24 version 1 of expanded view (need more streamlined )
#     for i, symbol in enumerate(selected_stocks):
#         stock_slice = display_df[display_df['Symbol'] == symbol]
#         if not stock_slice.empty:
#             stock_info = stock_slice.iloc[0]
#             centered_header_main(f"{symbol}")
            
#             if 'Fundamentals_OverallRating' in stock_info and 'total_ratings' in stock_info:
#                 st.write(f"**Overall Rating:** {stock_info['Fundamentals_OverallRating']} | **Total Ratings:** {stock_info['total_ratings']}")
            
#             if 'Fundamentals_Sector' in stock_info and 'Fundamentals_Industry' in stock_info:
#                 st.write(f"**Sector:** {stock_info['Fundamentals_Sector']} | **Industry:** {stock_info['Fundamentals_Industry']}")
            
#             if 'Fundamentals_CEO' in stock_info and 'Fundamentals_NumEmployees' in stock_info:
#                 st.write(f"**CEO:** {stock_info['Fundamentals_CEO']} | **Employees:** {stock_info['Fundamentals_NumEmployees']}")
            
#             if 'Fundamentals_YearFounded' in stock_info:
#                 year_founded = stock_info['Fundamentals_YearFounded']
#                 if isinstance(year_founded, str):
#                     year_founded = year_founded.replace(',', '')
#                 try:
#                     year_founded = int(float(year_founded))
#                     st.write(f"**Year Founded:** {year_founded}")
#                 except ValueError:
#                     st.write(f"**Year Founded:** {stock_info['Fundamentals_YearFounded']} (Unable to format)")
            
#             if 'Fundamentals_Dividends' in stock_info and 'Fundamentals_PE' in stock_info:
#                 st.write(f"**Dividends:** {stock_info['Fundamentals_Dividends']} | **P/E Ratio:** {stock_info['Fundamentals_PE']}")
            
#             if 'Fundamentals_PB' in stock_info and 'Fundamentals_MarketCap' in stock_info:
#                 st.write(f"**P/B Ratio:** {stock_info['Fundamentals_PB']} | **Market Cap:** {stock_info['Fundamentals_MarketCap']}")
            
#             if 'Fundamentals_avgVolume2Weeks' in stock_info and 'Fundamentals_avgVolume30Days' in stock_info:
#                 st.write(f"**Avg Volume (2 Weeks):** {stock_info['Fundamentals_avgVolume2Weeks']} | **Avg Volume (30 Days):** {stock_info['Fundamentals_avgVolume30Days']}")
            
#             if 'Fundamentals_52WeekHigh' in stock_info and 'Fundamentals_52WeekLow' in stock_info:
#                 st.write(f"**52 Week High:** {stock_info['Fundamentals_52WeekHigh']} | **52 Week Low:** {stock_info['Fundamentals_52WeekLow']}")
            
#             if 'Fundamentals_52WeekHighDate' in stock_info and 'Fundamentals_52WeekLowDate' in stock_info:
#                 st.write(f"**52 Week High Date:** {stock_info['Fundamentals_52WeekHighDate']} | **52 Week Low Date:** {stock_info['Fundamentals_52WeekLowDate']}")
            
#             if 'Fundamentals_Float' in stock_info and 'Fundamentals_SharesOutstanding' in stock_info:
#                 st.write(f"**Float:** {stock_info['Fundamentals_Float']} | **Shares Outstanding:** {stock_info['Fundamentals_SharesOutstanding']}")
            
#             if 'Fundamentals_Description' in stock_info:
#                 st.write(f"**Description:** {stock_info['Fundamentals_Description']}")
            
#             performance_plot, angle, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, datetime.now().strftime("%Y%m%d_%H%M%S"), market_cap)
#             if performance_plot:
#                 chart_key = f"{unique_prefix}_plotly_chart_{symbol}_{i}"
#                 st.plotly_chart(plotly_fig, key=chart_key)
#             else:
#                 st.write("No performance plot available for this stock.")
            
#             st.write("---")
#         else:
#             st.write(f"### {symbol}")
#             st.write("No information available for this stock.")
#             st.write("---")

    # Section retired 10.14.24
    # for i, symbol in enumerate(selected_stocks):
    #     stock_slice = display_df[display_df['Symbol'] == symbol]
    #     if not stock_slice.empty:
    #         stock_info = stock_slice.iloc[0]
    #         centered_header_main(f"{symbol}")
    #         if 'Fundamentals_CEO' in stock_info:
    #             st.write(f"**CEO:** {stock_info['Fundamentals_CEO']}")
    #         if 'Fundamentals_NumEmployees' in stock_info:
    #             st.write(f"**Employees:** {stock_info['Fundamentals_NumEmployees']}")
    #         if 'Fundamentals_YearFounded' in stock_info:
    #             year_founded = stock_info['Fundamentals_YearFounded']
    #             if isinstance(year_founded, str):
    #                 year_founded = year_founded.replace(',', '')
    #             try:
    #                 year_founded = int(float(year_founded))
    #                 st.write(f"**Year Founded:** {year_founded}")
    #             except ValueError:
    #                 st.write(f"**Year Founded:** {stock_info['Fundamentals_YearFounded']} (Unable to format)")
    #         if 'Fundamentals_Description' in stock_info:
    #             st.write(f"**Description:** {stock_info['Fundamentals_Description']}")
            
    #         performance_plot, angle, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, datetime.now().strftime("%Y%m%d_%H%M%S"), market_cap)
    #         if performance_plot:
    #             # Generate a unique key for each plotly chart
    #             chart_key = f"{unique_prefix}_plotly_chart_{symbol}_{i}"
    #             st.plotly_chart(plotly_fig, key=chart_key)
    #         else:
    #             st.write("No performance plot available for this stock.")
            
    #         st.write("---")
    #     else:
    #         st.write(f"### {symbol}")
    #         st.write("No information available for this stock.")
    #         st.write("---")
# 9.16 earlier version
    # # Display additional information for each stock
    # for symbol in selected_stocks:
    #     stock_slice = display_df[display_df['Symbol'] == symbol]
    #     if not stock_slice.empty:
    #         stock_info = stock_slice.iloc[0]
    #         st.write(f"### {symbol}")
    #         if 'Fundamentals_CEO' in stock_info:
    #             st.write(f"**CEO:** {stock_info['Fundamentals_CEO']}")
    #         if 'Fundamentals_NumEmployees' in stock_info:
    #             st.write(f"**Employees:** {stock_info['Fundamentals_NumEmployees']}")
    #         if 'Fundamentals_YearFounded' in stock_info:
    #             st.write(f"**Year Founded:** {stock_info['Fundamentals_YearFounded']}")
    #         if 'Fundamentals_Description' in stock_info:
    #             st.write(f"**Description:** {stock_info['Fundamentals_Description']}")
    #         st.write("---")
    #     else:
    #         st.write(f"### {symbol}")
    #         st.write("No information available for this stock.")
    #         st.write("---")

    # # Store the filtered and sorted DataFrame and plot in session state
    # st.session_state[f'{ranking_type}_filtered_df'] = sorted_df
    # st.session_state[f'{ranking_type}_plot'] = fig




    # if st.button("Submit", key=email_button_key):
    #     if user_email:
    #         send_user_email(user_email, formatted_df, ranking_type, display_df)
    #     else:
    #         st.warning("Please enter your email address.")          
              

# Modified display_interactive_rankings function
# # depreciated 9.5.24 - worked fine till then
# def display_interactive_rankings(rankings_df, ranking_type, fundamentals_df, filters, top_x):
#     # Merge rankings with fundamentals
#     merged_df = rankings_df.merge(fundamentals_df, on='Symbol', how='left')
    
#     # Get the latest date column (assuming it's the last timestamp column)
#     date_columns = [col for col in merged_df.columns if isinstance(col, pd.Timestamp)]
#     latest_date = max(date_columns)
    
#     # Use the latest date column for ranking
#     ranking_column = latest_date
    
#     # Unpack filters
#     analyst_rating, dividend_yield, pe_ratio, market_cap, ex_dividend_choice = filters
    
#     # Filter the DataFrame based on user selections
#     filtered_df = merged_df[
#         (merged_df['Fundamentals_OverallRating'].between(*analyst_rating)) &
#         (merged_df['Fundamentals_Dividends'].between(*dividend_yield)) &
#         (merged_df['Fundamentals_PE'].between(*pe_ratio)) &
#         (merged_df['Fundamentals_MarketCap'].between(*[x * 1e9 for x in market_cap]))
#     ]
    
#     # Apply Ex-Dividend Date filter
#     if ex_dividend_choice != "All":
#         today = pd.Timestamp.now().date()
#         two_days_later = today + pd.Timedelta(days=2)
#         one_week_later = today + pd.Timedelta(days=7)
        
#         # Convert today and one_week_later to datetime64[ns]
#         today = pd.Timestamp(today)
#         two_days_later = pd.Timestamp(two_days_later)
#         one_week_later = pd.Timestamp(one_week_later)
        
#         if ex_dividend_choice == "Within 2 days":
#             filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'].between(today, two_days_later)]
#         elif ex_dividend_choice ==  "Within 1 week":
#             filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'].between(today, one_week_later)]
#         # elif ex_dividend_choice == "Within 1 week":
#         #     filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'] > one_week_later]

#     # Sort the filtered DataFrame
#     sorted_df = filtered_df.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
    
#     # Use top_x to limit the number of stocks displayed
#     display_df = sorted_df.head(top_x)
    
#     # Multi-select for stocks
#     default_stocks = display_df['Symbol'].tolist()
#     selected_stocks = st.multiselect(
#         f"Select stocks to display ({ranking_type})",
#         options=sorted_df['Symbol'].tolist(),
#         default=default_stocks,
#         key=f"{ranking_type}_stock_multiselect"
#     )
#     st.session_state[f'{ranking_type}_selected_stocks'] = selected_stocks
    
#     # Add custom stock input
#     custom_stock = st.text_input(f"Add a custom stock symbol ({ranking_type})", key=f"{ranking_type}_custom_stock")
#     if custom_stock and custom_stock in rankings_df['Symbol'].values:
#         if custom_stock not in selected_stocks:
#             selected_stocks.append(custom_stock)
#     elif custom_stock:
#         st.warning(f"Symbol '{custom_stock}' not found in the data.")
    
#     # Display the selected stocks
#     display_df = sorted_df[sorted_df['Symbol'].isin(selected_stocks)]
#     st.dataframe(display_df[['Symbol', ranking_column]].style.format({ranking_column: "{:.4f}"}))
    
#     # Plot selected stocks
#     fig = go.Figure()
#     for symbol in selected_stocks:
#         stock_data = rankings_df[rankings_df['Symbol'] == symbol]
#         if symbol == custom_stock:
#             # Use a dashed line for the custom stock
#             fig.add_trace(go.Scatter(x=date_columns, y=stock_data[date_columns].values[0], 
#                                      mode='lines', name=symbol, line=dict(dash='dash')))
#         else:
#             fig.add_trace(go.Scatter(x=date_columns, y=stock_data[date_columns].values[0], 
#                                      mode='lines', name=symbol))
    
#     fig.update_layout(title=f'Selected Stocks Ranking Over Time ({ranking_type})',
#                       xaxis_title='Date',
#                       yaxis_title='Ranking',
#                       legend_title='Symbols')
#     st.plotly_chart(fig)
    
#     # Store the filtered and sorted DataFrame and plot in session state
#     st.session_state[f'{ranking_type}_filtered_df'] = sorted_df
#     st.session_state[f'{ranking_type}_plot'] = fig

# 9.3.24 - later to include fundamentals_df info in fine tuning filters section
# def display_interactive_rankings(rankings_df, ranking_type, fundamentals_df):
#     st.subheader("Fine-Tuning Parameters")
    
#     # Merge rankings with fundamentals
#     merged_df = rankings_df.merge(fundamentals_df, on='Symbol', how='left')
    
#     # Get the latest date column (assuming it's the last timestamp column)
#     date_columns = [col for col in merged_df.columns if isinstance(col, pd.Timestamp)]
#     latest_date = max(date_columns)
    
#     # Use the latest date column for ranking
#     ranking_column = latest_date
    
#     # Fine-tuning filters
#     analyst_rating = st.slider("Analyst Rating", 
#                                min_value=float(merged_df['Fundamentals_OverallRating'].min()), 
#                                max_value=float(merged_df['Fundamentals_OverallRating'].max()), 
#                                value=(float(merged_df['Fundamentals_OverallRating'].min()), float(merged_df['Fundamentals_OverallRating'].max())), 
#                                key=f"{ranking_type}_analyst_rating")
    
#     dividend_yield = st.slider("Dividend Yield (%)", 
#                                min_value=float(merged_df['Fundamentals_Dividends'].min()), 
#                                max_value=float(merged_df['Fundamentals_Dividends'].max()), 
#                                value=(float(merged_df['Fundamentals_Dividends'].min()), float(merged_df['Fundamentals_Dividends'].max())), 
#                                key=f"{ranking_type}_dividend_yield")
    
#     pe_ratio = st.slider("PE Ratio", 
#                          min_value=float(merged_df['Fundamentals_PE'].min()), 
#                          max_value=float(merged_df['Fundamentals_PE'].max()), 
#                          value=(float(merged_df['Fundamentals_PE'].min()), float(merged_df['Fundamentals_PE'].max())), 
#                          key=f"{ranking_type}_pe_ratio")
    
#     market_cap_billions = merged_df['Fundamentals_MarketCap'] / 1e9
#     market_cap = st.slider("Market Cap (Bn)", 
#                            min_value=float(market_cap_billions.min()), 
#                            max_value=float(market_cap_billions.max()), 
#                            value=(float(market_cap_billions.min()), float(market_cap_billions.max())), 
#                            step=float((market_cap_billions.max() - market_cap_billions.min()) / 5), 
#                            key=f"{ranking_type}_market_cap")
    
#     ex_dividend_options = ["All", "Within 1 week", "After 1 week"]
#     ex_dividend_choice = st.radio("Dividend", ex_dividend_options, key=f"{ranking_type}_ex_dividend")
    
#     # Filter the DataFrame based on user selections
#     filtered_df = merged_df[
#         (merged_df['Fundamentals_OverallRating'].between(*analyst_rating)) &
#         (merged_df['Fundamentals_Dividends'].between(*dividend_yield)) &
#         (merged_df['Fundamentals_PE'].between(*pe_ratio)) &
#         (merged_df['Fundamentals_MarketCap'].between(*[x * 1e9 for x in market_cap]))
#     ]
    
#     # Apply Ex-Dividend Date filter
#     if ex_dividend_choice != "All":
#         today = pd.Timestamp.now().date()
#         one_week_later = today + pd.Timedelta(days=7)
#         if ex_dividend_choice == "Within 1 week":
#             filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'].between(today, one_week_later)]
#         else:  # After 1 week
#             filtered_df = filtered_df[filtered_df['Fundamentals_ExDividendDate'] > one_week_later]
    
#     # Sort the filtered DataFrame
#     sorted_df = filtered_df.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
    
#     # Display top X stocks
#     top_x = st.slider("Number of top stocks to display", min_value=5, max_value=25, value=10, step=5, key=f"{ranking_type}_top_x")
    
#     # Display the top X stocks
#     st.dataframe(sorted_df[['Symbol', ranking_column]].head(top_x).style.format({ranking_column: "{:.4f}"}))
    
#     # Plot top X stocks
#     fig = go.Figure()
#     for symbol in sorted_df['Symbol'].head(top_x):
#         stock_data = rankings_df[rankings_df['Symbol'] == symbol]
#         fig.add_trace(go.Scatter(x=date_columns, y=stock_data[date_columns].values[0], mode='lines', name=symbol))
    
#     fig.update_layout(title=f'Top {top_x} Stocks Ranking Over Time ({ranking_type})',
#                       xaxis_title='Date',
#                       yaxis_title='Ranking',
#                       legend_title='Symbols')
#     st.plotly_chart(fig)
    
#     # Store the filtered and sorted DataFrame in session state
#     st.session_state[f'{ranking_type}_filtered_df'] = sorted_df

# 9.3.24

# def display_interactive_rankings(rankings_df, ranking_type):
    
#     # Prepare data
#     filtered_df, top_stocks = prepare_rankings_data(rankings_df, ranking_type)
    
#     # Dropdown for selecting number of top stocks to display
#     top_n = st.selectbox(f"Select number of top stocks ({ranking_type})", [5, 10, 15, 20, 25], key=f"{ranking_type}_top_n")
    
#     # Create multiselect for choosing which stocks to display
#     selected_stocks = st.multiselect(f"Select stocks to display ({ranking_type})", top_stocks, default=top_stocks[:top_n], key=f"{ranking_type}_stocks")
    
#     # Filter based on user selection
#     display_df = filtered_df[filtered_df['Symbol'].isin(selected_stocks)]
    
#     # Melt the dataframe to long format for plotting
#     try:
#         date_columns = [col for col in display_df.columns if col != 'Symbol']
#         melted_df = display_df.melt(id_vars=['Symbol'], value_vars=date_columns, var_name='Date', value_name='Score')
#         melted_df['Date'] = pd.to_datetime(melted_df['Date']).dt.date  # Convert to date without time

#         # Create the plot
#         fig = go.Figure()
#         for stock in selected_stocks:
#             stock_data = melted_df[melted_df['Symbol'] == stock]
#             fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Score'], mode='lines', name=stock))

#         fig.update_layout(
#             title=f'Top {top_n} Stocks Score Over Time ({ranking_type})',
#             xaxis_title='Date',
#             yaxis_title='Score',
#             xaxis=dict(
#                 tickformat='%Y-%m-%d',  # Format x-axis ticks as YYYY-MM-DD
#                 tickmode='auto',
#                 nticks=10  # Adjust this number to control the density of x-axis labels
#             )
#         )

#         st.plotly_chart(fig)

#         # Display the dataframe
#         st.dataframe(display_df)
#     except Exception as e:
#         st.error(f"Error processing data: {str(e)}")
#         st.write("DataFrame structure:")
#         st.write(display_df.head())
#         st.write("DataFrame columns:")
#         st.write(display_df.columns)



def format_email_table(formatted_df, high_risk_df, ranking_type):
    # Ensure we have a date column, if not use today's date
    if 'Date' not in high_risk_df.columns:
        max_date = pd.Timestamp.now().date()
    else:
        max_date = high_risk_df['Date'].max()

    # Convert max_date to datetime if it's not already
    if not isinstance(max_date, (datetime, pd.Timestamp)):
        try:
            max_date = pd.to_datetime(max_date)
        except:
            max_date = pd.Timestamp.now().date()

    html_table = f"""
    <h2>{ranking_type} Zoltar Ranks for {(max_date + BDay(1)).strftime('%Y-%m-%d')}</h2>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Ranking</th>
            <th>Sector</th>
            <th>Industry</th>
            <th>Market Cap</th>
            <th>P/B Ratio</th>
            <th>P/E Ratio</th>
            <th>Float</th>
            <th>Shares Outstanding</th>
            <th>Dividend Yield</th>
            <th>Ex-Dividend Date</th>
            <th>Payable Date</th>
        </tr>
    """

    for i, row in formatted_df.iterrows():
            # Format Dividend Yield
            div_yield = row.get('Dividend Yield', '')
            if pd.notna(div_yield) and div_yield != '':
                try:
                    div_yield = f"{float(div_yield):.2f}%"
                except ValueError:
                    # If conversion to float fails, keep the original value
                    div_yield = str(div_yield)
    
            html_table += f"""
            <tr>
                <td>{i+1}</td>
                <td>{row['Symbol']}</td>
                <td>{row['Ranking']:.2f}</td>
                <td>{row.get('Sector', '')}</td>
                <td>{row.get('Industry', '')}</td>
                <td>${row['Market Cap']:.1f}B</td>
                <td>{row['P/B Ratio']:.2f}</td>
                <td>{row['P/E Ratio']:.1f}</td>
                <td>{row.get('Float', '')}</td>
                <td>{row.get('Shares Outstanding', '')}</td>
                <td>{div_yield}</td>
                <td>{row.get('Ex-Dividend Date', '')}</td>
                <td>{row.get('Payable Date', '')}</td>
            </tr>
            """

    html_table += "</table>"
    return html_table


# 9.17 new version without png use (still create them though for now)
# import plotly.graph_objects as go
# import os
# from datetime import datetime, timedelta

def plot_expected_returns_path(selected_stocks, high_risk_df, future_date_str, current_time, market_cap):
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    import os
    import plotly.graph_objects as go
    import numpy as np
    
    # Plotly figure
    plotly_fig = go.Figure()
    # 9.21.24 - was not captuting last point in the plot
    # current_date = datetime.now()
    current_date = high_risk_df['Date'].max()+ timedelta(days=1) #changed to be one day ahead

    for symbol in selected_stocks:
        stock_data = high_risk_df[high_risk_df['Symbol'] == symbol].iloc[-1]
        hold_time = stock_data['High_Risk_Score_HoldPeriod']
        expected_return = stock_data['High_Risk_Score']
        
        end_date = current_date + timedelta(days=hold_time)
        plotly_fig.add_trace(go.Scatter(
            x=[current_date, end_date],
            y=[0, expected_return],
            mode='lines+markers',
            name=f"{symbol} ({hold_time:.0f} days)",
            text=[f"Start: {symbol}", f"End: {symbol}<br>Expected Return: {expected_return:.2%}"],
            hoverinfo='text'
        ))
    
    # Plotly settings
    plotly_fig.update_layout(
        title='Expected Returns Path to Exit for Portfolio Stocks',
        xaxis_title='Date',
        yaxis_title='Expected Return',
        yaxis=dict(tickformat='.1%'),  # Format y-axis as percentage with 1 decimal
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='closest'
    )
    
    # Try to save the Matplotlib plot if the directory exists
    filepath = None
    try:
        # Define the base output directory
        base_output_dir = r'C:\Users\apod7\StockPicker\daily_portfolios'
        
        # Create the subdirectory using future_date
        output_dir = os.path.join(base_output_dir, future_date_str)
        
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Matplotlib figure
        plt.figure(figsize=(12, 6))
        
        for symbol in selected_stocks:
            stock_data = high_risk_df[high_risk_df['Symbol'] == symbol].iloc[-1]
            hold_time = stock_data['High_Risk_Score_HoldPeriod']
            expected_return = stock_data['High_Risk_Score']
            
            x = [0, hold_time]
            y = [0, expected_return]
            
            # Matplotlib plot
            plt.plot(x, y, label=f"{symbol} ({hold_time:.0f} days)")
        
        # Matplotlib settings
        plt.xlabel('Days from Today')
        plt.ylabel('Expected Return')
        plt.title('Expected Returns Path to Exit for Portfolio Stocks')
        plt.legend()
        plt.grid(True)
        
        # Save the Matplotlib plot
        filename = f"expected_returns_path_{market_cap}_{current_time}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
    except Exception as e:
        print(f"Unable to save Matplotlib plot: {str(e)}")
    
    return filepath, plotly_fig

# 9.17.24 version that works just fine with pngs
# def plot_expected_returns_path(selected_stocks, high_risk_df, future_date_str, current_time, market_cap):
#     import matplotlib.pyplot as plt
#     from datetime import datetime
#     import os
    
#     # Define the base output directory
#     base_output_dir = r'C:\Users\apod7\StockPicker\daily_portfolios'
    
#     # Create the subdirectory using future_date
#     output_dir = os.path.join(base_output_dir, future_date_str)
    
#     # Ensure the directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.figure(figsize=(12, 6))
    
#     for symbol in selected_stocks:
#         stock_data = high_risk_df[high_risk_df['Symbol'] == symbol].iloc[-1]
#         hold_time = stock_data['High_Risk_Score_HoldPeriod']
#         expected_return = stock_data['High_Risk_Score']  # Using Low_Risk_Score as expected return
        
#         x = [0, hold_time]
#         y = [0, expected_return]
        
#         plt.plot(x, y, label=f"{symbol} ({hold_time:.0f} days)")
    
#     plt.xlabel('Days from Today')
#     plt.ylabel('Expected Return')
#     plt.title('Expected Returns Path to Exit for Portfolio Stocks')
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot with timestamp
#     filename = f"expected_returns_path_{market_cap}_{current_time}.png"
#     filepath = os.path.join(output_dir, filename)
#     plt.savefig(filepath)
#     plt.close()
    
#     return filepath


# def plot_expected_returns_path(selected_stocks, high_risk_df, future_date_str, current_time, cap_size):
#     import matplotlib.pyplot as plt
#     from datetime import datetime
#     import os
    
#     # Define the base output directory
#     base_output_dir = r'C:\Users\apod7\StockPicker\daily_portfolios'
    
#     # Create the subdirectory using future_date
#     output_dir = os.path.join(base_output_dir, future_date_str)
    
#     # Ensure the directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.figure(figsize=(12, 6))
    
#     for symbol in selected_stocks:
#         if isinstance(high_risk_df, pd.DataFrame):
#             stock_data = high_risk_df[high_risk_df['Symbol'] == symbol].iloc[-1]
#         elif isinstance(high_risk_df, dict):
#             stock_data = high_risk_df.get(symbol, {})
#         else:
#             print(f"Unexpected type for high_risk_df: {type(high_risk_df)}")
#             continue

#         hold_time = stock_data.get('Estimated_Hold_Time', 30)  # Default to 30 if not found
#         expected_return = stock_data.get('Low_Risk_Score', 0.1)  # Default to 0.1 if not found
        
#         x = [0, hold_time]
#         y = [0, expected_return]
        
#         plt.plot(x, y, label=f"{symbol} ({hold_time:.0f} days)")
    
#     plt.xlabel('Days from Today')
#     plt.ylabel('Expected Return')
#     plt.title('Expected Returns Path to Exit for Portfolio Stocks')
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot with timestamp
#     filename = f"expected_returns_path_{cap_size}_{current_time}.png"
#     filepath = os.path.join(output_dir, filename)
#     plt.savefig(filepath)
    
#     return filepath


def calculate_angle(slope1, slope2):
    angle_rad = np.arctan((slope2 - slope1) / (1 + slope1 * slope2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def plot_selected_stock(symbol, high_risk_df, future_date_str, current_time, cap_size, days_of_history=100):
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import timedelta
    import os
    import plotly.graph_objects as go

    symbol_data = high_risk_df[high_risk_df['Symbol'] == symbol].sort_values('Date')
    
    if symbol_data.empty:
        return None, None, None

    # validation in raw data 
    # print(high_risk_rankings[high_risk_rankings['Symbol']=='NAT'][high_risk_rankings['Date']==high_risk_rankings['Date'].max()].to_string())

    last_row = symbol_data.iloc[-1]
    start_date = last_row['Date'] - timedelta(days=days_of_history)
    historical_data = symbol_data[symbol_data['Date'] > start_date]
    
    # Calculate moving averages
    historical_data['MA_7'] = historical_data['Close_Price'].rolling(window=7).mean()
    historical_data['MA_14'] = historical_data['Close_Price'].rolling(window=14).mean()
    historical_data['MA_30'] = historical_data['Close_Price'].rolling(window=30).mean()
    
    best_period = last_row['High_Risk_Score_HoldPeriod']
    current_price = last_row['Close_Price']
    expected_return = last_row['High_Risk_Score']
    ma_14 = historical_data['MA_14'].iloc[-1]
    
    # Path 1: Expected Return
    end_price_1 = current_price * (1 + expected_return)
    slope_1 = (end_price_1 - current_price) / best_period
    
    # Path 2: Symmetrical reflection towards MA_14
    below_ma_data = historical_data[historical_data['Close_Price'] < historical_data['MA_14']]
    if not below_ma_data.empty:
        last_below_ma = below_ma_data.iloc[-1]
        days_since_below = (last_row['Date'] - last_below_ma['Date']).days
        if days_since_below > 0:
            slope_to_ma = (current_price - last_below_ma['Close_Price']) / days_since_below
            reflection_slope = -slope_to_ma
            end_price_2 = current_price + reflection_slope * best_period
        else:
            end_price_2 = ma_14
    else:
        end_price_2 = ma_14
    
    end_price_2 = np.clip(end_price_2, min(current_price, ma_14), max(current_price, ma_14))
    slope_2 = (end_price_2 - current_price) / best_period
    
    # Calculate angle between the two predicted paths
    angle = np.arctan2(slope_1, 1) - np.arctan2(slope_2, 1)
    angle = np.degrees(angle)
    angle = -angle  # Changed to minus to show up properly (positive to the upside)
    
    prediction_days = range(int(best_period) + 1)
    prediction_dates = [last_row['Date'] + timedelta(days=day+1) for day in prediction_days]
    
    # Create Plotly figure
    plotly_fig = go.Figure()
    
    # Add historical data
    plotly_fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Close_Price'],
                                    mode='lines', name='Historical'))
    
    # Define a color for 14-day MA and MA Reflection
    ma_14_color = 'red'
    
    # Add moving averages
    plotly_fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['MA_7'],
                                    mode='lines', name='7-day MA', line=dict(dash='dash', color='green')))
    plotly_fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['MA_14'],
                                    mode='lines', name='14-day MA', line=dict(dash='dash', color=ma_14_color)))
    plotly_fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['MA_30'],
                                    mode='lines', name='30-day MA', line=dict(dash='dash', color='purple')))
    
    # Define colors for each trace
    expected_return_color = 'orange'
    ma_reflection_color = ma_14_color
    
    # Add predicted paths
    plotly_fig.add_trace(go.Scatter(
        x=prediction_dates, 
        y=np.linspace(current_price, end_price_1, len(prediction_dates)),
        mode='lines+markers', 
        name='Expected Return', 
        line=dict(dash='dash', color=expected_return_color), 
        marker=dict(symbol='circle'),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>',
        hoverlabel=dict(font=dict(color=expected_return_color))
    ))
    plotly_fig.add_trace(go.Scatter(
        x=prediction_dates, 
        y=np.linspace(current_price, end_price_2, len(prediction_dates)),
        mode='lines+markers', 
        name='MA Reflection', 
        line=dict(dash='dot', color=ma_reflection_color), 
        marker=dict(symbol='square'),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>',
        hoverlabel=dict(font=dict(color=ma_reflection_color))
    ))
    
    plotly_fig.update_layout(
        title=f"{symbol} (Hold Period: {best_period:.0f}d, Expected Return: {expected_return:.2%}, Angle: {angle:.2f}°)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(x=0, y=1, traceorder='normal'),
        yaxis=dict(
            tickprefix='$', 
            tickformat=',.2f',  # Format y-axis labels as dollars with 0 decimals
        ),
        hoverlabel=dict(
            bgcolor="#663399",
            font_size=12,
            font_family="Rockwell"
        )
    )
    
    # Try to save the plot if the directory exists
    filepath = None
    try:
        # Define the base output directory
        base_output_dir = r'C:\Users\apod7\StockPicker\daily_portfolios'
        
        # Create the subdirectory using future_date
        output_dir = os.path.join(base_output_dir, future_date_str)
        
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical data and MAs
        ax.plot(historical_data['Date'], historical_data['Close_Price'], label='Historical', color='blue')
        ax.plot(historical_data['Date'], historical_data['MA_7'], label='7-day MA', color='green', linestyle='--')
        ax.plot(historical_data['Date'], historical_data['MA_14'], label='14-day MA', color='red', linestyle='--')
        ax.plot(historical_data['Date'], historical_data['MA_30'], label='30-day MA', color='purple', linestyle='--')
        
        # Plot predicted paths
        ax.plot(prediction_dates, np.linspace(current_price, end_price_1, len(prediction_dates)), 
                label='Expected Return', color='orange', linestyle='--', marker='o')
        ax.plot(prediction_dates, np.linspace(current_price, end_price_2, len(prediction_dates)), 
                label='MA Reflection', color='red', linestyle=':', marker='s')
        
        ax.set_title(f"{symbol} (Hold Period: {best_period:.0f}d, Expected Return: {expected_return:.2%}, Angle: {angle:.2f}°)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis as dollars with 0 decimals for matplotlib
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.2f}'))
        
        plt.tight_layout()
        
        # Save the plot with timestamp
        filename = f"{symbol}_performance_{cap_size}_{current_time}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)  # Close the figure to free up memory
    except Exception as e:
        print(f"Unable to save matplotlib plot: {str(e)}")
    
    return filepath, angle, plotly_fig


              # other options: background-color: #DDA0DD;
              # Lavender: #E6E6FA
              # xml
              # background-color: #E6E6FA;
              
              # Medium Purple: #9370DB
              # xml
              # background-color: #9370DB;
              
              # Rebecca Purple: #663399
              # xml
              # background-color: #663399;
              
              # Slate Blue: #6A5ACD
              # xml
              # background-color: #6A5ACD;
              
              # Dark Orchid: #9932CC
              # xml
              # background-color: #9932CC;
              
              # Plum: #DDA0DD
              # xml
              # background-color: #DDA0DD;
              
              # Indigo: #4B0082
              # xml
              # background-color: #4B0082;
              
              # Violet: #EE82EE
              # xml
              # background-color: #EE82EE;


# 9.16.24 version that served me well
# def plot_selected_stock(symbol, high_risk_df, future_date_str, current_time, cap_size, days_of_history=90):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from datetime import timedelta
#     import os

#     # Define the base output directory
#     base_output_dir = r'C:\Users\apod7\StockPicker\daily_portfolios'
    
#     # Create the subdirectory using future_date
#     output_dir = os.path.join(base_output_dir, future_date_str)
    
#     # Ensure the directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     symbol_data = high_risk_df[high_risk_df['Symbol'] == symbol].sort_values('Date')
    
#     if symbol_data.empty:
#         ax.text(0.5, 0.5, f"No data for {symbol}", ha='center', va='center')
#         plt.close(fig)
#         return None, None

#     last_row = symbol_data.iloc[-1]
#     start_date = last_row['Date'] - timedelta(days=days_of_history)
#     historical_data = symbol_data[symbol_data['Date'] > start_date]
    
#     # Calculate moving averages
#     historical_data['MA_7'] = historical_data['Close_Price'].rolling(window=7).mean()
#     historical_data['MA_14'] = historical_data['Close_Price'].rolling(window=14).mean()
#     historical_data['MA_30'] = historical_data['Close_Price'].rolling(window=30).mean()
    
#     best_period = last_row['High_Risk_Score_HoldPeriod']
#     current_price = last_row['Close_Price']
#     expected_return = last_row['High_Risk_Score']  # Using Low_Risk_Score as expected return
#     ma_14 = historical_data['MA_14'].iloc[-1]
    
#     # Path 1: Expected Return
#     end_price_1 = current_price * (1 + expected_return)
#     slope_1 = (end_price_1 - current_price) / best_period
    
#     # Path 2: Symmetrical reflection towards MA_14
#     below_ma_data = historical_data[historical_data['Close_Price'] < historical_data['MA_14']]
#     if not below_ma_data.empty:
#         last_below_ma = below_ma_data.iloc[-1]
#         days_since_below = (last_row['Date'] - last_below_ma['Date']).days
#         if days_since_below > 0:
#             slope_to_ma = (current_price - last_below_ma['Close_Price']) / days_since_below
#             reflection_slope = -slope_to_ma
#             end_price_2 = current_price + reflection_slope * best_period
#         else:
#             end_price_2 = ma_14
#     else:
#         end_price_2 = ma_14
    
#     end_price_2 = np.clip(end_price_2, min(current_price, ma_14), max(current_price, ma_14))
#     slope_2 = (end_price_2 - current_price) / best_period
    
#     # Calculate angle between the two predicted paths
#     angle = np.arctan2(slope_1, 1) - np.arctan2(slope_2, 1)
#     angle = np.degrees(angle)
#     angle = -angle  # Changed to minus to show up properly (positive to the upside)
    
#     prediction_days = range(int(best_period) + 1)
#     prediction_dates = [last_row['Date'] + timedelta(days=day) for day in prediction_days]
    
#     # Plot historical data and MAs
#     ax.plot(historical_data['Date'], historical_data['Close_Price'], label='Historical', color='blue')
#     ax.plot(historical_data['Date'], historical_data['MA_7'], label='7-day MA', color='green', linestyle='--')
#     ax.plot(historical_data['Date'], historical_data['MA_14'], label='14-day MA', color='red', linestyle='--')
#     ax.plot(historical_data['Date'], historical_data['MA_30'], label='30-day MA', color='purple', linestyle='--')
    
#     # Plot predicted paths
#     ax.plot(prediction_dates, np.linspace(current_price, end_price_1, len(prediction_dates)), 
#             label='Expected Return', color='orange', linestyle='--', marker='o')
#     ax.plot(prediction_dates, np.linspace(current_price, end_price_2, len(prediction_dates)), 
#             label='MA Reflection', color='cyan', linestyle=':', marker='s')
    
#     ax.set_title(f"{symbol} (Hold: {best_period:.0f}d, ER: {expected_return:.2%}, Angle: {angle:.2f}°)")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Price")
#     ax.legend(loc='upper left', fontsize='small')
#     ax.grid(True)
#     ax.tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
    
#     # Save the plot with timestamp
#     filename = f"{symbol}_performance_{cap_size}_{current_time}.png"
#     filepath = os.path.join(output_dir, filename)
#     plt.savefig(filepath)
#     plt.close(fig)  # Close the figure to free up memory
    
#     return filepath, angle


def plot_all_selected_stocks(selected_stocks, high_risk_df, future_date_str, current_time, cap_size, days_of_history=90):
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import timedelta
    import os

    # Define the base output directory
    base_output_dir = r'C:\Users\apod7\StockPicker\daily_portfolios'
    
    # Create the subdirectory using future_date
    output_dir = os.path.join(base_output_dir, future_date_str)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    num_stocks = len(selected_stocks)
    
    num_cols = 3
    num_rows = math.ceil(num_stocks / num_cols)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows))
    fig.suptitle("Predicted Paths for Selected Stocks", fontsize=16)
    
    angles = {}  # Dictionary to store angles for each stock
    
    for idx, symbol in enumerate(selected_stocks):
        row = idx // num_cols
        col = idx % num_cols
        
        ax = axs[row, col] if num_rows > 1 else axs[col]
        
        symbol_data = high_risk_df[high_risk_df['Symbol'] == symbol].sort_values('Date')
        
        if symbol_data.empty:
            ax.text(0.5, 0.5, f"No data for {symbol}", ha='center', va='center')
            continue
        
        last_row = symbol_data.iloc[-1]
        start_date = last_row['Date'] - timedelta(days=days_of_history)
        historical_data = symbol_data[symbol_data['Date'] > start_date]
        
        best_period = last_row['High_Risk_Score_HoldPeriod']
        current_price = last_row['Close_Price']
        expected_return = last_row['High_Risk_Score']  # Using Low_Risk_Score as expected return
        ma_14 = historical_data['Close_Price'].rolling(window=14).mean().iloc[-1]
        # Calculate moving averages
        historical_data['MA_14'] = historical_data['Close_Price'].rolling(window=14).mean()
        # Calculate additional MAs if not already in the dataframe
        historical_data['MA_7'] = historical_data['Close_Price'].rolling(window=7).mean()
        historical_data['MA_30'] = historical_data['Close_Price'].rolling(window=30).mean()
        
        # Path 1: Expected Return
        end_price_1 = current_price * (1 + expected_return)
        slope_1 = (end_price_1 - current_price) / best_period
        
        # Path 2: Symmetrical reflection towards MA_14
        below_ma_data = historical_data[historical_data['Close_Price'] < historical_data['MA_14']]
        if not below_ma_data.empty:
            last_below_ma = below_ma_data.iloc[-1]
            days_since_below = (last_row['Date'] - last_below_ma['Date']).days
            if days_since_below > 0:
                slope_to_ma = (current_price - last_below_ma['Close_Price']) / days_since_below
                reflection_slope = -slope_to_ma
                end_price_2 = current_price + reflection_slope * best_period
            else:
                end_price_2 = ma_14
        else:
            end_price_2 = ma_14
        
        end_price_2 = np.clip(end_price_2, min(current_price, ma_14), max(current_price, ma_14))
        slope_2 = (end_price_2 - current_price) / best_period
        
        # Calculate angle between the two predicted paths
        angle = np.arctan2(slope_1, 1) - np.arctan2(slope_2, 1)
        angle = np.degrees(angle)
        angles[symbol] = -angle  # Changed to minus to show up properly (positive to the upside)
        
        prediction_days = range(int(best_period) + 1)
        prediction_dates = [last_row['Date'] + timedelta(days=day) for day in prediction_days]
        
        # Plot historical data and MAs
        ax.plot(historical_data['Date'], historical_data['Close_Price'], label='Historical', color='blue')
        ax.plot(historical_data['Date'], historical_data['MA_7'], label='7-day MA', color='green', linestyle='--')
        ax.plot(historical_data['Date'], historical_data['MA_14'], label='14-day MA', color='red', linestyle='--')
        ax.plot(historical_data['Date'], historical_data['MA_30'], label='30-day MA', color='purple', linestyle='--')
        
        # Plot predicted paths
        ax.plot(prediction_dates, np.linspace(current_price, end_price_1, len(prediction_dates)), 
                label='Expected Return', color='orange', linestyle='--', marker='o')
        ax.plot(prediction_dates, np.linspace(current_price, end_price_2, len(prediction_dates)), 
                label='MA Reflection', color='cyan', linestyle=':', marker='s')
        
        ax.set_title(f"{symbol} (Hold: {best_period:.0f}d, ER: {expected_return:.2%}, Angle: {angle:.2f}°)")
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
    filename = f"selected_stocks_performance_{cap_size}_{current_time}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)  # Close the figure to free up memory
    
    return filepath, angles  # Return both the filepath and the angles dictionary


# 9.17.24pm version - using plotly_fig instead of pngs
# import base64
# import io
# from plotly.io import to_image

# def send_user_email(user_email, high_risk_df, formatted_df, ranking_type, display_df, market_cap):
#     subject = f"Your {ranking_type} Stock Rankings from Zoltar Financial"
    
#     # Format the table
#     html_table = format_email_table(formatted_df, high_risk_df, ranking_type)
#     max_date = high_risk_df['Date'].max()
    
#     # Generate expected returns path
#     future_date_str = (max_date + BDay(1)).strftime("%Y-%m-%d")
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     selected_stocks = formatted_df['Symbol'].tolist()
#     expected_returns_path, expected_returns_plotly = plot_expected_returns_path(selected_stocks, high_risk_df, future_date_str, current_time, market_cap)
    
#     # Create additional information HTML
#     additional_info = ""
#     for symbol in formatted_df['Symbol']:
#         stock_slice = display_df[display_df['Symbol'] == symbol]
#         if not stock_slice.empty:
#             stock_info = stock_slice.iloc[0]
#             additional_info += f"<h3 style='text-align: center;'>{symbol}</h3>"
#             if 'Fundamentals_CEO' in stock_info:
#                 additional_info += f"<p><strong>CEO:</strong> {stock_info['Fundamentals_CEO']}</p>"
#             if 'Fundamentals_NumEmployees' in stock_info:
#                 additional_info += f"<p><strong>Employees:</strong> {stock_info['Fundamentals_NumEmployees']:,.0f}</p>"
#             if 'Fundamentals_YearFounded' in stock_info:
#                 additional_info += f"<p><strong>Year Founded:</strong> {stock_info['Fundamentals_YearFounded']:.0f}</p>"
#             if 'Fundamentals_Description' in stock_info:
#                 additional_info += f"<p><strong>Description:</strong> {stock_info['Fundamentals_Description']}</p>"
            
#             # Generate performance plot for individual stock
#             _, angle, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, current_time, market_cap)
#             if plotly_fig:
#                 img_bytes = to_image(plotly_fig, format="png")
#                 img_base64 = base64.b64encode(img_bytes).decode('utf-8')
#                 additional_info += f'<img src="data:image/png;base64,{img_base64}" alt="Performance Plot for {symbol}">'
#                 additional_info += f"<p><strong>Angle between Expected Return and MA Reflection:</strong> {angle:.2f}°</p>"
            
#             additional_info += "<hr>"
#         else:
#             additional_info += f"<h3>{symbol}</h3>"
#             additional_info += "<p>No information available for this stock.</p>"
#             additional_info += "<hr>"

#     # Convert expected returns Plotly figure to base64
#     expected_returns_bytes = to_image(expected_returns_plotly, format="png")
#     expected_returns_base64 = base64.b64encode(expected_returns_bytes).decode('utf-8')

#     # Combine the table and additional information
#     html_content = f"""
#        <html>
#            <body>
#                {html_table}
#                <h2>Expected Returns Path for Selected Stocks</h2>
#                <img src="data:image/png;base64,{expected_returns_base64}" alt="Expected Returns Path">
#                <h2>Additional Stock Information</h2>
#                {additional_info}
#                <p><img src="data:image/png;base64,{get_image_base64()}" alt="ZoltarSurf" style="max-width: 600px; width: 30%; height: auto;"></p>
#                <p>May the riches be with you..</p>
#            </body>
#        </html>
#        """

#     # Create message
#     message = MIMEMultipart()
#     message['From'] = f"Zoltar Financial <{GMAIL_ACCT}>"
#     message['To'] = user_email
#     message['Subject'] = subject

#     # Attach HTML content
#     message.attach(MIMEText(html_content, 'html'))

#     # Send email
#     try:
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login(GMAIL_ACCT, GMAIL_PASS)
#             server.send_message(message)
        
#         st.success("Email sent successfully!")
        
#     except Exception as e:
#         st.error(f"Failed to send email: {str(e)}")

# 9.17.24 8:40pm version - use plotly_fig and expected_returns_plotly
# from plotly.io import to_image
# import base64

# import plotly.io as pio

# import plotly.io as pio
# from io import BytesIO

# 9.17.24 9:48pm - this version fails to append plots
# import plotly.io as pio

# def send_user_email(user_email, high_risk_df, formatted_df, ranking_type, display_df, market_cap):
#     subject = f"Your {ranking_type} Stock Rankings from Zoltar Financial"
    
#     # Format the table
#     html_table = format_email_table(formatted_df, high_risk_df, ranking_type)
#     max_date = high_risk_df['Date'].max()
    
#     # Generate expected returns path
#     future_date_str = (max_date + BDay(1)).strftime("%Y-%m-%d")
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     selected_stocks = formatted_df['Symbol'].tolist()
#     _, expected_returns_plotly = plot_expected_returns_path(selected_stocks, high_risk_df, future_date_str, current_time, market_cap)
    
#     # Convert Plotly figure to HTML
#     expected_returns_html = pio.to_html(expected_returns_plotly, full_html=False, include_plotlyjs='cdn')
    
#     # Create additional information HTML
#     additional_info = ""
#     for symbol in formatted_df['Symbol']:
#         stock_slice = display_df[display_df['Symbol'] == symbol]
#         if not stock_slice.empty:
#             stock_info = stock_slice.iloc[0]
#             additional_info += f"<h3 style='text-align: center;'>{symbol}</h3>"
#             if 'Fundamentals_CEO' in stock_info:
#                 additional_info += f"<p><strong>CEO:</strong> {stock_info['Fundamentals_CEO']}</p>"
#             if 'Fundamentals_NumEmployees' in stock_info:
#                 additional_info += f"<p><strong>Employees:</strong> {stock_info['Fundamentals_NumEmployees']:,.0f}</p>"
#             if 'Fundamentals_YearFounded' in stock_info:
#                 additional_info += f"<p><strong>Year Founded:</strong> {stock_info['Fundamentals_YearFounded']:.0f}</p>"
#             if 'Fundamentals_Description' in stock_info:
#                 additional_info += f"<p><strong>Description:</strong> {stock_info['Fundamentals_Description']}</p>"
            
#             # Generate performance plot for individual stock
#             _, angle, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, current_time, market_cap)
#             if plotly_fig:
#                 stock_html = pio.to_html(plotly_fig, full_html=False, include_plotlyjs='cdn')
#                 additional_info += stock_html
#                 additional_info += f"<p><strong>Angle between Expected Return and MA Reflection:</strong> {angle:.2f}°</p>"
            
#             additional_info += "<hr>"
#         else:
#             additional_info += f"<h3>{symbol}</h3>"
#             additional_info += "<p>No information available for this stock.</p>"
#             additional_info += "<hr>"

#     # Combine the table and additional information
#     html_content = f"""
#     <html>
#         <body>
#             {html_table}
#             <h2>Expected Returns Path for Selected Stocks</h2>
#             {expected_returns_html}
#             <h2>Additional Stock Information</h2>
#             {additional_info}
#             <p>May the riches be with you..</p>
#         </body>
#     </html>
#     """

#     # Create message
#     message = MIMEMultipart()
#     message['From'] = f"Zoltar Financial <{GMAIL_ACCT}>"
#     message['To'] = user_email
#     message['Subject'] = subject

#     # Attach HTML content
#     message.attach(MIMEText(html_content, 'html'))

#     # Send email
#     try:
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login(GMAIL_ACCT, GMAIL_PASS)
#             server.send_message(message)
        
#         st.success("Email sent successfully!")
        
#     except Exception as e:
#         st.error(f"Failed to send email: {str(e)}")

# 9.17 - working version with pngs
def send_user_email(user_email, high_risk_df, formatted_df, ranking_type, display_df, market_cap):

    try:
        sender_email = st.secrets["GMAIL"]["GMAIL_ACCT"]
        sender_password = st.secrets["GMAIL"]["GMAIL_PASS"]
    except:
        # If Streamlit secrets are not available, use environment variables
        sender_email = os.getenv('GMAIL_ACCT')
        sender_password = os.getenv('GMAIL_PASS') 
        st.error("Gmail credentials not found in secrets. Please check your configuration.")
        return
    # try:
    #     sender_email = st.secrets["GMAIL"]["GMAIL_ACCT"]
    #     sender_password = st.secrets["GMAIL"]["GMAIL_PASS"]
    # except KeyError:
    #     st.error("Gmail credentials not found in secrets. Please check your configuration.")
    #     return
    recipient_email = user_email
    subject = f"Your {ranking_type} Stock Rankings from Zoltar Financial"
    
    # Format the table
    html_table = format_email_table(formatted_df, high_risk_df, ranking_type)
    max_date = high_risk_df['Date'].max()
    max_date = pd.to_datetime(max_date)
    # Generate expected returns path
    future_date_str = (max_date + BDay(1)).strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_stocks = formatted_df['Symbol'].tolist()
    expected_returns_path, expected_returns_plotly = plot_expected_returns_path(selected_stocks, high_risk_df, future_date_str, max_date, market_cap) # change to max_date from current_date 9.21.24
    
    # Create additional information HTML
    additional_info = ""
    for symbol in formatted_df['Symbol']:
        stock_slice = display_df[display_df['Symbol'] == symbol]
        if not stock_slice.empty:
            stock_info = stock_slice.iloc[0]
            additional_info += f"<h3 style='text-align: center;'>{symbol}</h3>"
            # additional_info += f"<h3 centered_header_main({symbol})</h3>"
            if 'Fundamentals_CEO' in stock_info:
                additional_info += f"<p><strong>CEO:</strong> {stock_info['Fundamentals_CEO']}</p>"
            if 'Fundamentals_NumEmployees' in stock_info:
                additional_info += f"<p><strong>Employees:</strong> {stock_info['Fundamentals_NumEmployees']:,.0f}</p>"
            if 'Fundamentals_YearFounded' in stock_info:
                additional_info += f"<p><strong>Year Founded:</strong> {stock_info['Fundamentals_YearFounded']:.0f}</p>"
            if 'Fundamentals_Description' in stock_info:
                additional_info += f"<p><strong>Description:</strong> {stock_info['Fundamentals_Description']}</p>"
            
            # Generate performance plot for individual stock
            performance_plot, angle, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, current_time, market_cap)
            if performance_plot:
                additional_info += f'<img src="cid:performance_plot_{symbol}" alt="Performance Plot for {symbol}">'
                # additional_info += f"<p><strong>Angle between Expected Return and MA Reflection:</strong> {angle:.2f}°</p>"
                # this is where the AI assisted response box will be
            
            additional_info += "<hr>"
        else:
            additional_info += f"<h3>{symbol}</h3>"
            additional_info += "<p>No information available for this stock.</p>"
            additional_info += "<hr>"

    # Combine the table and additional information
    html_content = f"""
        <html>
            <body>
                {html_table}
                <h2>Expected Returns Path for Selected Stocks</h2>
                <img src="cid:expected_returns_path" alt="Expected Returns Path">
                <h2>Additional Stock Information</h2>
                {additional_info}
                <p><img src="data:image/png;base64,{get_image_base64()}" alt="ZoltarSurf" style="max-width: 600px; width: 30%; height: auto;"></p>
                <p>May the riches be with you..</p>
            </body>
        </html>
        """

    # Create message
    message = MIMEMultipart()
    message['From'] = f"Zoltar Financial <{sender_email}>"
    message['To'] = user_email
    message['Subject'] = subject

    # Attach HTML content
    message.attach(MIMEText(html_content, 'html'))

    # Attach expected returns path image
    with open(expected_returns_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-ID', '<expected_returns_path>')
        message.attach(img)

    # Attach individual stock performance plots
    for symbol in formatted_df['Symbol']:
        performance_plot, _, plotly_fig = plot_selected_stock(symbol, high_risk_df, future_date_str, current_time, market_cap)
        if performance_plot:
            with open(performance_plot, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', f'<performance_plot_{symbol}>')
                message.attach(img)

    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        st.success("Email sent successfully!")
        
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")

# 9.16.24 - was good but working to add plots :)
# def send_user_email(user_email, formatted_df, ranking_type, display_df):
#     try:
#         # Try to get credentials from environment variables first
#         sender_email = os.environ.get('GMAIL_ACCT')
#         sender_password = os.environ.get('GMAIL_PASS')
        
#         # If not found in environment, try Streamlit secrets
#         if not sender_email or not sender_password:
#             sender_email = st.secrets["GMAIL"]["GMAIL_ACCT"]
#             sender_password = st.secrets["GMAIL"]["GMAIL_PASS"]
        
#         if not sender_email or not sender_password:
#             raise ValueError("Email credentials not found")

#     except Exception as e:
#         st.error(f"Error accessing email credentials: {str(e)}")
#         return

#     subject = f"Your {ranking_type} Stock Rankings from Zoltar Financial"
    
#     # Format the table using the new function
#     html_table = format_email_table(formatted_df, ranking_type)
    
#     # Create additional information HTML
#     additional_info = ""
#     for symbol in formatted_df['Symbol']:
#         stock_slice = display_df[display_df['Symbol'] == symbol]
#         if not stock_slice.empty:
#             stock_info = stock_slice.iloc[0]
#             additional_info += f"<h3>{symbol}</h3>"
#             if 'Fundamentals_CEO' in stock_info:
#                 additional_info += f"<p><strong>CEO:</strong> {stock_info['Fundamentals_CEO']}</p>"
#             if 'Fundamentals_NumEmployees' in stock_info:
#                 additional_info += f"<p><strong>Employees:</strong> {stock_info['Fundamentals_NumEmployees']:,.0f}</p>"
#             if 'Fundamentals_YearFounded' in stock_info:
#                 additional_info += f"<p><strong>Year Founded:</strong> {stock_info['Fundamentals_YearFounded']:.0f}</p>"
#             if 'Fundamentals_Description' in stock_info:
#                 additional_info += f"<p><strong>Description:</strong> {stock_info['Fundamentals_Description']}</p>"
#             additional_info += "<hr>"
#         else:
#             additional_info += f"<h3>{symbol}</h3>"
#             additional_info += "<p>No information available for this stock.</p>"
#             additional_info += "<hr>"

#     # Combine the table and additional information
#     html_content = f"""
#     <html>
#         <body>
#             {html_table}
#             <h2>Additional Stock Information</h2>
#             {additional_info}
#         </body>
#     </html>
#     """

#     # Create message
#     message = MIMEMultipart()
#     message['From'] = f"Zoltar Financial <{sender_email}>"
#     message['To'] = user_email
#     message['Subject'] = subject

#     # Attach HTML content
#     message.attach(MIMEText(html_content, 'html'))

#     # Send email
#     try:
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login(sender_email, sender_password)
#             server.send_message(message)
        
#         st.success("Email sent successfully!")
        
#     except Exception as e:
#         st.error(f"Failed to send email: {str(e)}")

# @st.cache_data(persist="disk")
# def generate_top_20_table(top_ranked_symbols_last_day=None):
#     if 'best_strategy' in st.session_state and st.session_state.best_strategy is not None and 'Top_Ranked_Symbols' in st.session_state.best_strategy:
#         # Use the best strategy data
#         ranking_metric = st.session_state.best_strategy['Settings']['Ranking Metric']
#         max_date = st.session_state.best_strategy.get('Date')
#         top_ranked_symbols = st.session_state.best_strategy['Top_Ranked_Symbols'][:20]
#     elif top_ranked_symbols_last_day is not None:
#         # Use the provided top_ranked_symbols_last_day
#         ranking_metric = 'TstScr7_Top3ER'  # Adjust this if you use a different metric for initial simulation
#         max_date = st.session_state.get('last_simulation_date')
#         top_ranked_symbols = top_ranked_symbols_last_day[:20]
#     else:
#         return "No data available for top ranked symbols."

#     # Ensure max_date is a valid datetime object
#     if max_date is None or max_date == 'Unknown Date':
#         max_date = pd.Timestamp.now().date()
#     else:
#         try:
#             max_date = pd.to_datetime(max_date).date()
#         except Exception as e:
#             st.error(f"Error converting max_date to datetime: {e}. Using current date instead.")
#             max_date = pd.Timestamp.now().date()

#     top_symbols_data = {
#         "Rank": list(range(1, 21)),
#         "Symbol": [symbol['Symbol'] for symbol in top_ranked_symbols],
#         "Score": [f"{symbol[ranking_metric]:.2f}" for symbol in top_ranked_symbols],
#         "Best ER": [f"{symbol['TstScr7_Top3ER'] * 100:.2f}%" for symbol in top_ranked_symbols],
#         "Best Period": [f"{int(symbol['Best_Period7'])}" for symbol in top_ranked_symbols]
#     }

#     html_table = f"""
#     <h2>Top 20 Strategy for {(max_date + BDay(1)).strftime('%Y-%m-%d')}</h2>
#     <table border="1" cellpadding="5" cellspacing="0">
#         <tr>
#             <th>Rank</th>
#             <th>Symbol</th>
#             <th>Score</th>
#             <th>Best ER</th>
#             <th>Best Period</th>
#         </tr>
#     """

#     for i in range(20):
#         html_table += f"""
#         <tr>
#             <td>{top_symbols_data['Rank'][i]}</td>
#             <td>{top_symbols_data['Symbol'][i]}</td>
#             <td>{top_symbols_data['Score'][i]}</td>
#             <td>{top_symbols_data['Best ER'][i]}</td>
#             <td>{top_symbols_data['Best Period'][i]}</td>
#         </tr>
#         """

#     html_table += "</table>"
#     return html_table

# 9.16.24 - fixing formats in table
# def send_user_email(user_email, formatted_df, ranking_type, display_df):
#     subject = f"Your {ranking_type} Stock Rankings from Zoltar Financial"
    
#     # Format Market Cap, P/B Ratio, and P/E Ratio
#     formatted_df['Market Cap'] = formatted_df['Market Cap'].apply(lambda x: f"${x:.1f}B" if x < 1000 else f"${x/1000:.1f}T")
#     formatted_df['P/B Ratio'] = formatted_df['P/B Ratio'].apply(lambda x: f"{x:.2f}")
#     formatted_df['P/E Ratio'] = formatted_df['P/E Ratio'].apply(lambda x: f"{x:.1f}")
    
#     # Convert DataFrame to HTML with improved styling
#     html_table = formatted_df.to_html(index=False, classes="table table-striped table-hover")
    
#     # Create additional information HTML
#     additional_info = ""
#     for symbol in formatted_df['Symbol']:
#         stock_slice = display_df[display_df['Symbol'] == symbol]
#         if not stock_slice.empty:
#             stock_info = stock_slice.iloc[0]
#             additional_info += f"<h3>{symbol}</h3>"
#             if 'Fundamentals_CEO' in stock_info:
#                 additional_info += f"<p><strong>CEO:</strong> {stock_info['Fundamentals_CEO']}</p>"
#             if 'Fundamentals_NumEmployees' in stock_info:
#                 additional_info += f"<p><strong>Employees:</strong> {stock_info['Fundamentals_NumEmployees']:,.0f}</p>"
#             if 'Fundamentals_YearFounded' in stock_info:
#                 additional_info += f"<p><strong>Year Founded:</strong> {stock_info['Fundamentals_YearFounded']:.0f}</p>"
#             if 'Fundamentals_Description' in stock_info:
#                 additional_info += f"<p><strong>Description:</strong> {stock_info['Fundamentals_Description']}</p>"
#             additional_info += "<hr>"
#         else:
#             additional_info += f"<h3>{symbol}</h3>"
#             additional_info += "<p>No information available for this stock.</p>"
#             additional_info += "<hr>"

#     # Combine the table and additional information with improved styling
#     html_content = f"""
#     <html>
#         <head>
#             <style>
#                 body {{ font-family: Arial, sans-serif; }}
#                 .table {{ border-collapse: collapse; width: 100%; }}
#                 .table th, .table td {{ border: 1px solid #ddd; padding: 8px; }}
#                 .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
#                 .table th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #4CAF50; color: white; }}
#             </style>
#         </head>
#         <body>
#             <h1>Zoltar Financial Stock Rankings</h1>
#             <h2>{ranking_type} Stock Rankings</h2>
#             {html_table}
#             <h2>Additional Stock Information</h2>
#             {additional_info}
#             <p>Thank you for using Zoltar Financial services!</p>
#         </body>
#     </html>
#     """

#     # Create message
#     message = MIMEMultipart()
#     message['From'] = f"Zoltar Financial <{GMAIL_ACCT}>"
#     message['To'] = user_email
#     message['Subject'] = subject

#     # Attach HTML content
#     message.attach(MIMEText(html_content, 'html'))

#     # Send email
#     try:
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login(GMAIL_ACCT, GMAIL_PASS)
#             server.send_message(message)
        
#         # Display success message with green button in Streamlit
#         st.success("Email sent successfully!")
#         st.markdown("""
#         <style>
#         .stSuccess {
#             background-color: #4CAF50;
#             color: white;
#             padding: 10px;
#             border-radius: 5px;
#         }
#         </style>
#         """, unsafe_allow_html=True)
        
#         print(f"Email sent to {user_email}")
#     except Exception as e:
#         st.error(f"Failed to send email: {str(e)}")
#         print(f"Error sending email to {user_email}: {str(e)}")
# # 9.16.24 - adding prose about companies
# def send_user_email(user_email, formatted_df, ranking_type, display_df):
#     subject = f"Your {ranking_type} Stock Rankings"
    
#     # Convert DataFrame to HTML
#     html_table = formatted_df.to_html(index=False)
    
#     # Create additional information HTML
#     additional_info = ""
#     for symbol in formatted_df['Symbol']:
#         stock_slice = display_df[display_df['Symbol'] == symbol]
#         if not stock_slice.empty:
#             stock_info = stock_slice.iloc[0]
#             additional_info += f"<h3>{symbol}</h3>"
#             if 'Fundamentals_CEO' in stock_info:
#                 additional_info += f"<p><strong>CEO:</strong> {stock_info['Fundamentals_CEO']}</p>"
#             if 'Fundamentals_NumEmployees' in stock_info:
#                 additional_info += f"<p><strong>Employees:</strong> {stock_info['Fundamentals_NumEmployees']}</p>"
#             if 'Fundamentals_YearFounded' in stock_info:
#                 additional_info += f"<p><strong>Year Founded:</strong> {stock_info['Fundamentals_YearFounded']}</p>"
#             if 'Fundamentals_Description' in stock_info:
#                 additional_info += f"<p><strong>Description:</strong> {stock_info['Fundamentals_Description']}</p>"
#             additional_info += "<hr>"
#         else:
#             additional_info += f"<h3>{symbol}</h3>"
#             additional_info += "<p>No information available for this stock.</p>"
#             additional_info += "<hr>"

#     # Combine the table and additional information
#     html_content = f"""
#     <html>
#         <body>
#             <h2>{ranking_type} Stock Rankings</h2>
#             {html_table}
#             <h2>Additional Stock Information</h2>
#             {additional_info}
#         </body>
#     </html>
#     """

#     # Create message
#     message = MIMEMultipart()
#     message['From'] = GMAIL_ACCT
#     message['To'] = user_email
#     message['Subject'] = subject

#     # Attach HTML content
#     message.attach(MIMEText(html_content, 'html'))

#     # Send email
#     with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#         server.login(GMAIL_ACCT, GMAIL_PASS)
#         server.send_message(message)

#     print(f"Email sent to {user_email}")

# 9.15.25 - with debugging info
# def send_user_email(user_email, formatted_df, ranking_type):
#     try:
#         # Try to get credentials from environment variables first
#         sender_email = os.environ.get('GMAIL_ACCT')
#         sender_password = os.environ.get('GMAIL_PASS')
        
#         # If not found in environment, try Streamlit secrets
#         if not sender_email or not sender_password:
#             sender_email = st.secrets["GMAIL"]["GMAIL_ACCT"]
#             sender_password = st.secrets["GMAIL"]["GMAIL_PASS"]
        
#         if not sender_email or not sender_password:
#             raise ValueError("Email credentials not found")

#     except Exception as e:
#         st.error(f"Error accessing email credentials: {str(e)}")
#         return

#     recipient_email = user_email
#     subject = f"Your {ranking_type.capitalize()} Portfolio (powered by Zoltar)"
    
#     msg = MIMEMultipart()
#     msg['From'] = f"ZF <{sender_email}>"
#     msg['To'] = recipient_email
#     msg['Subject'] = subject
    
#     # Convert DataFrame to HTML table
#     df_html = formatted_df.to_html(index=False, classes='dataframe')
    
#     html_body = f"""
#     <html>
#       <body>
#         <p>Greetings from the ZF community!</p>
#         <h2>Your {ranking_type.capitalize()} Portfolio:</h2>
#         {df_html}
#         <p><img src="data:image/png;base64,{get_image_base64()}" alt="ZoltarSurf"></p>
#         <p>May the riches be with you...</p>
#       </body>
#     </html>
#     """
#     msg.attach(MIMEText(html_body, 'html'))
 
#     try:
#         with smtplib.SMTP('smtp.gmail.com', 587) as server:
#             server.starttls()
#             # st.info(f"Attempting to login with email: {sender_email}")
#             server.login(sender_email, sender_password)
#             server.send_message(msg)
#         st.success('Email sent successfully!')
#         # time.sleep(0.5)
#     except Exception as e:
#         st.error(f'Error sending email: {str(e)}')
        

# 7.15.24 - new version for complete df
# def send_user_email(user_email, formatted_df, ranking_type):
#     try:
#         sender_email = os.getenv('GMAIL_ACCT')
#         sender_password = os.getenv('GMAIL_PASS') 
#     except:
#         # If Streamlit secrets are not available, use environment variables
#         sender_email = st.secrets["GMAIL"]["GMAIL_ACCT"]
#         sender_password = st.secrets["GMAIL"]["GMAIL_PASS"]

#     recipient_email = user_email
#     subject = f"Your {ranking_type.capitalize()} Portfolio (powered by Zoltar)"
    
#     msg = MIMEMultipart()
#     msg['From'] = f"ZF <{sender_email}>"
#     msg['To'] = recipient_email
#     msg['Subject'] = subject
    
#     # Convert DataFrame to HTML table
#     df_html = formatted_df.to_html(index=False, classes='dataframe')
    
#     html_body = f"""
#     <html>
#       <body>
#         <p>Greetings from the ZF community!</p>
#         <h2>Your {ranking_type.capitalize()} Portfolio:</h2>
#         {df_html}
#         <p><img src="data:image/png;base64,{get_image_base64()}" alt="ZoltarSurf"></p>
#         <p>May the riches be with you...</p>
#       </body>
#     </html>
#     """
#     msg.attach(MIMEText(html_body, 'html'))
 
#     try:
#         with smtplib.SMTP('smtp.gmail.com', 587) as server:
#             server.starttls()
#             server.login(sender_email, sender_password)
#             server.send_message(msg)
#         st.success('Email sent successfully!')
#     except Exception as e:
#         st.error(f'Error sending email: {e}')

# def send_user_email(user_email):
#     try:
#         sender_email = os.getenv('GMAIL_ACCT')
#         sender_password = os.getenv('GMAIL_PASS') 
#     except:
#         # If Streamlit secrets are not available, use environment variables
#         sender_email = st.secrets["GMAIL"]["GMAIL_ACCT"]
#         sender_password = st.secrets["GMAIL"]["GMAIL_PASS"]
#         return

#     recipient_email = user_email
#     subject = "Your Top 20 Strategy (powered by Zoltar)"
    
#     msg = MIMEMultipart()
#     msg['From'] = f"ZF <{sender_email}>"
#     msg['To'] = recipient_email
#     msg['Subject'] = subject
    
#     top_ranked_symbols_last_day = st.session_state.get('top_ranked_symbols_last_day')
#     top_20_table = generate_top_20_table(top_ranked_symbols_last_day)
    
#     html_body = f"""
#     <html>
#       <body>
#         <p>Establishing communication with ZF community (phase 1 complete).</p>
#         {top_20_table}
#         <p><img src="data:image/png;base64,{get_image_base64()}" alt="ZoltarSurf"></p>
#         <p>May the riches be with you..</p>
#       </body>
#     </html>
#     """
#     msg.attach(MIMEText(html_body, 'html'))
 
#     try:
#         with smtplib.SMTP('smtp.gmail.com', 587) as server:
#             server.starttls()
#             server.login(sender_email, sender_password)
#             server.send_message(msg)
#         st.success('Email sent successfully!')
#     except Exception as e:
#         st.error(f'Error sending email: {e}')
        
        
        

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
    import requests
    # Function to fetch and encode the image as base64
    image_url = 'https://github.com/apod-1/ZoltarFinancial/raw/main/docs/ZoltarSurf2.png'
    response = requests.get(image_url)
    if response.status_code == 200:
        img_data = response.content
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return img_base64
    else:
        print(f"Failed to fetch image. Status code: {response.status_code}")
        return None


# 8.2.24 - will use this version once we are going off of a repository of these (to save runtime and get more precise)
# This version uses stdev - may be ok but outliers will be an issue
# def calculate_market_rank_metrics(rankings_df):
#     # Calculate the average TstScr7_Top3ER for each day
#     daily_avg_metric = rankings_df.groupby('Date')['TstScr7_Top3ER'].mean()

#     # Calculate standard deviation
#     std_dev = daily_avg_metric.std()

#     avg_market_rank = daily_avg_metric.mean()
#     latest_market_rank = daily_avg_metric.iloc[-1]

#     # Calculate low and high settings
#     low_setting = avg_market_rank - 2 * std_dev
#     high_setting = avg_market_rank + 2 * std_dev

    # return avg_market_rank, std_dev, latest_market_rank, low_setting, high_setting


# # 8.2.24 - new non-parametric using Wilcoxon Sign-rank
# from scipy.stats import wilcoxon
# def hodges_lehmann_estimator(data):
#     n = len(data)
#     pairwise_means = [(data[i] + data[j]) / 2 for i in range(n) for j in range(i, n)]
#     return np.median(pairwise_means)

# def calculate_market_rank_metrics(rankings_df):
#     # Calculate the average TstScr7_Top3ER for each day
#     daily_avg_metric = rankings_df.groupby('Date')['TstScr7_Top3ER'].mean()

#     # Calculate Hodges-Lehmann estimator
#     avg_market_rank = hodges_lehmann_estimator(daily_avg_metric)

#     # Calculate the pseudo-standard deviation using the Wilcoxon signed-rank test
#     _, p_value = wilcoxon(daily_avg_metric - avg_market_rank)
#     pseudo_std = np.sqrt(2) * stats.norm.ppf((1 + p_value) / 2)

#     latest_market_rank = daily_avg_metric.iloc[-1]

#     # Calculate low and high settings using pseudo-standard deviation
#     low_setting = avg_market_rank - 2 * pseudo_std
#     high_setting = avg_market_rank + 2 * pseudo_std

#     return avg_market_rank, pseudo_std, latest_market_rank, low_setting, high_setting


#8.2.24 - non-parametric approach using IQR
# v1
# def calculate_market_rank_metrics(rankings_df):
#     # Calculate the average TstScr7_Top3ER for each day
#     daily_avg_metric = rankings_df.groupby('Date')['TstScr7_Top3ER'].mean()

#     # Calculate non-parametric measures
#     q1 = daily_avg_metric.quantile(0.25)
#     q3 = daily_avg_metric.quantile(0.75)
#     iqr = q3 - q1

#     avg_market_rank = daily_avg_metric.median()  # Use median instead of mean
#     latest_market_rank = daily_avg_metric.iloc[-1]

#     # Calculate low and high settings using IQR
#     low_setting = q1 - 1.5 * iqr
#     high_setting = q3 + 1.5 * iqr

#     return avg_market_rank, iqr, latest_market_rank, low_setting, high_setting


import sqlite3

def init_db():
    conn = sqlite3.connect('strategies.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS best_strategy (
                    id INTEGER PRIMARY KEY,
                    strategy TEXT,
                    total_return REAL,
                    final_value REAL,
                    starting_value REAL,
                    num_transactions INTEGER,
                    current_holdings INTEGER,
                    annualized_return REAL,
                    initial_investment REAL,
                    ranking_metric TEXT,
                    skip_top_n INTEGER,
                    depth INTEGER,
                    start_date TEXT,
                    end_date TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS prior_champion (
                    id INTEGER PRIMARY KEY,
                    strategy TEXT,
                    total_return REAL,
                    final_value REAL,
                    starting_value REAL,
                    num_transactions INTEGER,
                    current_holdings INTEGER,
                    annualized_return REAL,
                    initial_investment REAL,
                    ranking_metric TEXT,
                    skip_top_n INTEGER,
                    depth INTEGER,
                    start_date TEXT,
                    end_date TEXT
                )''')
    conn.commit()
    conn.close()

init_db()




def run_streamlit_app(high_risk_df, low_risk_df, full_start_date, full_end_date):
    # Initialize session state variables
    if 'iteration' not in st.session_state:
        st.session_state.iteration = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'email' not in st.session_state:
        st.session_state.email = ""
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'transactions' not in st.session_state:
        st.session_state.transactions = None
    if 'strategy_results' not in st.session_state:
        st.session_state.strategy_results = None
    if 'strategy_summary_df' not in st.session_state:
        st.session_state.strategy_summary_df = None
    if 'show_image' not in st.session_state:
        st.session_state.show_image = False
    if 'new_wisdom' not in st.session_state:
        st.session_state.new_wisdom = ""
    if 'initial_simulation_run' not in st.session_state:
        st.session_state.initial_simulation_run = False

    # 9.3.24 - new filters/initializations
    # if 'filters' not in st.session_state:
    #     st.session_state.filters = create_fine_tuning_filters(combined_fundamentals_df)
    if 'high_risk_top_x' not in st.session_state:
        st.session_state.high_risk_top_x = 10
    if 'low_risk_top_x' not in st.session_state:
        st.session_state.low_risk_top_x = 10
    if 'high_risk_selected_stocks' not in st.session_state:
        st.session_state.high_risk_selected_stocks = []
    if 'low_risk_selected_stocks' not in st.session_state:
        st.session_state.low_risk_selected_stocks = []
        
    # Initialize new DataFrames for rankings
    # ranking_metric_rankings = pd.DataFrame(columns=['Symbol'])
    # score_original_rankings = pd.DataFrame(columns=['Symbol'])

        
    # CSS for moving ribbons
    st.markdown(
        """
    <style>
    .ticker-wrapper {
        width: 100%;
        overflow: hidden;
        background: black;
        border-bottom: 1px solid #ddd;
        position: relative;
        color: white;
    }
    .ticker {
        display: inline-block;
        white-space: nowrap;
        padding-right: 100%;
        animation-iteration-count: infinite;
        animation-timing-function: linear;
        animation-name: ticker;
    }
    .ticker-1 {
        animation-duration: 1200s;
    }
    .ticker-2 {
        animation-duration: 1500s;
    }
    .ticker-item {
        display: inline-block;
        padding: 0 1rem;
        font-size: 1.2rem;
    }
    @keyframes ticker {
        0% {
            transform: translate3d(0, 0);
        }
        100% {
            transform: translate3d(-100%, 0, 0);
        }
    }
    .top-frame {
        position: relative;
        height: 33vh;
        overflow: hidden;
        width: 100%;
        margin: 0 auto;
    }
    .image-container {
        position: absolute;
        top: 30%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 2;
        width: 9.5vw;
        height: 9.5vw;
        border-radius: 50%;
        overflow: hidden;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .top-frame video {
        position: absolute;
        top: 0%;
        bottom: -30%;
        left: 0;
        width: 100%;
        height: 166.67%;
        object-fit: cover;
        object-position: center center;
        z-index: 1;
    }
    .divider {
        border-top: 3px solid black;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .instructions {
        font-size: 14px;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 20px;
    }
        
    /* Media query for portrait mode on any device */
    @media (orientation: portrait) {
        .top-frame {
            height: 25vh;
        }
        .top-frame video {
            top: -37.5%;
            bottom: -37.5%;
            height: 175%;
            object-position: center center;
        }
        .image-container {
            width: 19vw;
            height: 19vw;
        }
    }
    </style>
        """,
        unsafe_allow_html=True
    )

    # Define wise cracks
    if 'wise_cracks' not in st.session_state:
        st.session_state.wise_cracks = [
        "Buy low, sell high!",
        "Time in the market beats timing the market.",
        "Risk comes from not knowing what you're doing.",
        "Price is what you pay, value is what you get.",
        "The stock market is filled with individuals who know the price of everything, but the value of nothing.",
        "Investing should be more like watching paint dry or watching grass grow. If you want excitement, take $800 and go to Las Vegas.",
        "In investing, what is comfortable is rarely profitable.",
        "The four most dangerous words in investing are: 'This time it's different.'",
        "Know what you own, and know why you own it.",
        "Wide diversification is only required when investors do not understand what they are doing.",
        "The stock market is a device for transferring money from the impatient to the patient.",
        "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price.",
        "Only buy something that you'd be perfectly happy to hold if the market shut down for ten years.",
        "Our favorite holding period is forever.",
        "The most important quality for an investor is temperament, not intellect.",
        "Opportunities come infrequently. When it rains gold, put out the bucket, not the thimble.",
        "The best investment you can make is in yourself.",
        "Never invest in a business you cannot understand.",
        "It's better to hang out with people better than you. Pick out associates whose behavior is better than yours and you'll drift in that direction.",
        "The difference between successful people and really successful people is that really successful people say no to almost everything.",
        "The first rule is not to lose. The second rule is not to forget the first rule.",
        "Someone's sitting in the shade today because someone planted a tree a long time ago.",
        "Predicting rain doesn't count, building the ark does.",
        "Chains of habit are too light to be felt until they are too heavy to be broken.",
        "I always knew I was going to be rich. I don't think I ever doubted it for a minute.",
        "If you aren't willing to own a stock for ten years, don't even think about owning it for ten minutes.",
        "The best chance to deploy capital is when things are going down.",
        "You only have to do a very few things right in your life so long as you don't do too many things wrong.",
        "The business schools reward difficult complex behavior more than simple behavior, but simple behavior is more effective.",
        "If past history was all there was to the game, the richest people would be librarians.",
        "You know... you keep doing the same things and you keep getting the same result over and over again.",
        "The best thing that happens to us is when a great company gets into temporary trouble... We want to buy them when they're on the operating table.",
        "We simply attempt to be fearful when others are greedy and to be greedy only when others are fearful.",
        "Time is the friend of the wonderful company, the enemy of the mediocre.",
        "Wall Street is the only place that people ride to in a Rolls Royce to get advice from those who take the subway.",
        "You can't produce a baby in one month by getting nine women pregnant.",
        "It's better to have a partial interest in the Hope diamond than to own all of a rhinestone.",
        "Beware the investment activity that produces applause; the great moves are usually greeted by yawns.",
        "I will tell you how to become rich. Close the doors. Be fearful when others are greedy. Be greedy when others are fearful.",
        "The investor of today does not profit from yesterday's growth.",
        "Do not save what is left after spending, but spend what is left after saving.",
        "The individual investor should act consistently as an investor and not as a speculator.",
        "An investment in knowledge pays the best interest.",
        "I never attempt to make money on the stock market. I buy on the assumption that they could close the market the next day and not reopen it for five years.",
        "The intelligent investor is a realist who sells to optimists and buys from pessimists.",
        "The function of economic forecasting is to make astrology look respectable.",
        "I'm only rich because I know when I'm wrong... I basically have survived by recognizing my mistakes.",
        "If you have trouble imagining a 20% loss in the stock market, you shouldn't be in stocks.",
        "Every once in a while, the market does something so stupid it takes your breath away.",
        "The stock market is a device for transferring money from the Active to the Patient."
        # Additional Warren Buffett quotes
        "Rule No. 1: Never lose money. Rule No. 2: Never forget Rule No. 1.",
        "The most important investment you can make is in yourself.",
        "It takes 20 years to build a reputation and five minutes to ruin it. If you think about that, you'll do things differently.",
        "Be fearful when others are greedy and greedy when others are fearful.",
        
        # Elon Musk quotes
        "When something is important enough, you do it even if the odds are not in your favor.",
        "I think it's very important to have a feedback loop, where you're constantly thinking about what you've done and how you could be doing it better.",
        "Failure is an option here. If things are not failing, you are not innovating enough.",
        "The first step is to establish that something is possible; then probability will occur.",
        "If you get up in the morning and think the future is going to be better, it is a bright day. Otherwise, it's not.",
        
        # Mark Cuban quotes
        "It doesn't matter how many times you fail. You only have to be right once and then everyone can tell you that you are an overnight success.",
        "Sweat equity is the most valuable equity there is. Know your business and industry better than anyone else in the world.",
        "Work like there is someone working 24 hours a day to take it all away from you.",
        
        # Gary Vaynerchuk quotes
        "Stop whining, start hustling.",
        "Patience is the key to success in business and in life.",
        "Your personal brand is your resume. And your resume is no longer a piece of paper.",
        
        # Oprah Winfrey quotes
        "The biggest adventure you can take is to live the life of your dreams.",
        "You become what you believe, not what you think or what you want.",
        "The more you praise and celebrate your life, the more there is in life to celebrate.",
        
        # Steve Jobs quotes
        "Your work is going to fill a large part of your life, and the only way to be truly satisfied is to do what you believe is great work.",
        "Innovation distinguishes between a leader and a follower.",
        "Stay hungry, stay foolish.",
        
        # Michelle Obama quotes
        "Success isn't about how much money you make. It's about the difference you make in people's lives.",
        "There is no limit to what we, as women, can accomplish.",
        "When they go low, we go high.",
        
        # Jeff Bezos quotes
        "I knew that if I failed I wouldn't regret that, but I knew the one thing I might regret is not trying.",
        "If you double the number of experiments you do per year you're going to double your inventiveness.",
        "The common question that gets asked in business is, 'why?' That's a good question, but an equally valid question is, 'why not?'",
        "The best way to predict the future is to create it.",
        "Your time is limited, don't waste it living someone else's life." 
        "The only place where success comes before work is in the dictionary.",
        "Don't watch the clock; do what it does. Keep going." ,
        "The greatest glory in living lies not in never falling, but in rising every time we fall.",
        "The way to get started is to quit talking and begin doing.",
        "If you really look closely, most overnight successes took a long time." ,
        "Twenty years from now you will be more disappointed by the things that you didn't do than by the ones you did do." ,
        "The future belongs to those who believe in the beauty of their dreams.",
        "Don't be afraid to give up the good to go for the great." ,
        "I find that the harder I work, the more luck I seem to have.",
        "Success is not final, failure is not fatal: it is the courage to continue that counts." ,
        "The only limit to our realization of tomorrow will be our doubts of today." ,
        "Believe you can and you're halfway there.",
        "I have not failed. I've just found 10,000 ways that won't work." ,
        "The secret of getting ahead is getting started." ,
        "Don't cry because it's over, smile because it happened." ,
        "Life is what happens to you while you're busy making other plans." ,
        "The mind is everything. What you think you become." ,
        "The best revenge is massive success." ,
        "Strive not to be a success, but rather to be of value.",
        "The most difficult thing is the decision to act, the rest is merely tenacity." ,
        "Every strike brings me closer to the next home run." ,
        "The two most important days in your life are the day you are born and the day you find out why." ,
        "There is only one way to avoid criticism: do nothing, say nothing, and be nothing." ,
        "Ask and it will be given to you; search, and you will find; knock and the door will be opened for you." ,
        "We can easily forgive a child who is afraid of the dark; the real tragedy of life is when men are afraid of the light." ,
        "Everything you've ever wanted is on the other side of fear." ,
        "Start where you are. Use what you have. Do what you can." ,
        "When one door of happiness closes, another opens, but often we look so long at the closed door that we do not see the one that has been opened for us."
    ]
# 7.29.24 - moved over here from down below by IMPORTANT
    st.title("Interactive Strategy Evaluation Engine powered by Zoltar Ranks")
    

    # HTML for moving ribbons
    st.markdown(
        f"""
        <div class="ticker-wrapper">
            <div class="ticker ticker-1">
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in st.session_state.wise_cracks])}
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in st.session_state.wise_cracks])}
            </div>
        </div>
        <div class="ticker-wrapper">
            <div class="ticker ticker-2">
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in st.session_state.wise_cracks[::-1]])}
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in st.session_state.wise_cracks[::-1]])}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Top frame with image and video background
    st.markdown(
        """
        <div class="top-frame">
            <video autoplay loop muted>
                <source src="https://github.com/apod-1/ZoltarFinancial/raw/main/docs/wave_vid.mp4" type="video/mp4">
            </video>
            <div class="image-container">
                <img src="https://github.com/apod-1/ZoltarFinancial/raw/main/docs/ZoltarSurf2.png" alt="Zoltar Image">
            </div>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True
    )
    
    # New section to enable users to enter their own wise cracks
    # st.subheader("Share Your Wisdom")  # Add a subheader to create space
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_wisdom = st.text_input("Add your own wisdom!", key="new_wisdom_input", value=st.session_state.get('new_wisdom', ''))
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add a line break for spacing
        if st.button("Submit", key='new_wisdom_input2'):
            if new_wisdom:
                st.session_state.wise_cracks.append(new_wisdom)
                st.session_state.new_wisdom = ""  # Clear the stored new wisdom
                st.rerun()  # Rerun the app to reflect changes

    # st.write("IMPORTANT: For best experience please use in landscape mode on high-memory device (optimization under way to address lackluster mobile experience). Thank you for your patience!")
    
    # Calculate the overall date range
    min_date = min(high_risk_df['Date'].min(), low_risk_df['Date'].min())
    max_date = max(high_risk_df['Date'].max(), low_risk_df['Date'].max())
    
    st.write("Date range:", min_date.strftime('%m-%d-%Y'), "to", max_date.strftime('%m-%d-%Y'))
    
    # Calculate the total number of unique symbols across both dataframes
    unique_symbols = set(high_risk_df['Symbol'].unique()) | set(low_risk_df['Symbol'].unique())
    st.write("Number of unique symbols:", len(unique_symbols))
    
    
    # Instructions section
    st.subheader("Instructions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="instructions">
            <strong>Date Range Selection:</strong><br>
            1,200 pre-filtered Symbols based on liquidity, market cap and analyst rank (refreshed infrequently)<br>
            - Use Pre-selected buttons: Select from data used for Training Ranks, Validation, or Out-of-Time Validation Ranges<br>
            <br>
            Narrow down selected ranges further with more precise selection if needed<br>
            - Start Date: Select the start date for analysis<br>
            - End Date: Select the end date for analysis<br>
            <br>
            <br>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="instructions">
            <strong>Rank Selection:</strong><br>
            Risk Controls: Select HIgh Return or Low Risk<br>
            - Fine-Tuning: Choose to use Sharpe ratio for rank (Shape-ify), Sector round-robin (Bullet-proof)(all are driven by Zoltar Score Suite) <br>
            - Enable Alternate Execution: use ML-driven triage of model to use based on low Market Gauge Trigger<br>
            - Enable Sell and Hold: Option available for Alternate Execution mode to panic sell X stocks with lowest Zoltar Rank (Fine-Tuning Slider)<br>
            Rank Use Criteria: Number of top ranked stocks in each purchase (Select top X, Omit first Y), or use Hard-coded Score Criteria<br>
            - Portfolio Fine-tuning: Filter based on specific Market Cap, Sector, and Industry preferences<br>
            <strong>Sell Criteria:</strong><br>
            - Use sliders to adjust stop-loss and annualized target gain thresholds<br>
            </div>
            """,
            unsafe_allow_html=True
        )
            # ATTENTION: Users are currently experiencing lackluster navigation experience, may take 2 clicks to change settings<br>
    st.write('ATTENTION: Users are currently experiencing lackluster navigation experience, may take 2 clicks to change settings')





    def calculate_market_rank_metrics(high_risk_dft, low_risk_dft, risk_level, use_sharpe, sectors=None, industries=None, market_cap="All"):
        # Select the appropriate dataframe based on risk level
        df = high_risk_dft if risk_level == 'High' else low_risk_dft
        
        # Filter by sectors and industries if specified
        if sectors:
            df = df[df['Sector'].isin(sectors)]
        if industries:
            df = df[df['Industry'].isin(industries)]
        
        # Filter by market cap
        if market_cap != "All":
            df = df[df['Cap_Size'] == market_cap]
            
        # Get the last date in the dataframe
        last_date = df['Date'].max()
        
        # Calculate the date 15 days before the last date
        start_date = last_date - timedelta(days=15)
        
        # Filter the dataframe to keep only the last 15 days of data
        df = df[df['Date'] > start_date]
        
        ranking_metric = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
        
        # Calculate daily average of the ranking metric
        daily_avg_metric = df.groupby('Date')[ranking_metric].mean()
        print("Daily Average Metric:")
        print(daily_avg_metric)
        
        # Sort the daily average metrics
        sorted_metrics = daily_avg_metric.sort_values(ascending=False)
        print("Sorted Metrics:")
        print(sorted_metrics)
        
        # Calculate the mean of the top 20 values after omitting the top 2
        if len(sorted_metrics) > 22:
            avg_market_rank = sorted_metrics.iloc[2:22].mean()
        else:
            avg_market_rank = sorted_metrics.mean()  # Fallback if there are not enough values
        
        print(f"Average Market Rank: {avg_market_rank}")
        
        latest_market_rank = daily_avg_metric.iloc[-1]
        print(f"Latest Market Rank: {latest_market_rank}")
        
        # Calculate standard deviation
        std_dev = sorted_metrics.iloc[2:22].std() if len(sorted_metrics) > 22 else sorted_metrics.std()
        print(f"Standard Deviation: {std_dev}")
        
        # Calculate low and high settings
        low_setting = avg_market_rank - 2 * std_dev
        high_setting = avg_market_rank + 2 * std_dev
        print(f"Low Setting: {low_setting}, High Setting: {high_setting}")
    
        return avg_market_rank, std_dev, latest_market_rank, low_setting, high_setting
    
    def generate_daily_rankings_strategies(selected_df, select_portfolio_func, start_date=None, end_date=None,
                                           initial_investment=10000,
                                           strategy_3_annualized_gain=0.3, strategy_3_loss_threshold=-0.07,
                                           omit_first=2, top_x=15, ranking_metric='High_Risk_Score',
                                           use_sharpe=False, use_bullet_proof=False,
                                           market_cap="All", sectors=None, industries=None,
                                           risk_level='High', show_industries=False, score_cutoff=None,
                                           enable_alternate_execution=False, gauge_trigger=None,
                                           high_risk_df=None, low_risk_df=None,
                                           enable_panic_sell=False):
        if start_date is None:
            start_date = selected_df['Date'].min()
        if end_date is None:
            end_date = selected_df['Date'].max()
    
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_date, end=end_date)
    
        # Initialize SPY data
        spy_data = selected_df[selected_df['Symbol'] == 'SPY'].copy()
        spy_data['Return'] = spy_data['Close_Price'].pct_change()
        spy_data = spy_data.set_index('Date')
    
        # Create a Series of SPY returns for the entire date range
        spy_returns = spy_data['Return'].reindex(date_range).fillna(0)
    
        if spy_returns.empty:
            print("Error: No SPY data found in selected_df")
            return None, None, None, None, None
    
        # Initialize rankings DataFrame
        rankings = pd.DataFrame(columns=['Date', 'Symbol', ranking_metric])
    
        # Initialize strategy tracking
        strategy_results = {
            'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment, 'Date': []}
        }
    
        # Calculate total number of days
        total_days = len(date_range)

  
        # Create a progress bar and progress text
        # progress_bar = st.progress(0)
        # progress_text = st.empty()

        
        # Initialize top_ranked_symbols_last_day
        top_ranked_symbols_last_day = []
        
        # Create these outside your loop, before starting the iterations
        progress_bar_placeholder = st.empty()
        progress_text_placeholder = st.empty()
        completion_message_placeholder = st.empty()
    
        for i, current_date in enumerate(date_range):
            normalized_rank = None  # Initialize normalized_rank
            
            # Update progress
            progress = (i + 1) / total_days
            
            # # Update progress bar
            # html_progress = f"""
            # <div style="width:100%; background-color:#ddd; border-radius:5px;">
            #     <div style="width:{progress*100}%; height:20px; background-color:#4CAF50; border-radius:5px;">
            #     </div>
            # </div>
            # """
            # progress_bar_placeholder.markdown(html_progress, unsafe_allow_html=True)
            
            if progress < 1:
                # Update progress bar
                html_progress = f"""
                <div style="width:100%; background-color:#ddd; border-radius:5px;">
                    <div style="width:{progress*100}%; height:20px; background-color: #663399; border-radius:5px;">
                    </div>
                </div>
                """
                # other options: background-color: #DDA0DD;
                # Lavender: #E6E6FA
                # xml
                # background-color: #E6E6FA;
                
                # Medium Purple: #9370DB
                # xml
                # background-color: #9370DB;
                
                # Rebecca Purple: #663399
                # xml
                # background-color: #663399;
                
                # Slate Blue: #6A5ACD
                # xml
                # background-color: #6A5ACD;
                
                # Dark Orchid: #9932CC
                # xml
                # background-color: #9932CC;
                
                # Plum: #DDA0DD
                # xml
                # background-color: #DDA0DD;
                
                # Indigo: #4B0082
                # xml
                # background-color: #4B0082;
                
                # Violet: #EE82EE
                # xml
                # background-color: #EE82EE;
                progress_bar_placeholder.markdown(html_progress, unsafe_allow_html=True)
                
                # Update progress text
                progress_text_placeholder.text(f"Progress: {progress:.2%}")
            else:
                # Remove progress bar and text
                progress_bar_placeholder.empty()
                progress_text_placeholder.empty()
                
                # Show completion celebration
                st.balloons()  # or st.snow()
                
                # Optionally, you can add a completion message
                completion_message_placeholder.success("Simulation completed successfully!")                # Wait for 2 seconds
                time.sleep(0.7)
                
                # Remove the success message
                completion_message_placeholder.empty()

                
            # Update temporary dataframes with data up to the current date
            temp_high_risk_df = high_risk_df[high_risk_df['Date'] <= current_date].copy()
            temp_low_risk_df = low_risk_df[low_risk_df['Date'] <= current_date].copy()
        
            # Merge high and low risk data for the current date
            current_high_risk = temp_high_risk_df[temp_high_risk_df['Date'] == current_date]
            current_low_risk = temp_low_risk_df[temp_low_risk_df['Date'] == current_date]
            current_data = pd.merge(current_high_risk, current_low_risk, on=['Date', 'Symbol', 'Close_Price', 'Cap_Size', 'Sector', 'Industry'], suffixes=('_high', '_low'))
        
            # Rename columns to standard names
            current_data = current_data.rename(columns={
                'High_Risk_Score_high': 'High_Risk_Score',
                'High_Risk_Score_Sharpe_high': 'High_Risk_Score_Sharpe',
                'Low_Risk_Score_low': 'Low_Risk_Score',
                'Low_Risk_Score_Sharpe_low': 'Low_Risk_Score_Sharpe'
            })
        
            if current_data.empty:
                print(f"No data available for date: {current_date}")
                continue
        
            print(f"Processing date: {current_date}")

             # Set default ranking metric
            # Set default ranking metric
            default_ranking_metric = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
            ranking_metric = default_ranking_metric
       
            # Alternate Execution Logic
            if enable_alternate_execution and gauge_trigger is not None:
                avg_market_rank, std_dev, latest_market_rank, low_setting, high_setting = calculate_market_rank_metrics(
                    temp_high_risk_df, temp_low_risk_df, risk_level, use_sharpe, sectors, industries, market_cap
                )
                
                try:
                    if high_setting == low_setting:
                        print("Warning: high_setting equals low_setting. Setting normalized_rank to 50.")
                        normalized_rank = 50
                    else:
                        normalized_rank = (latest_market_rank - low_setting) / (high_setting - low_setting) * 100
                        normalized_rank = max(0, min(100, normalized_rank))  # Ensure it's within 0-100
                    print(f"Calculated normalized_rank: {normalized_rank}")
                except Exception as e:
                    print(f"Error calculating normalized_rank: {str(e)}")
                    print(f"latest_market_rank: {latest_market_rank}")
                    print(f"low_setting: {low_setting}")
                    print(f"high_setting: {high_setting}")
                    normalized_rank = 50  # Default to middle value if calculation fails
  
            
                risk_level = st.session_state.get('risk_level', 'High')
                use_sharpe =st.session_state.get('use_sharpe', False)
                if normalized_rank < gauge_trigger:
                    print(f"Current market gauge ({normalized_rank:.2f}) is below the trigger ({gauge_trigger}). Searching for alternate execution.")
                    
                    scenarios = [
                        ('Low', False),
                        ('Low', True),
                        ('High', False),
                        ('High', True)
                    ]
                    
                    market_gauges = {}
                    for scenario_risk_level, scenario_use_sharpe in scenarios:
                        avg_market_rank, std_dev, scenario_latest_market_rank, low_setting, high_setting = calculate_market_rank_metrics(
                            temp_high_risk_df, temp_low_risk_df, scenario_risk_level, scenario_use_sharpe, sectors, industries, market_cap
                        )
                        
                        try:
                            if high_setting == low_setting:
                                print(f"Warning: high_setting equals low_setting for {scenario_risk_level} {'Sharpe' if scenario_use_sharpe else 'standard'}. Setting normalized_rank to 50.")
                                scenario_normalized_rank = 50
                            else:
                                scenario_normalized_rank = (scenario_latest_market_rank - low_setting) / (high_setting - low_setting) * 100
                                scenario_normalized_rank = max(0, min(100, scenario_normalized_rank))  # Ensure it's within 0-100
                            print(f"Calculated normalized_rank for {scenario_risk_level} {'Sharpe' if scenario_use_sharpe else 'standard'}: {scenario_normalized_rank}")
                        except Exception as e:
                            print(f"Error calculating normalized_rank for {scenario_risk_level} {'Sharpe' if scenario_use_sharpe else 'standard'}: {str(e)}")
                            scenario_normalized_rank = 50  # Default to middle value if calculation fails
            
                        market_gauges[(scenario_risk_level, scenario_use_sharpe)] = {
                            'avg_market_rank': avg_market_rank,
                            'std_dev': std_dev,
                            'latest_market_rank': scenario_latest_market_rank,
                            'normalized_rank': scenario_normalized_rank,
                            'low_setting': low_setting,
                            'high_setting': high_setting
                        }
                    
                    selected_scenario = None
                    for (scenario_risk_level, scenario_use_sharpe), gauge in market_gauges.items():
                        if gauge['normalized_rank'] >= gauge_trigger:
                            selected_scenario = (scenario_risk_level, scenario_use_sharpe)
                            break

                    if selected_scenario:
                        risk_level, use_sharpe = selected_scenario
                        print(f"Using alternate execution: {risk_level} risk with {'Sharpe' if use_sharpe else 'standard'} scoring")
                        ranking_metric = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
                    else:
                        print("No suitable alternate scenario found. Using default execution settings.")
                        ranking_metric = default_ranking_metric

                    
                                        # Panic Sell Logic
                    # if enable_panic_sell and gauge_trigger is not None and normalized_rank < gauge_trigger:
                    #     top_x = 0
                    #     omit_first = 10  # Omit all stocks
                    #     score_cutoff = 10000  # Set an impossibly high score cutoff
                    #     strategy_3_loss_threshold = 0.05  # Sell all holdings regardless of loss
                        
                        # ranking_metric = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
                        # risk_level, use_sharpe = selected_scenario
                    
            # Apply portfolio selection
            if score_cutoff is not None:
                portfolio = select_portfolio_func(
                    current_data[current_data[ranking_metric] >= score_cutoff],
                    None,  # top_x is not used for score cut-off method
                    omit_first,
                    use_sharpe,
                    market_cap,
                    sectors,
                    industries,
                    risk_level,
                    show_industries,
                    use_bullet_proof
                )
            else:
                portfolio = select_portfolio_func(
                    current_data,
                    top_x,
                    omit_first,
                    use_sharpe,
                    market_cap,
                    sectors,
                    industries,
                    risk_level,
                    show_industries,
                    use_bullet_proof
                )

                        
            # Ensure all four ranking metrics are in the portfolio DataFrame
            ranking_metrics = ['High_Risk_Score', 'High_Risk_Score_Sharpe', 'Low_Risk_Score', 'Low_Risk_Score_Sharpe']
            for metric in ranking_metrics:
                if metric not in portfolio.columns:
                    portfolio[metric] = np.nan
            
            # Calculate rankings for the day
            # Calculate rankings for the day
            daily_rankings = []
            for _, stock in current_data.iterrows():  # Use current_data instead of portfolio
                symbol = stock['Symbol']
                ranking_data = {
                    'Symbol': symbol,
                    'High_Risk_Score': stock.get('High_Risk_Score', np.nan),
                    'High_Risk_Score_Sharpe': stock.get('High_Risk_Score_Sharpe', np.nan),
                    'Low_Risk_Score': stock.get('Low_Risk_Score', np.nan),
                    'Low_Risk_Score_Sharpe': stock.get('Low_Risk_Score_Sharpe', np.nan),
                    'Close_Price': stock['Close_Price'],
                    'Cap_Size': stock['Cap_Size'],
                    'Sector': stock['Sector'],
                    'Industry': stock['Industry']
                }
                daily_rankings.append(ranking_data)
            
            # Sort and update rankings
            daily_rankings_df = pd.DataFrame(daily_rankings)
            daily_rankings_df['Date'] = current_date
            
            # Ensure the ranking_metric column exists
            if ranking_metric not in daily_rankings_df.columns:
                print(f"Warning: {ranking_metric} not found in daily_rankings_df. Using 'Close_Price' for sorting.")
                ranking_metric = 'Close_Price'
            
            if ranking_metric in daily_rankings_df.columns:
                daily_rankings_df = daily_rankings_df.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
            else:
                print(f"Error: {ranking_metric} still not found in daily_rankings_df. Cannot sort.")
                print("Available columns:", daily_rankings_df.columns)
            
            rankings = pd.concat([rankings, daily_rankings_df], ignore_index=True)    
            # daily_rankings_df = daily_rankings_df.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
            # rankings = pd.concat([rankings, daily_rankings_df], ignore_index=True)
    
            # Update strategy results
            update_strategy(strategy_results['Strategy_3'], portfolio, current_data, current_date,
                            strategy_3_annualized_gain, strategy_3_loss_threshold,
                            ranking_metric, top_x, omit_first, score_cutoff, enable_panic_sell,
                            normalized_rank, gauge_trigger, bottom_z_percent)
    
            # Store top ranked symbols for the last day
            if i == total_days - 1:
                top_ranked_symbols_last_day = daily_rankings_df['Symbol'].tolist()[:top_x]
        
        # Remove progress bar and text after completion
        # progress_bar.empty()
        # progress_text.empty()
        
        # Calculate final strategy value
        strategy_values = {'Strategy_3': strategy_results['Strategy_3']['Daily_Value'][-1] if strategy_results['Strategy_3']['Daily_Value'] else initial_investment}
        
        # Prepare summary statistics
        summary = {
            'Strategy_3': {
                'Starting Value': initial_investment,
                'Final Value': strategy_values['Strategy_3'],
                'Total Return': (strategy_values['Strategy_3'] - initial_investment) / initial_investment,
                'Number of Transactions': len(strategy_results['Strategy_3']['Transactions']),
                'Average Days Held': np.mean([t['Days_Held'] for t in strategy_results['Strategy_3']['Transactions'] if 'Days_Held' in t]) if strategy_results['Strategy_3']['Transactions'] else 0
            }
        }
        
        return rankings, strategy_results, strategy_values, summary, top_ranked_symbols_last_day




    # st.write("Available secret keys:", list(st.secrets.keys()))
    # if "GMAIL" in st.secrets:
    #     st.write("GMAIL secret keys:", list(st.secrets["GMAIL"].keys()))    
    # Add the chatbot section
    
    st.subheader("Zoltar Chat Assistant | Knowledge is your friend")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask Zoltar a question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Set your OpenAI API key from secrets
        try:
            openai.api_key = st.secrets["openai"]["api_key"]
        except KeyError:
            st.error("OpenAI API key not found in secrets. Please clear cache and reboot app.")
            st.stop()        
        # openai.api_key = st.secrets["openai"]["api_key"]
        # openai.api_key = st.secrets["openai"]["api_key"]
    
        # Send the prompt to the ChatGPT API and get a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a stock trading application named Zoltar that prepares responses as a short summary followed by more details in table format for most requests."},
                {"role": "user", "content": prompt}
            ]
        )
    
        # Extract the response text
        response_text = response.choices[0].message['content']
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_text)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})




    # Sidebar user selections
    st.sidebar.header("Strategy Parameters")
    
    # Extract date ranges for validate, validate_oot, and train
    validate_dates = high_risk_df[high_risk_df['source'] == 'validate']['Date'].dropna()
    validate_oot_dates = high_risk_df[high_risk_df['source'] == 'validate_oot']['Date'].dropna()
    train_dates = high_risk_df[high_risk_df['source'] == 'train']['Date'].dropna()
    
    # Initialize session state for selected option if not exists
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = "Validate"

    
    # Custom CSS for button styling
    st.markdown("""
    <style>
        div.stButton > button {
            width: 100%;
            height: auto;
            padding: 5px 2px;
            border: none;
            font-size: 10px;
            font-weight: bold;
            white-space: normal;
            line-height: 1.2;
        }
        div.stButton > button:first-child {
            border-radius: 5px 0 0 5px;
        }
        div.stButton > button:last-child {
            border-radius: 0 5px 5px 0;
        }
        div.stButton > button:hover {
            filter: brightness(90%);
        }
        .all-button button {
            background-color: #1E90FF;
            color: white;
        }
        .train-button button {
            background-color: #FFA500;
            color: black;
        }
        .validate-button button {
            background-color: #4CAF50;
            color: white;
        }
        .oot-button button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a single row with all buttons
    col1, col2, col3, col4 = st.sidebar.columns(4)
    
    with col1:
        st.markdown('<div class="all-button">', unsafe_allow_html=True)
        if st.button("ALL", key="all", help="Select all date ranges"):
            st.session_state.selected_option = "All"
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="train-button">', unsafe_allow_html=True)
        if st.button("TRAIN MODELS", key="train", help="Select training date range"):
            st.session_state.selected_option = "Train"
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="validate-button">', unsafe_allow_html=True)
        if st.button("TEST STRATEGY", key="validate", help="Select validation date range"):
            st.session_state.selected_option = "Validate"
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="oot-button">', unsafe_allow_html=True)
        if st.button("VAL STRATEGY", key="validate_oot", help="Select out-of-time validation date range"):
            st.session_state.selected_option = "Validate OOT"
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Set default start and end dates based on selection
    if st.session_state.selected_option == "All":
        start_date = high_risk_df['Date'].min()
        end_date = high_risk_df['Date'].max()
    elif st.session_state.selected_option == "Train":
        start_date = train_dates.min()
        end_date = train_dates.max()
    elif st.session_state.selected_option == "Validate":
        start_date = validate_dates.min()
        end_date = validate_dates.max()
    elif st.session_state.selected_option == "Validate OOT":
        start_date = validate_oot_dates.min()
        end_date = validate_oot_dates.max()

    # Allow user to adjust start and end dates
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", start_date)
    end_date = col2.date_input("End Date", end_date)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)    
    
    # Ensure start_date and end_date are valid dates
    # start_date = pd.Timestamp.now().date() if pd.isnull(start_date) else start_date.date()
    # end_date = pd.Timestamp.now().date() if pd.isnull(end_date) else end_date.date()
    
    # Allow user to adjust start and end dates
    # col1, col2 = st.sidebar.columns(2)
    # start_date = col1.date_input("Start Date", start_date)
    # end_date = col2.date_input("End Date", end_date)
    
    # start_date = pd.to_datetime(start_date)
    # end_date = pd.to_datetime(end_date)
    
    # Filter the dataframe based on the selected date range
    # selected_df = high_risk_df[(high_risk_df['Date'] >= start_date) & (high_risk_df['Date'] <= end_date)]

    # Score selection
    # Score selection
    centered_header("Rank Selection")
    col1, col2 = st.sidebar.columns(2)
    with col2:
        st.subheader("Fine-tuning")
        use_sharpe = st.checkbox("Sharpe-ify")
        use_bullet_proof = st.checkbox("Bullet-proof")
        # Add the "Enable Alternate Execution" checkbox
        enable_alternate_execution = st.checkbox("Enable Alternate Execution")
        # 9.11.24 additions..

    with col1:
        st.subheader("Risk Controls")
        risk_level = st.radio(
            label="Risk Level",
            options=["High", "Low"],
            label_visibility="collapsed"
        )
        selected_df = high_risk_df if risk_level == "High" else low_risk_df
        if enable_alternate_execution:
            st.write("Alternate Execution")
            gauge_trigger = st.number_input("Low Market Gauge Trigger", min_value=0, max_value=100, value=15)
            enable_panic_sell = st.checkbox("Enable Sell and Hold")
        else:
            enable_panic_sell = False  # Set a default value when alternate execution is not enabled
        
    with col2:
       if enable_panic_sell:
            bottom_z_percent = st.slider("Bottom Z% for Sell Trigger", min_value=0, max_value=100, value=20, step=1)
       else:
            bottom_z_percent = 0  # Set a default value when not enabled


        # Display the "Low Market Gauge Trigger" input if alternate execution is enabled
        # if enable_alternate_execution:
        #     gauge_trigger = st.number_input("Low Market Gauge Trigger", min_value=0, max_value=100, value=25)
    

    
    # Score selection criteria
    centered_header("Rank Use Criteria")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        portfolio_selection_method = st.radio("Selection Method", ["Top X", "Hard-coded Score"])
    
    with col2:
        if portfolio_selection_method == "Top X":
            top_x = st.number_input("Select top X stocks", min_value=1, max_value=100, value=1)
            omit_first = st.number_input("Omit first Y stocks", min_value=0, max_value=100, value=0)
            score_cutoff = 0.01  # Default value, not used in this method
        else:
            score_cutoff = st.number_input("Enter score cut-off", min_value=0.0, max_value=5.0, value=0.005, step=0.005)
            top_x = 15  # Default value, not used in this method
            omit_first = 0  # Default value, not used in this method


    
    # Display the selected values
    # st.sidebar.write(f"Selected method: {portfolio_selection_method}")
    if portfolio_selection_method == "Top X":
        st.sidebar.write(f"Selecting top {top_x} stocks, skipping first {omit_first} stocks")
    else:
        st.sidebar.write(f"Using hard-coded score cut-off of {score_cutoff}")
    
    # Fine Tuning Section
    centered_header("Portfolio Fine-Tuning")
    market_cap = st.sidebar.selectbox("Market Cap", ["All", "Small", "Mid", "Large"])
    sectors = st.sidebar.multiselect("Sectors", selected_df['Sector'].unique())
    show_industries = st.sidebar.checkbox("Show Industries")
    if show_industries:
        industries = st.sidebar.multiselect("Industries", selected_df['Industry'].unique())
    else:
        industries = None
    
    # Buy and Sell criteria
    centered_header("Sell Criteria")
    strategy_3_annualized_gain = st.sidebar.number_input("Annualized Gain", min_value=0.0, max_value=1.0, value=0.27, step=0.005)
    strategy_3_loss_threshold = st.sidebar.number_input("Loss Threshold", min_value=-1.0, max_value=0.0, value=-0.01, step=0.005)
    
    # Initial investment
    initial_investment = st.sidebar.number_input("Initial Investment", min_value=1000, max_value=1000000, value=10000, step=1000)


    #9.3.24 - moved gauge code here In your main Streamlit app:
    # In the market gauge section, pass the user-selected sectors and industries
    if True:  # st.button("Generate Market Gauge"):
        # Calculate Market Rank Metrics
        avg_market_rank, std_dev, latest_market_rank, low_setting, high_setting = calculate_market_rank_metrics(
            high_risk_df, low_risk_df, risk_level, use_sharpe, sectors=sectors, industries=industries, market_cap=market_cap
        )
        
        # Normalize the latest market rank to a 0-100 scale
        try:
            if high_setting == low_setting:
                print("Warning: high_setting equals low_setting. Setting normalized_rank to 50.")
                normalized_rank = 50
            else:
                normalized_rank = (latest_market_rank - low_setting) / (high_setting - low_setting) * 100
                normalized_rank = max(0, min(100, normalized_rank))  # Ensure it's within 0-100
            print(f"Calculated normalized_rank: {normalized_rank}")
        except Exception as e:
            print(f"Error calculating normalized_rank: {str(e)}")
            print(f"latest_market_rank: {latest_market_rank}")
            print(f"low_setting: {low_setting}")
            print(f"high_setting: {high_setting}")
            normalized_rank = 50  # Default to middle value if calculation fails
        
        # Get the maximum date from both dataframes
        max_date = max(high_risk_df['Date'].max(), low_risk_df['Date'].max())
        
        # Calculate the next business day
        next_bd = (max_date + BDay(1)).strftime('%m-%d-%Y')
        # Display the Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=normalized_rank,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
            },
            title={'text': f"Market Gauge for {next_bd} using {risk_level} Risk Rank {'w/ Sharpe' if use_sharpe else ''}"}
        ))
        
        st.plotly_chart(fig)

        # Add a horizontal double-line before the section
        st.markdown("<hr style='height:4px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

   

    # Function to select portfolio based on sector logic
    # def select_portfolio_with_sectors(df, top_x, omit_first, use_sharpe, market_cap, sectors, industries):
    #     score_column = f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}"
        
    #     # Filter based on market cap, sectors, and industries
    #     if market_cap != "All":
    #         df = df[df['Cap_Size'] == market_cap]
    #     if sectors:
    #         df = df[df['Sector'].isin(sectors)]
    #     if show_industries and industries:
    #         df = df[df['Industry'].isin(industries)]
        
    #     # Sort and select top stocks
    #     df_sorted = df.sort_values(score_column, ascending=False)
    #     top_stocks = df_sorted.iloc[omit_first:omit_first+top_x]
        
    #     # Implement sector-based selection logic
    #     if use_bullet_proof:
    #         selected_stocks = []
    #         selected_sectors = set()
    #         for _, stock in top_stocks.iterrows():
    #             if len(selected_stocks) >= top_x:
    #                 break
    #             if stock['Sector'] not in selected_sectors or len(selected_stocks) < len(sectors):
    #                 selected_stocks.append(stock)
    #                 selected_sectors.add(stock['Sector'])
    #         return pd.DataFrame(selected_stocks)
    #     else:
    #         return top_stocks
        
    # Use the function to select portfolio
    if st.sidebar.button("Generate Portfolio"):
        # Select the appropriate dataframe based on risk level
        selected_df = high_risk_df if risk_level == 'High' else low_risk_df
    
        # Initialize variables
        # gauge_trigger = None
        latest_market_rank = None
        selected_scenario = None
        # gauge_trigger = st.session_state.get('gauge_trigger', 15)  # Default to 25 if not set
    
        rankings, strategy_results, strategy_values, summary, top_ranked_symbols_last_day = generate_daily_rankings_strategies(
            selected_df,
            select_portfolio_with_sectors,
            start_date=start_date,
            end_date=end_date,
            initial_investment=10000,
            strategy_3_annualized_gain=strategy_3_annualized_gain,
            strategy_3_loss_threshold=strategy_3_loss_threshold,
            omit_first=omit_first if omit_first is not None else 0,
            top_x=top_x if top_x is not None else None,
            ranking_metric=f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}",
            use_sharpe=use_sharpe,
            use_bullet_proof=use_bullet_proof,
            market_cap=market_cap,
            sectors=sectors,
            industries=industries if show_industries else None,
            risk_level=risk_level,
            enable_alternate_execution=enable_alternate_execution,
            gauge_trigger=gauge_trigger if enable_alternate_execution else None,
            high_risk_df=high_risk_df,
            low_risk_df=low_risk_df,
            enable_panic_sell=enable_panic_sell
        )
    
        st.subheader("Strategy Summary")
        strategy_summary_df = pd.DataFrame(summary['Strategy_3'], index=[0])
        st.dataframe(strategy_summary_df.style.format({
            'Starting Value': "${:.2f}",
            'Final Value': "${:.2f}",
            'Total Return': "{:.2%}",
            'Average Days Held': "{:.1f}"
        }))
    
        # Display strategy performance chart
        st.subheader("Strategy Performance")
        strategy_df = pd.DataFrame({
            'Date': strategy_results['Strategy_3']['Date'],
            'Strategy_3': strategy_results['Strategy_3']['Daily_Value']
        })
        
        # Add SPY performance for comparison
        spy_data = selected_df[selected_df['Symbol'] == 'SPY'].copy()
        spy_data['Return'] = spy_data['Close_Price'].pct_change()
        spy_data = spy_data.set_index('Date')
        spy_returns = spy_data['Return'].reindex(strategy_df['Date']).fillna(0)
        spy_values = [10000]  # Assuming same initial investment
        for ret in spy_returns:
            spy_values.append(spy_values[-1] * (1 + ret))
        strategy_df['SPY'] = spy_values[1:]
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strategy_df['Date'], y=strategy_df['Strategy_3'], mode='lines', name='Strategy 3'))
        fig.add_trace(go.Scatter(x=strategy_df['Date'], y=strategy_df['SPY'], mode='lines', name='SPY'))
        fig.update_layout(title='Strategy vs SPY Performance', xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig)
    
        # Display transactions
        st.subheader("Transactions")
        transactions_df = pd.DataFrame(strategy_results['Strategy_3']['Transactions'])
        if not transactions_df.empty:
            st.dataframe(transactions_df)
    
        # Display current holdings
        st.subheader("Current Holdings")
        holdings = strategy_results['Strategy_3']['Book']
        holdings_df = pd.DataFrame(holdings, columns=['Symbol'])
        if not holdings_df.empty:
            st.dataframe(holdings_df)
    
        # Store results in session state
        st.session_state.strategy_results = strategy_results
        st.session_state.summary = summary
        st.session_state.rankings = rankings
        
        # Update best strategy
        st.session_state.best_strategy = {
            'Strategy': f"{risk_level} Risk Score {'w/ Sharpe' if use_sharpe else ''} {' w/ BulletProof' if use_bullet_proof else ''}",
            **summary['Strategy_3'],
            'Settings': {
                'Risk Level': risk_level,
                'Ranking Metric': f"{risk_level}_Risk_Score{'_Sharpe' if use_sharpe else ''}",
                'Use Sharpe': use_sharpe,
                'Use Bullet Proof': use_bullet_proof,
                'Skip Top N': omit_first,
                'Depth': top_x,
                'Start Date': start_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d'),
                'Strategy Parameters': {
                    'Annualized Gain': strategy_3_annualized_gain,
                    'Loss Threshold': strategy_3_loss_threshold
                },
                'Market Cap': market_cap,
                'Sectors': sectors,
                'Industries': industries if show_industries else None,
                'Initial Investment': initial_investment,
                'Enable Alternate Execution': enable_alternate_execution,
                'Low Market Gauge Trigger': gauge_trigger if enable_alternate_execution else None
            },
            'Top_Ranked_Symbols': top_ranked_symbols_last_day
        }
        
        # Display top-ranked symbols for the last day
        st.write("Top Ranked Symbols for the Last Day:")
        st.write(top_ranked_symbols_last_day)
        
        # Record settings and summary
        if 'iteration' not in st.session_state:
            st.session_state.iteration = 0
        st.session_state.iteration += 1
        
        history_entry = {
            'Iteration': st.session_state.iteration,
            'Settings': st.session_state.best_strategy['Settings'],
            'Summary': summary['Strategy_3']
        }
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(history_entry)
    
        # After generating rankings, store them in session state
        st.session_state.high_risk_rankings = convert_to_ranking_format(high_risk_df, f"High_Risk_Score{'_Sharpe' if use_sharpe else ''}")
        st.session_state.low_risk_rankings = convert_to_ranking_format(low_risk_df, f"Low_Risk_Score{'_Sharpe' if use_sharpe else ''}")
    
        # Display alternate execution information if enabled
        if enable_alternate_execution:
            st.subheader("Alternate Execution Information")
            st.write(f"Low Market Gauge Trigger: {gauge_trigger}")
            if 'used_alternate_execution' in summary['Strategy_3']:
                st.write(f"Used Alternate Execution: {'Yes' if summary['Strategy_3']['used_alternate_execution'] else 'No'}")
            if 'alternate_execution_details' in summary['Strategy_3']:
                st.write("Alternate Execution Details:")
                st.write(summary['Strategy_3']['alternate_execution_details'])
        
        # #  # Convert high and low risk dataframes to the required format
        # # high_risk_rankings = convert_to_ranking_format(high_risk_df, f"High_Risk_Score{'_Sharpe' if use_sharpe else ''}")
        # # low_risk_rankings = convert_to_ranking_format(low_risk_df, f"Low_Risk_Score{'_Sharpe' if use_sharpe else ''}")
        
        # # # Store in session state for use in display_interactive_rankings
        # st.session_state.high_risk_rankings = high_risk_rankings
        # st.session_state.low_risk_rankings = low_risk_rankings
        
        # # Use the rankings in the display_interactive_rankings function
        # # st.subheader("Latest Iteration Ranks Research")
        
        # if 'high_risk_rankings' in st.session_state and 'low_risk_rankings' in st.session_state:
        #     col1, col2 = st.columns(2)
            
        #     with col1:
        #         st.subheader("High Risk Rankings")
        #         display_interactive_rankings(st.session_state.high_risk_rankings, f"High_Risk_Score{'_Sharpe' if use_sharpe else ''}")
            
        #     with col2:
        #         st.subheader("Low Risk Rankings")
        #         display_interactive_rankings(st.session_state.low_risk_rankings, f"Low_Risk_Score{'_Sharpe' if use_sharpe else ''}")
        # else:
        #     st.write("Rankings data not available. Please run a simulation first.")
        
        
        
          
        # Display Best Strategy
        st.subheader("Best Strategy in the Iteration")
        best_strategy = st.session_state.best_strategy
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Strategy", best_strategy['Strategy'])
            st.metric("Total Return", f"{best_strategy['Total Return']:.2%}")
            st.metric("Final Value", f"${best_strategy['Final Value']:.2f}")
        with col2:
            st.metric("Initial Investment", f"${best_strategy['Starting Value']:.2f}")
            st.metric("Number of Transactions", best_strategy['Number of Transactions'])
            st.metric("Average Days Held", f"{best_strategy['Average Days Held']:.1f}")
        
        # Add table with strategy settings
        st.subheader("Best Strategy Settings")
        settings_data = {
            "Setting": [
                "Initial Investment", "Ranking Metric", "Skip Top N", "Depth", 
                "Start Date", "End Date", "Annualized Gain", "Loss Threshold",
                "Use Sharpe", "Use Bullet Proof", "Market Cap", "Sectors", "Industries", "Risk Level"
            ],
            "Value": [
                f"${best_strategy['Settings']['Initial Investment']:.2f}",
                best_strategy['Settings']['Ranking Metric'],
                best_strategy['Settings']['Skip Top N'],
                best_strategy['Settings']['Depth'],
                best_strategy['Settings']['Start Date'],
                best_strategy['Settings']['End Date'],
                f"{best_strategy['Settings']['Strategy Parameters']['Annualized Gain']:.2%}",
                f"{best_strategy['Settings']['Strategy Parameters']['Loss Threshold']:.2%}",
                str(best_strategy['Settings']['Use Sharpe']),
                str(best_strategy['Settings']['Use Bullet Proof']),
                best_strategy['Settings']['Market Cap'],
                ', '.join(best_strategy['Settings']['Sectors']) if best_strategy['Settings']['Sectors'] else 'None',
                ', '.join(best_strategy['Settings']['Industries']) if best_strategy['Settings']['Industries'] else 'None',
                best_strategy['Settings']['Risk Level']
            ]
        }
        
        settings_df = pd.DataFrame(settings_data)
        st.dataframe(settings_df)
        
        # Display top-ranked symbols
        st.write("Top Ranked Symbols:")
        st.write(best_strategy['Top_Ranked_Symbols'])
        
        # Display persistent results
        if 'strategy_results' in st.session_state and st.session_state.strategy_results is not None:
            st.subheader("Strategy Performance")
            
            strategy_data = st.session_state.strategy_results['Strategy_3']
            
            # Ensure that Date and Daily_Value have the same length
            min_length = min(len(strategy_data['Date']), len(strategy_data['Daily_Value']))
            
            strategy_df = pd.DataFrame({
                'Date': pd.to_datetime(strategy_data['Date'][:min_length]),
                'Value': strategy_data['Daily_Value'][:min_length]
            })
            
            # Sort the DataFrame by date to ensure chronological order
            strategy_df = strategy_df.sort_values('Date').reset_index(drop=True)
            
            # Create the chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategy_df['Date'], y=strategy_df['Value'], mode='lines', name='Strategy 3'))
            fig.update_layout(title='Strategy 3 Performance', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig)
            
            st.subheader("Strategy Values")
            st.dataframe(strategy_df.style.format({'Value': "${:.2f}"}))
            
            st.subheader("Strategy Summary")
            if 'summary' in st.session_state:
                summary_df = pd.DataFrame(st.session_state.summary['Strategy_3'], index=[0])
                st.dataframe(summary_df.style.format({
                    'Starting Value': "${:.2f}",
                    'Final Value': "${:.2f}",
                    'Total Return': "{:.2%}",
                    'Number of Transactions': "{:,.0f}",
                    'Average Days Held': "{:.1f}"
                }))
            
            st.subheader("Transactions")
            transactions_df = pd.DataFrame(st.session_state.strategy_results['Strategy_3']['Transactions'])
            if not transactions_df.empty:
                st.dataframe(transactions_df)
            
                
        # Display Interactive Strategy Training History
        st.header("Strategy Training History")
        if st.session_state.history:
            for entry in st.session_state.history:
                st.subheader(f"Iteration {entry['Iteration']}")
                st.json(entry['Settings'])
                summary_df = pd.DataFrame(entry['Summary'], index=[0])
                st.dataframe(summary_df.style.format({
                    'Starting Value': "${:.2f}",
                    'Final Value': "${:.2f}",
                    'Total Return': "{:.2%}",
                    'Number of Transactions': "{:.0f}",
                    'Average Days Held': "{:.1f}"
                }))
                st.markdown("---")
        else:
            st.write("No iterations have been run yet. Use the 'Generate Portfolio' button to start.")
            
    # Add a horizontal double-line before the section
    # st.markdown("<hr style='height:4px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)


               # other options: background-color: #DDA0DD;
               # Lavender: #E6E6FA
               # xml
               # background-color: #E6E6FA;
               
               # Medium Purple: #9370DB
               # xml
               # background-color: #9370DB;
               
               # Rebecca Purple: #663399
               # xml
               # background-color: #663399;
               
               # Slate Blue: #6A5ACD
               # xml
               # background-color: #6A5ACD;
               
               # Dark Orchid: #9932CC
               # xml
               # background-color: #9932CC;
               
               # Plum: #DDA0DD
               # xml
               # background-color: #DDA0DD;
               
               # Indigo: #4B0082
               # xml
               # background-color: #4B0082;
               
               # Violet: #EE82EE
               # xml
               # background-color: #EE82EE;



    # 9.3.24 -  Place this after the "Generate Portfolio" button callback
    centered_header_main("Zoltar Ranks Research")
    

    # Create fine-tuning filters
    if 'filters' not in st.session_state:
        st.session_state.filters = create_fine_tuning_filters(combined_fundamentals_df)
    
    # Display fine-tuning parameters in two columns with padding
    filters,line, col1, padding, col2 = st.columns([5,1,10, 1, 10])
    
    with filters:
        
        # Add a horizontal double-line before the section
        # st.markdown("<hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)
        #7851A9
        # st.markdown("""
        # <div style="
        #     background-color: #663399; 
        #     border-radius: 10px;
        #     padding: 10px;
        #     text-align: center;
        #     margin: 10px 0;
        # ">
        #     <span style="
        #         color: white;
        #         font-weight: bold;
        #         font-size: 18px;
        #     ">Settings</span>
        # </div>
        # """, unsafe_allow_html=True)  
        centered_header_main("Settings")

        centered_header_main_small("Fundamentals [1+1=2]")
        
        # Analyst Rating
        st.session_state.filters = list(st.session_state.filters)  # Convert tuple to list for modification
        min_rating = round(float(combined_fundamentals_df['Fundamentals_OverallRating'].min()) * 2) / 2
        max_rating = round(float(combined_fundamentals_df['Fundamentals_OverallRating'].max()) * 2) / 2
        # Round the min and max values to one decimal place
        min_rating = np.round(min_rating, 1)
        max_rating = np.round(max_rating, 1)
        
        # Round the current values to one decimal place
        if isinstance(st.session_state.filters[0], tuple):
            current_min_rating, current_max_rating = st.session_state.filters[0]
            current_min_rating = np.round(current_min_rating, 1)
            current_max_rating = np.round(current_max_rating, 1)
        else:
            current_min_rating, current_max_rating = min_rating, max_rating
        
        # Create the slider with values rounded to one decimal place
        st.session_state.filters[0] = st.slider(
            "Analyst Rating", 
            min_value=float(min_rating),
            max_value=float(max_rating),
            value=(float(max(min_rating, current_min_rating)), 
                   float(min(max_rating, current_max_rating))),
            step=0.1,
            format="%.1f",  # This forces the display to show only one decimal place
            key="analyst_rating_slider"
        )
        
        # Ensure the selected values are also rounded to one decimal place
        min_rating, max_rating = st.session_state.filters[0]
        min_rating = np.round(min_rating, 1)
        max_rating = np.round(max_rating, 1)
        st.session_state.filters[0] = (min_rating, max_rating)
        
        # # Market Cap
        # market_cap_billions = combined_fundamentals_df['Fundamentals_MarketCap'] / 1e9
        # min_cap = round(float(market_cap_billions.min()) * 2) / 2
        # max_cap = round(float(market_cap_billions.max()) * 2) / 2
        # st.session_state.filters[3] = st.slider(
        #     "Market Cap (Bn)", 
        #     min_value=min_cap,
        #     max_value=max_cap,
        #     value=(min_cap, max_cap),  # Default to full range
        #     step=0.5,
        #     key="market_cap_slider"
        # )

        # # Market Cap
        # market_cap_billions = combined_fundamentals_df['Fundamentals_MarketCap'] / 1e9
        # true_min_cap = float(market_cap_billions.min())
        # true_max_cap = float(market_cap_billions.max())
        
        # # Set display range for slider
        # display_min_cap = round(true_min_cap * 2) / 2
        # display_max_cap = 1000.0  # Cap at 1T (1000 billion)
        
        # # Initialize or get current values
        # if isinstance(st.session_state.filters[3], tuple):
        #     current_min_cap, current_max_cap = st.session_state.filters[3]
        # else:
        #     current_min_cap, current_max_cap = display_min_cap, min(display_max_cap, true_max_cap)
        
        # # Create the slider
        # selected_min_cap, selected_max_cap = st.slider(
        #     "Market Cap (Bn)", 
        #     min_value=display_min_cap,
        #     max_value=display_max_cap,
        #     value=(max(display_min_cap, min(current_min_cap, display_max_cap)), 
        #            min(display_max_cap, max(current_max_cap, display_min_cap))),
        #     step=0.5,
        #     key="market_cap_slider"
        # )
        
        # # Adjust the filter values to include out-of-range records when max is selected
        # filter_min_cap = selected_min_cap
        # filter_max_cap = true_max_cap if selected_max_cap == display_max_cap else selected_max_cap
        
        # # Update the session state
        # st.session_state.filters[3] = (filter_min_cap, filter_max_cap)
        
        # # Display the actual filter range being applied
        # if filter_max_cap > display_max_cap:
        #     st.write(f"Actual Market Cap filter range: ${filter_min_cap:.1f}B to ${filter_max_cap:.1f}B (includes all above {display_max_cap}B)")
        # else:
        #     st.write(f"Actual Market Cap filter range: ${filter_min_cap:.1f}B to ${filter_max_cap:.1f}B")
        
        # Market Cap
        market_cap_billions = combined_fundamentals_df['Fundamentals_MarketCap'] / 1e9
        true_min_cap = float(market_cap_billions.min())
        true_max_cap = float(market_cap_billions.max())
        
        # Set fixed display range for slider
        display_min_cap = 0.0  # Assuming no negative market caps
        display_max_cap = 1000.0  # Cap at 1T (1000 billion)
        
        # Initialize or get current values
        if isinstance(st.session_state.filters[3], tuple):
            current_min_cap, current_max_cap = st.session_state.filters[3]
        else:
            current_min_cap, current_max_cap = display_min_cap, display_max_cap
        
        # Round the display values to one decimal place
        display_min_cap = np.round(display_min_cap, 1)
        display_max_cap = np.round(display_max_cap, 1)
        
        # Round the current values to one decimal place
        current_min_cap = np.round(current_min_cap, 1)
        current_max_cap = np.round(current_max_cap, 1)
        
        # Create the slider with values rounded to one decimal place
        st.session_state.filters[3] = st.slider(
            "Market Cap (Bn)", 
            min_value=float(display_min_cap),
            max_value=float(display_max_cap),
            value=(float(max(display_min_cap, min(current_min_cap, display_max_cap))), 
                   float(min(display_max_cap, max(current_max_cap, display_min_cap)))),
            step=0.1,
            format="%.1f",  # This forces the display to show only one decimal place
            key="market_cap_slider_unique"  # Unique key to avoid conflicts
        )
        
        # Ensure the selected values are also rounded to one decimal place
        min_cap, max_cap = st.session_state.filters[3]
        min_cap = np.round(min_cap, 1)
        max_cap = np.round(max_cap, 1)
        st.session_state.filters[3] = (min_cap, max_cap)
        
        # Adjust the filter values to include out-of-range records
        min_cap, max_cap = st.session_state.filters[3]
        if min_cap == display_min_cap:
            min_cap = true_min_cap
        if max_cap == display_max_cap:
            max_cap = true_max_cap
        
        # Update the session state with adjusted values
        st.session_state.filters[3] = (min_cap, max_cap)
        
        # Display the actual filter range being applied
        # st.write(f"Actual Market Cap range: ${min_cap:.1f} B to ${max_cap/1000:.1f} T")
        if max_cap >= 1000:
            st.markdown(f"<p style='font-size: 10px;'>Note: Actual Market Cap range: ${min_cap:.1f}B to ${max_cap/1000:.1f}T</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-size: 10px;'>Note: Actual Market Cap range: ${min_cap:.1f}B to ${max_cap:.1f}B</p>", unsafe_allow_html=True)
        # PE Ratio
        pe_ratios = combined_fundamentals_df['Fundamentals_PE'].dropna()
        true_min_pe = float(pe_ratios.min())
        true_max_pe = float(pe_ratios.max())
        
        # Set fixed display range for slider
        display_min_pe = -10.0
        display_max_pe = 100.0
        
        # Initialize or get current values
        if isinstance(st.session_state.filters[2], tuple):
            current_min_pe, current_max_pe = st.session_state.filters[2]
        else:
            current_min_pe, current_max_pe = display_min_pe, display_max_pe
        
        # Create the slider
        # Round the display values to one decimal place
        display_min_pe = np.round(display_min_pe, 1)
        display_max_pe = np.round(display_max_pe, 1)
        
        # Round the current values to one decimal place
        current_min_pe = np.round(current_min_pe, 1)
        current_max_pe = np.round(current_max_pe, 1)
        
        # Create the slider with values rounded to one decimal place
        selected_min_pe, selected_max_pe = st.slider(
            "PE Ratio", 
            min_value=float(display_min_pe),
            max_value=float(display_max_pe),
            value=(float(max(display_min_pe, min(current_min_pe, display_max_pe))), 
                   float(min(display_max_pe, max(current_max_pe, display_min_pe)))),
            step=0.1,
            format="%.1f",  # This forces the display to show only one decimal place
            key="pe_ratio_slider"
        )
        
        # Ensure the selected values are also rounded to one decimal place
        selected_min_pe = np.round(selected_min_pe, 1)
        selected_max_pe = np.round(selected_max_pe, 1)
        
        # Adjust the filter values to include out-of-range records
        filter_min_pe = true_min_pe if selected_min_pe == display_min_pe else selected_min_pe
        filter_max_pe = true_max_pe if selected_max_pe == display_max_pe else selected_max_pe
        
        # Update the session state
        st.session_state.filters[2] = (filter_min_pe, filter_max_pe)
        
        # Display the actual filter range being applied
        st.markdown(f"<p style='font-size: 10px;'>Note: Actual PE Ratio range: {filter_min_pe:.0f} to {filter_max_pe:.0f}</p>", unsafe_allow_html=True)
        
    # Empty padding column
    padding.write("")
    
    with filters:

        centered_header_main_small("Dividends ($$$)")
        
        # Dividend Yield
        min_yield = float(0) #round(float(combined_fundamentals_df['Fundamentals_Dividends'].min()) * 2) / 2
        max_yield = round(float(combined_fundamentals_df['Fundamentals_Dividends'].max()) * 2) / 2
        # Round the min and max values to one decimal place
        min_yield = np.round(min_yield, 1)
        max_yield = np.round(max_yield, 1)
        
        # Round the current values to one decimal place
        if isinstance(st.session_state.filters[1], tuple):
            current_min_yield, current_max_yield = st.session_state.filters[1]
            current_min_yield = np.round(current_min_yield, 1)
            current_max_yield = np.round(current_max_yield, 1)
        else:
            current_min_yield, current_max_yield = min_yield, max_yield
        
        # Create the slider with values rounded to one decimal place
        st.session_state.filters[1] = st.slider(
            "Dividend Yield (%)", 
            min_value=float(min_yield),
            max_value=float(max_yield),
            value=(float(max(min_yield, current_min_yield)), 
                   float(min(max_yield, current_max_yield))),
            step=0.1,
            format="%.1f",  # This forces the display to show only one decimal place
            key="dividend_yield_slider"
        )
        
        # Ensure the selected values are also rounded to one decimal place
        min_yield, max_yield = st.session_state.filters[1]
        min_yield = np.round(min_yield, 1)
        max_yield = np.round(max_yield, 1)
        st.session_state.filters[1] = (min_yield, max_yield)
        
        # Ex-Dividend
        ex_dividend_options = ["All", "Within 2 days", "Within 1 week", "Within 1 month"]
        st.session_state.filters[4] = st.radio(
            "Dividend", 
            ex_dividend_options, 
            index=ex_dividend_options.index(st.session_state.filters[4]), 
            key="ex_dividend_radio"
        )
    
    st.session_state.filters = tuple(st.session_state.filters)  # Convert back to tuple
    
    # # Add a horizontal double-line before the section
    # 9.14.24 - REMOVED
    # st.markdown("<hr style='height:4px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

    with line:
        # Use a loop to create multiple small elements that form a line
        for _ in range(35):  # Adjust the range to control the line's height
            # st.markdown(
            #     """
            #     <div style="
            #         background-color: #808080;
            #         width: 5px;
            #         height: 10px;
            #         margin: 5px auto;
            #     "></div>
            #     """,
            #     unsafe_allow_html=True
            # )

            st.markdown(
                """
                <div style="
                    background-color: #808080;
                    width: 5px;
                    height: 1.9vh;
                    margin: 0 auto;
                "></div>
                """,
                unsafe_allow_html=True
            )


    # Initialize session state for persistent values
    if 'high_risk_top_x' not in st.session_state:
        st.session_state.high_risk_top_x = 10
    if 'low_risk_top_x' not in st.session_state:
        st.session_state.low_risk_top_x = 10
    
    # Main content area with two columns
    # 9.14.24 - REMOVED
    # col1, col2 = st.columns(2)


    
    with col1:

        st.markdown("""
        <div style="
            background-color: #663399;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            margin: 10px 0;
        ">
            <span style="
                color: white;
                font-weight: bold;
                font-size: 18px;
            ">High Risk Rankings</span>
        </div>
        """, unsafe_allow_html=True)  


        
        # centered_header_main("High Risk Rankings")
        if 'high_risk_rankings' in st.session_state:
            st.session_state.high_risk_top_x = st.slider(
                "Number of top stocks to display (High Risk)", 
                min_value=1, max_value=50, value=st.session_state.high_risk_top_x, step=1, 
                key="high_risk_top_x_slider"
            )
            display_interactive_rankings(
                st.session_state.high_risk_rankings, 
                f"High_Risk_Score{'_Sharpe' if use_sharpe else ''}", 
                combined_fundamentals_df, 
                st.session_state.filters, 
                st.session_state.high_risk_top_x,
                date_range=(start_date, end_date),
                unique_prefix="high_risk"  # Add this line
            )

        else:
            st.write("High Risk rankings data not available. Please generate a portfolio first.")
        if 'High_Risk_filtered_df' in st.session_state:
            st.dataframe(st.session_state['High_Risk_filtered_df'].head(st.session_state.high_risk_top_x))
    
    with col2:
        st.markdown("""
        <div style="
            background-color: #663399;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            margin: 10px 0;
        ">
            <span style="
                color: white;
                font-weight: bold;
                font-size: 18px;
            ">Low Risk Rankings</span>
        </div>
        """, unsafe_allow_html=True)  
        # centered_header_main("Low Risk Rankings")
        if 'low_risk_rankings' in st.session_state:
            st.session_state.low_risk_top_x = st.slider(
                "Number of top stocks to display (Low Risk)", 
                min_value=1, max_value=50, value=st.session_state.low_risk_top_x, step=1, 
                key="low_risk_top_x_slider"
            )
            display_interactive_rankings(
                st.session_state.low_risk_rankings, 
                f"Low_Risk_Score{'_Sharpe' if use_sharpe else ''}", 
                combined_fundamentals_df, 
                st.session_state.filters, 
                st.session_state.low_risk_top_x,
                date_range=(start_date, end_date),
                unique_prefix="low_risk"  # Add this line
            )
        else:
            st.write("Low Risk rankings data not available. Please generate a portfolio first.")
        if 'Low_Risk_filtered_df' in st.session_state:
            st.dataframe(st.session_state['Low_Risk_filtered_df'].head(st.session_state.low_risk_top_x))

#     # Filter based on user selection
#     display_df = filtered_df[filtered_df['Symbol'].isin(selected_stocks)]
   
#         # Display the dataframe
#         st.dataframe(display_df)






    
    # # Display persistent results
    # st.subheader("Persistent Rankings")
    
    # if 'High_Risk_filtered_df' in st.session_state:
    #     st.subheader("High Risk Rankings")
    #     st.dataframe(st.session_state['High_Risk_filtered_df'].head(st.session_state.high_risk_top_x))
    #     # if 'High_Risk_plot' in st.session_state:
    #     #     st.plotly_chart(st.session_state['High_Risk_plot'])
    
    # if 'Low_Risk_filtered_df' in st.session_state:
    #     st.subheader("Low Risk Rankings")
    #     st.dataframe(st.session_state['Low_Risk_filtered_df'].head(st.session_state.low_risk_top_x))
    #     # if 'Low_Risk_plot' in st.session_state:
    #     #     st.plotly_chart(st.session_state['Low_Risk_plot'])
    
        # Display persistent results
        # st.subheader("Persistent Rankings")
        
        # # High Risk Rankings
        # if 'High_Risk_filtered_df' in st.session_state:
        #     st.subheader("Persistent High Risk Rankings")
        #     st.dataframe(st.session_state['High_Risk_filtered_df'].head(st.session_state.high_risk_top_x))
        #     if 'High_Risk_plot' in st.session_state:
        #         st.plotly_chart(st.session_state['High_Risk_plot'])
        #     else:
        #         st.write("High Risk plot not available.")
    
        # # Low Risk Rankings
        # if 'Low_Risk_filtered_df' in st.session_state:
        #     st.subheader("Persistent Low Risk Rankings")
        #     st.dataframe(st.session_state['Low_Risk_filtered_df'].head(st.session_state.low_risk_top_x))
        #     if 'Low_Risk_plot' in st.session_state:
        #         st.plotly_chart(st.session_state['Low_Risk_plot'])
        #     else:
        #         st.write("Low Risk plot not available.")

        
        
    # Clear Results button
    if st.sidebar.button("Clear Simulation", key="clear_simulation_button"):
        # Logic to clear history and best strategy
        st.session_state.history = []
        st.session_state.best_strategy = {}
        st.success("Simulation cleared.")


    # 8.3.24 - email yourself    
    # Sidebar input for email
    # st.sidebar.markdown("---")  # Add a separator
    # user_email = st.sidebar.text_input("Email your best strategy results:")
    # if st.sidebar.button("Send Email"):
    #     if user_email:
    #         if 'top_ranked_symbols_last_day' in st.session_state or 'best_strategy' in st.session_state:
    #             send_user_email(user_email)
    #         else:
    #             st.sidebar.error("Strategy data not available. Please run a simulation first.")
    #     else:
    #         st.sidebar.error("Please enter a valid email address.")            

    # 7.25.24 - adding email list and Main menu
    # Email list sign-up section
    st.sidebar.markdown("---")
    st.sidebar.header("Subscribe to Our Newsletter (under construction)")
    
    # Use a unique key for the text input
    email_key = f"email_input_{st.session_state.iteration}"
    email = st.sidebar.text_input("Enter your email:", key=email_key, value=st.session_state.email)
    # Store the email in session state
    st.session_state.email = email

    if st.sidebar.button("Subscribe", key=f"subscribe_button_{st.session_state.iteration}"):
        if email:
            try:
                if add_email_to_list(email):
                    st.sidebar.success("Thank you for subscribing!")
                else:
                    st.sidebar.info("You're already subscribed!")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")
                print(f"Error details: {e}")
        else:
            st.sidebar.error("Please enter a valid email address.")
 
    # Initialize a session state variable for the Pi click
    if 'pi_clicked' not in st.session_state:
        st.session_state.pi_clicked = False
    # Use a container to hold the button that will be hidden
    # button_container = st.empty()
    
    # Interactive menu section on the right pane
    menu_options = ["About", "Our Mission", "Methodology", "ZF Blockchain", "Investors"]
    selected_option = st.sidebar.selectbox("Menu", menu_options)

    if selected_option == "About":
        st.header("About Zoltar Financial")
        
        # Display the image
        image_path = "https://github.com/apod-1/ZoltarFinancial/raw/main/docs/AboutZoltar.png"
        st.image(image_path, caption="Zoltar Financial 2024", use_column_width=True)
        
        st.write("Zoltar Financial is a quant-based research firm focused on stock market ranking, custom strategy selection and building a community around our ZF blockchain project")
    elif selected_option == "Our Mission":
        st.header("Our Mission")
        st.write("We surgically designed a set of features and a segmentation that with the help of a suite of a Machine Learning/Time-Series/Optimization routines that systemically train, test, validate solutions to derive Zoltar ranks, design strategies and deploy through brokerage buy/sell actions.  We are happy to release the 'behind-the-scenes' on the methodology and the research, with potential to go even further in evolution of Financial products with the help of those eager to:")
        st.write("  1) Use the trading research platform to A/B test strategies in s structured environment with a well-defined research design")
        st.write("  2) Design and backtest strategies, with buylists of the day ")
        st.write("  3) Learn about sector and industry trends, and broader model parameter estimate changes that lead to overall market swings ")
        st.write("  4) Participate and rival in broader leaderboard of strategies found on the platform (that are also accessible to everyone)")
        st.write("  5) Be part of the community to create and launch Zoltar Financial blockchain (ZF token)")

    elif selected_option == "Methodology":
        st.header("Methodology")
        st.write("1. Target definition")
        st.write("2. Sector and Industry level modeling and Feature engineering")
        st.write("3. Segmentation")
        st.write("4. Transparent, Repeatable Binning and other Transformations")
        st.write("5. A suite of Machine Learning algorithms")
        st.write("6. Optimization and tuning of portofolio using a suite of models with varying levels of Zoltar Users' risk tolerance criteria")
        st.write("7. Strategy training and validation is available for Zoltar Users to customize, share, and compete")
        st.write("8. Leader strategy is run live daily, trading on Zoltar Corp to showcase Zoltar community strength and marking the start of ZF blockchain")



    elif selected_option == "ZF Blockchain":
        st.header("ZF Blockchain")
        st.write("Explore our blockchain solutions for secure and transparent financial transactions, community-guided algorithm and a decentralized profit sharing smart contract...")

    elif selected_option == "Investors":
        st.header("Investor Relations")
        st.write("Information for current and potential investors...coming soon")

    # # Register the callback function
    # query_params = st.query_params()
    # if 'print_email_list' in query_params:
    #     print_email_list()
    
        
    # Add a button to the bottom right corner
    st.markdown(
        """
        <style>
        .pi-button-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
        }
        .pi-button-container .stButton > button {
            font-size: 24px !important;
            padding: 5px 10px !important;
            line-height: 1 !important;
            background-color: transparent !important;
            border: none !important;
            color: blue !important;
            text-align: right !important;
            width: auto !important;
            min-width: 40px !important;
            display: flex !important;
            justify-content: flex-end !important;
        }
        </style>
        <div class="pi-button-container">
        """,
        unsafe_allow_html=True
    )

    
    st.button("π", key="show_image_button", on_click=toggle_show_image)
    
    st.markdown("</div>", unsafe_allow_html=True)
   
    
    # Get the maximum date from both dataframes
    max_date = max(high_risk_df['Date'].max(), low_risk_df['Date'].max())
    
    # Calculate the next business day
    next_bd = (max_date + BDay(1)).strftime('%m-%d-%Y')

    # Display image when button is clicked
    if st.session_state.show_image:
        # Title of the Section
        st.markdown(f"<h2 style='text-align: center;'>Recommendations for {next_bd}</h2>", unsafe_allow_html=True)
    
        # Generate rankings_df for the last 3 days
        end_date = max_date
        rankings_df = generate_last_week_rankings(
            high_risk_df=high_risk_df,
            low_risk_df=low_risk_df,
            end_date=end_date,
            risk_level=risk_level,
            use_sharpe=use_sharpe
        )
        # Display the rankings
        st.write("Last Week Rankings:")
        st.dataframe(rankings_df)
    

    # 9.3.24 - removed gauge code from here to above
    
        # Row 1: Recommendations
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("<h3 style='text-align: center;'>Small Cap </h3>", unsafe_allow_html=True)
            small_rec = get_latest_file("expected_returns_path_Small_")
            if small_rec:
                st.image(small_rec)
            else:
                st.write("Small Cap Recommendations image not found")
    
        with col2:
            st.markdown("<h3 style='text-align: center;'>Mid Cap </h3>", unsafe_allow_html=True)
            mid_rec = get_latest_file("expected_returns_path_Mid_")
            if mid_rec:
                st.image(mid_rec)
            else:
                st.write("Mid Cap Recommendations image not found")
    
        with col3:
            st.markdown("<h3 style='text-align: center;'>Large Cap </h3>", unsafe_allow_html=True)
            large_rec = get_latest_file("expected_returns_path_Large_")
            if large_rec:
                st.image(large_rec)
            else:
                st.write("Large Cap Recommendations image not found")
    
        # Row 2: Performance
        col1, col2, col3 = st.columns(3)
    
        with col1:
            small_perf = get_latest_file("selected_stocks_performance_Small_")
            if small_perf:
                st.image(small_perf)
            else:
                st.write("Small Cap Performance image not found")
    
        with col2:
            mid_perf = get_latest_file("selected_stocks_performance_Mid_")
            if mid_perf:
                st.image(mid_perf)
            else:
                st.write("Mid Cap Performance image not found")
    
        with col3:
            large_perf = get_latest_file("selected_stocks_performance_Large_")
            if large_perf:
                st.image(large_perf)
            else:
                st.write("Large Cap Performance image not found")
    
        # New Section: Overall Zoltar Stock Picks
        st.markdown(f"<h2 style='text-align: center;'>Overall Zoltar Stock Picks - {next_bd}</h2>", unsafe_allow_html=True)
    
        # Display images in a single column
        all_rec_1 = get_latest_file("expected_returns_path_ALL_")
        all_rec_2 = get_latest_file("selected_stocks_performance_ALL_")
        
        if all_rec_1:
            st.markdown(f"<div style='text-align: center;'><img src='{all_rec_1}' style='max-width: 100%; height: auto;'></div>", unsafe_allow_html=True)
        else:
            st.write("Overall Recommendations image not found")
        
        if all_rec_2:
            st.markdown(f"<div style='text-align: center;'><img src='{all_rec_2}' style='max-width: 100%; height: auto;'></div>", unsafe_allow_html=True)
        else:
            st.write("Overall Performance image not found")
    
    # Add this block here, just before the if __name__ == "__main__": block
    if st.session_state.get('componentValue'):
        st.session_state.show_image = True
        st.session_state.componentValue = False

    # To make it persistent, add this outside of any button callbacks:

    # Display persistent results
    # st.subheader("Persistent Rankings")
    # if 'High_Risk_filtered_df' in st.session_state:
    #     st.subheader("Persistent High Risk Rankings")
    #     st.dataframe(st.session_state['High_Risk_filtered_df'].head(10))

    # if 'Low_Risk_filtered_df' in st.session_state:
    #     st.subheader("Persistent Low Risk Rankings")
    #     st.dataframe(st.session_state['Low_Risk_filtered_df'].head(10))

if __name__ == "__main__":
    # Initialize session state for button visibility
    if 'show_confirmation' not in st.session_state:
        st.session_state.show_confirmation = False
        st.session_state.start_time = 0
    
    # Initialize session state for general use
    if 'session_state' not in st.session_state:
        st.session_state['session_state'] = {}
    
    # Function to hide confirmation after 2 seconds
    def hide_confirmation():
        if time.time() - st.session_state.start_time > 2:
            st.session_state.show_confirmation = False
    
    # Get the latest files
    if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
        # Cloud environment
        data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
    else:
        # Local environment
        data_dir = '/mount/src/zoltarfinancial/daily_ranks'    
    # data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
    latest_files = get_latest_files(data_dir)
    
    # Load the data
    # @st.cache_data
    def load_data(file_path):
        return pd.read_pickle(file_path)

    high_risk_df = load_data(os.path.join(data_dir, latest_files['high_risk'])) if latest_files['high_risk'] else None
    low_risk_df = load_data(os.path.join(data_dir, latest_files['low_risk'])) if latest_files['low_risk'] else None

    if high_risk_df is None or low_risk_df is None:
        st.error("Failed to load necessary data. Please check your data files.")
        st.stop()

    # Get start and end dates from the data
    full_start_date = min(high_risk_df['Date'].min(), low_risk_df['Date'].min())
    full_end_date = max(high_risk_df['Date'].max(), low_risk_df['Date'].max())

    def get_data_directory():
        if os.path.exists('/mount/src/zoltarfinancial'):
            # Cloud environment
            return '/mount/src/zoltarfinancial/data'
        else:
            # Local environment
            return r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\data'
    
    # def find_most_recent_file(directory, prefix):
    #     files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    #     if not files:
    #         return None
    #     return max(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    
    # Use these functions in your main code
    output_dir_fund = get_data_directory()
    fundamentals_file_prefix = 'fundamentals_df_'
    
    # Find the most recent fundamentals file
    most_recent_fundamentals_file = find_most_recent_file(output_dir_fund, fundamentals_file_prefix)
    
    if most_recent_fundamentals_file:
        most_recent_fundamentals_path = os.path.join(output_dir_fund, most_recent_fundamentals_file)
        # Now you can use most_recent_fundamentals_path to load your file
    else:
        print("No fundamentals file found.")
    
    # Load the DataFrame if a file was found
    if most_recent_fundamentals_file:
        combined_fundamentals_df = pd.read_pickle(most_recent_fundamentals_file)
        print(f"Loaded fundamentals data from {most_recent_fundamentals_file}")
    else:
        print("No fundamentals file found.")
    # Call your main app function
    run_streamlit_app(high_risk_df, low_risk_df, full_start_date, full_end_date)
