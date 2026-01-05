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
    cd C:\ Users\apod\Stockpicker\app    
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



# 7.21 - back to dealing with spy again
@st.cache_data(ttl=1*24*3600,persist="disk")
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
    
    
   # 8.5.24 version  
@st.cache_data(ttl=1*24*3600, persist="disk")
def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None, initial_investment=20000, strategy_1_annualized_gain=0.4, strategy_1_loss_threshold=-0.07, strategy_2_gain_threshold=0.025, strategy_2_loss_threshold=-0.07, strategy_3_annualized_gain=0.4, strategy_3_loss_threshold=-0.07, skip=2, depth=20, ranking_metric='TstScr7_Top3ER'):
    if start_date is None:
        start_date = validate_df['Week'].min()
    if stop_date is None:
        stop_date = validate_df['Week'].max()
    
    start_date = pd.to_datetime(start_date)
    stop_date = pd.to_datetime(stop_date)

    # Initialize SPY data
    spy_data = validate_df[validate_df['Symbol'] == 'SPY'].copy()
    spy_data['Return'] = spy_data['Close Price'].pct_change()
    spy_data = spy_data.set_index('Week')

    # Create a Series of SPY returns for the entire date range
    date_range = pd.date_range(start=start_date, end=stop_date)
    spy_returns = spy_data['Return'].reindex(date_range).fillna(0)

    if spy_returns.empty:
        print("Error: No SPY data found in validate_df")
        return None, None, None, None, None, None, None

    # Reinitialize DataFrames to store rankings before each iteration
    ranking_metric_rankings = pd.DataFrame(columns=['Symbol'])
    score_original_rankings = pd.DataFrame(columns=['Symbol'])

    # Initialize strategy tracking
    strategy_results = {
        'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment}
    }

    # Calculate total number of days
    total_days = len(date_range)

    # Create a progress bar and progress text
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Initialize top_ranked_symbols_last_day
    top_ranked_symbols_last_day = []
    
    for i, current_date in enumerate(date_range):
        # Update progress bar and text
        progress = (i + 1) / total_days-0.0001
        progress_bar.progress(progress)
        progress_text.text(f"Progress: {progress:.2%}")

        current_data = validate_df[validate_df['Week'] == current_date]
        if current_data.empty:
            print(f"No data available for date: {current_date}")
            continue
        
        print(f"Processing date: {current_date}")

        # Calculate rankings for the day
        daily_rankings = []
        for _, stock in current_data.iterrows():
            symbol = stock['Symbol']
            score_original, best_er_original, beta, alpha_original, original_scores, score_updated, best_er_updated, alpha_updated, updated_scores, additional_scores, best_periods = calculate_multi_roi_score(
                validate_df, current_data, symbol, spy_returns, models, updated_models
            )
            daily_rankings.append({
                'Symbol': symbol,
                'Score_Original': score_original,
                'Score_Updated': score_updated,
                'Best_ER_Original': best_er_original,
                'Best_ER_Updated': best_er_updated,
                'TstScr1_AvgWin': additional_scores[0],
                'TstScr2_AvgReturn': additional_scores[1],
                'TstScr3_AvgER': additional_scores[2],
                'TstScr4_OlympER': additional_scores[3],
                'TstScr5_Top3Win': additional_scores[4],
                'TstScr6_Top3Return': additional_scores[5],
                'TstScr7_Top3ER': additional_scores[6],
                'Best_Period6': best_periods[0],
                'Best_Period7': best_periods[1],
                'Close_Price': stock['Close Price']
            })
        
        # Create DataFrame and sort by the selected ranking metric
        daily_rankings_df = pd.DataFrame(daily_rankings).sort_values(ranking_metric, ascending=False)
        daily_rankings_df['Rank'] = daily_rankings_df[ranking_metric].rank(method='min', ascending=False).astype(int)
        daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

        # Sort and rank based on the ranking metric and score_original
        daily_ranking_metric_df = daily_rankings_df[['Symbol', ranking_metric]].sort_values(ranking_metric, ascending=False)
        daily_ranking_metric_df['Rank'] = daily_ranking_metric_df[ranking_metric].rank(method='min', ascending=False)

        daily_score_original_df = daily_rankings_df[['Symbol', 'Score_Original']].sort_values('Score_Original', ascending=False)
        daily_score_original_df['Rank'] = daily_score_original_df['Score_Original'].rank(method='min', ascending=False)

        # Add to ranking DataFrames
        ranking_metric_rankings = ranking_metric_rankings.merge(daily_rankings_df[['Symbol', ranking_metric]], on='Symbol', how='outer', suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}'))
        ranking_metric_rankings = ranking_metric_rankings.rename(columns={ranking_metric: current_date.strftime("%Y-%m-%d")})

        score_original_rankings = score_original_rankings.merge(daily_rankings_df[['Symbol', 'Score_Original']], on='Symbol', how='outer', suffixes=('', f'_{current_date.strftime("%Y-%m-%d")}'))
        score_original_rankings = score_original_rankings.rename(columns={'Score_Original': current_date.strftime("%Y-%m-%d")})

        # Implement strategies
        if current_date == start_date:
            print(f"Initializing strategies on start date: {current_date}")
            top_stocks = daily_rankings_df.iloc[skip:skip + depth]['Symbol'].tolist()  # Use variable skip and depth
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
                    if 'Expected_Sell_Date' not in holding:
                        # Calculate ROI score to get best_period_original
                        _, _, _, _, original_scores, _, _, _, _ = calculate_roi_score(
                            validate_df, current_data, symbol, spy_returns, models, updated_models
                        )
                        best_period_original = max(original_scores, key=lambda k: original_scores[k]['er'])
                        holding['Expected_Sell_Date'] = holding['Buy_Date'] + timedelta(days=best_period_original)
                        holding['Best_Period'] = best_period_original
                    
                    days_held = (current_date - holding['Buy_Date']).days
                    if days_held > 0:
                        annualized_gain = (1 + gain_loss) ** (365 / days_held) - 1
                        if annualized_gain > strategy_3_annualized_gain or gain_loss < strategy_3_loss_threshold or current_date >= holding['Expected_Sell_Date']:
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
            data['Book'] = new_book

            # Reinvestment logic (updated to invest all available cash)
            if data['Cash'] > 0:
                top_stocks = daily_rankings_df.iloc[skip:skip + depth]['Symbol'].tolist()  # Use variable skip and depth
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

    # Remove the progress bar after completion
    progress_bar.empty()

    # Generate final report
    strategy_summaries = {}
    for strategy, data in strategy_results.items():
        final_value = data['Daily_Value'][-1]['Value']
        total_return = (final_value - initial_investment) / initial_investment
        strategy_summaries[strategy] = {
            'Starting Value': initial_investment,
            'Final Value': final_value,
            'Total Return': total_return,
            'Number of Transactions': len(data['Transactions']),
            'Current Holdings': len(data['Book']),
            'Cash Balance': data['Cash']
        }

    # Generate current holdings report
    current_holdings_report = {}
    for strategy, data in strategy_results.items():
        holdings = []
        for holding in data['Book']:
            symbol = holding['Symbol']
            buy_date = holding['Buy_Date']
            buy_price = holding['Buy_Price']
            shares = holding['Shares']
            
            current_price = daily_rankings_df[daily_rankings_df['Symbol'] == symbol]['Close_Price'].values[0]
            days_since_purchase = (stop_date - buy_date).days
            gain_loss = (current_price - buy_price) / buy_price
            
            holdings.append({
                'Symbol': symbol,
                'Buy Date': buy_date,
                'Buy Price': buy_price,
                'Current Price': current_price,
                'Shares': shares,
                'Days Since Purchase': days_since_purchase,
                f'Gain/Loss (as of {stop_date.date()})': gain_loss
            })
        
        current_holdings_report[strategy] = pd.DataFrame(holdings)

    # Save the top-ranked 20 symbols for the last day
    top_ranked_symbols_last_day = daily_rankings_df.head(20).to_dict('records')
    
    # Return all the results
    return strategy_results, ranking_metric_rankings, strategy_summaries, current_holdings_report, top_ranked_symbols_last_day, ranking_metric_rankings, score_original_rankings

# 7.31.24 - new version to take advantage of all 28 models in some shape (7 new scores, and 2 new best periods added)

@st.cache_data(ttl=1*24*3600,persist="disk")
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
        symbol_returns = symbol_data['Close Price'].pct_change().dropna()
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


@st.cache_data(ttl=1*24*3600, persist="disk")
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
                'Week': daily_value['Date'],
                'Strategy': strategy,
                'Value': value
            })
    
    strategy_values_df = pd.DataFrame(strategy_values)
    
    # Drop duplicate entries if any
    strategy_values_df = strategy_values_df.drop_duplicates(subset=['Week', 'Strategy'])
    
    # Pivot the DataFrame
    strategy_values_df = strategy_values_df.pivot(index='Week', columns='Strategy', values='Value').reset_index()
    
    return strategy_values_df

@st.cache_data(ttl=1*24*3600, persist="disk")
def fill_missing_dates(strategy_values_df, _date_range):
    strategy_values_df = strategy_values_df.set_index('Week').reindex(_date_range, method='ffill').reset_index()
    strategy_values_df = strategy_values_df.rename(columns={'index': 'Week'})
    return strategy_values_df

# Set the page configuration at the very top
st.set_page_config(layout="wide")


# 7.26.24 - let user select which file to analyze - Large, Mid, or Small-caps - Streamlit can't handla all to be loaded (not sure about Small actually)
@st.cache_data(ttl=1*24*3600,persist="disk")
def get_latest_files(data_dir):
    files = os.listdir(data_dir)
    latest_files = {'Small': None, 'Mid': None, 'Large': None, 'Tot': None}
    
    for file in files:
        if file.startswith('combined_data_') and file.endswith('.pkl') and not file.startswith('spy_'):
            for category in ['Small', 'Mid', 'Large', 'Tot']:
                if category in file:
                    date_str = file.split('_')[-1].split('.')[0]
                    date = datetime.strptime(date_str, '%Y%m%d')
                    if latest_files[category] is None or date > datetime.strptime(latest_files[category].split('_')[-1].split('.')[0], '%Y%m%d'):
                        latest_files[category] = file
    
    return latest_files


# 7.26.24 - selection of small, mid, large
# Your existing load_data function
@st.cache_data(ttl=1*24*3600,persist="disk")
def load_data(file_prefix):
    base_dir = "data"
    today = date.today()
    
    # Try to load the file with today's date
    for days_back in range(7):  # Try up to 7 days back
        current_date = today - timedelta(days=days_back)
        filename = f"{file_prefix}_{current_date.strftime('%Y%m%d')}.pkl"
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            return pd.read_pickle(file_path)
    
    # If no file found in the last 7 days, list available files and let user choose
    st.warning(f"No recent {file_prefix} file found. Please select a file manually.")
    available_files = [f for f in os.listdir(base_dir) if f.startswith(file_prefix) and f.endswith('.pkl')]
    if available_files:
        selected_file = st.selectbox(f"Select a {file_prefix} file:", available_files)
        return pd.read_pickle(os.path.join(base_dir, selected_file))
    else:
        st.error(f"No {file_prefix} files found in the data directory.")
        return None

@st.cache_data(persist="disk")
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

@st.cache_data(persist="disk")
def get_image_urls(date):
    base_url = "https://github.com/apod-1/ZoltarFinancial/raw/main/daily_ranks/"
    return [
        f"{base_url}expected_returns_path_Small_{date}.png",
        f"{base_url}expected_returns_path_Mid_{date}.png",
        f"{base_url}expected_returns_path_Large_{date}.png"
    ]

@st.cache_data(persist="disk")
def get_latest_file(prefix):
    import requests
    url = f"https://api.github.com/repos/apod-1/ZoltarFinancial/contents/daily_ranks"
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
@st.cache_data(ttl=1*24*3600, persist="disk")
def generate_last_day_rankings(validate_df, end_date, initial_investment, strategy_params, ranking_metric):
    start_date = end_date - timedelta(days=5)  # Get last 3 days
    # Reset DataFrames in session state before each run
    st.session_state.ranking_metric_rankings = pd.DataFrame(columns=['Symbol'])
    st.session_state.score_original_rankings = pd.DataFrame(columns=['Symbol'])

    _, rankings_df, _, _, _,ranking_metric_rankings, score_original_rankings = generate_daily_rankings_strategies(
        validate_df, 
        None,  # select_portfolio_func
        None,  # models
        start_date, 
        end_date, 
        None,  # updated_models
        initial_investment,
        strategy_params['Strategy_1']['annualized_gain_threshold'], 
        strategy_params['Strategy_1']['loss_threshold'],
        strategy_params['Strategy_2']['gain_threshold'], 
        strategy_params['Strategy_2']['loss_threshold'],
        strategy_params['Strategy_3']['annualized_gain_threshold'], 
        strategy_params['Strategy_3']['loss_threshold'],
        skip=2, 
        depth=20,
        ranking_metric=ranking_metric
    )
    
    print(f"Generated rankings DataFrame columns: {rankings_df.columns}")
    print(f"Generated rankings DataFrame shape: {rankings_df.shape}")
    print(f"First few rows of generated rankings DataFrame:\n{rankings_df.head()}")
    print(f"Data types of columns:\n{rankings_df.dtypes}")
    
    return rankings_df

@st.cache_data(ttl=1*24*3600, persist="disk")
def generate_last_week_rankings(validate_df, end_date, models, updated_models=None):
    start_date = end_date - timedelta(days=4)
    
    # Get SPY data
    spy_data = validate_df[validate_df['Symbol'] == 'SPY'].copy()
    spy_data['Return'] = spy_data['Close Price'].pct_change()
    spy_data = spy_data.set_index('Week')
    
    # Create a Series of SPY returns for the last 3 days
    date_range = pd.date_range(start=start_date, end=end_date)
    spy_returns = spy_data['Return'].reindex(date_range).fillna(0)
    
    # Initialize DataFrame to store rankings
    rankings_df = pd.DataFrame(columns=['Symbol', 'Date', 'TstScr7_Top3ER'])
    
    for current_date in date_range:
        print(f"Processing date: {current_date}")
        current_data = validate_df[validate_df['Week'] == current_date]
        latest_stocks = current_data.to_dict('records')
        
        daily_rankings = []
        
        for stock in latest_stocks:
            symbol = stock['Symbol']
            if symbol == 'SPY':
                continue
            
            score_original, best_er, beta, alpha_original, original_scores, _, _, _, _, additional_scores, _ = calculate_multi_roi_score(
                validate_df, current_data, symbol, spy_returns, models, updated_models
            )
            
            daily_rankings.append({
                'Symbol': symbol,
                'Date': current_date,
                'TstScr7_Top3ER': additional_scores[6]  # Index 6 corresponds to TstScr7_Top3ER TRY 5 (RETURN)
            })
        
        # Add daily rankings to the main DataFrame
        rankings_df = pd.concat([rankings_df, pd.DataFrame(daily_rankings)], ignore_index=True)
    
    return rankings_df

# 8.2.24 - late night: use only top x to define strength of portfolio potential
@st.cache_data(persist="disk")
def calculate_market_rank_metrics(rankings_df):
    # Calculate the average TstScr7_Top3ER for each day
    daily_avg_metric = rankings_df.groupby('Date')['TstScr7_Top3ER'].mean()

    # Sort the daily average metrics
    sorted_metrics = daily_avg_metric.sort_values(ascending=False)

    # Calculate the mean of the top 20 values after omitting the top 2
    if len(sorted_metrics) > 22:
        avg_market_rank = sorted_metrics.iloc[2:22].mean()
    else:
        avg_market_rank = sorted_metrics.mean()  # Fallback if there are not enough values

    latest_market_rank = daily_avg_metric.iloc[-1]

    # Calculate standard deviation
    std_dev = sorted_metrics.iloc[2:22].std() if len(sorted_metrics) > 22 else sorted_metrics.std()

    # Calculate low and high settings
    low_setting = avg_market_rank - 2 * std_dev
    high_setting = avg_market_rank + 2 * std_dev

    return avg_market_rank, std_dev, latest_market_rank, low_setting, high_setting



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
@st.cache_data(ttl=1*24*3600, persist="disk")
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
def display_interactive_rankings(rankings_df, ranking_type):
    # Prepare data
    filtered_df, top_stocks = prepare_rankings_data(rankings_df, ranking_type)
    
    # Dropdown for selecting number of top stocks to display
    top_n = st.selectbox(f"Select number of top stocks ({ranking_type})", [5, 10, 15, 20, 25], key=f"{ranking_type}_top_n")
    
    # Create multiselect for choosing which stocks to display
    selected_stocks = st.multiselect(f"Select stocks to display ({ranking_type})", top_stocks, default=top_stocks[:top_n], key=f"{ranking_type}_stocks")
    
    # Filter based on user selection
    display_df = filtered_df[filtered_df['Symbol'].isin(selected_stocks)]
    
    # Melt the dataframe to long format for plotting
    try:
        date_columns = [col for col in display_df.columns if col != 'Symbol']
        melted_df = display_df.melt(id_vars=['Symbol'], value_vars=date_columns, var_name='Date', value_name='Score')
        melted_df['Date'] = pd.to_datetime(melted_df['Date']).dt.date  # Convert to date without time

        # Create the plot
        fig = go.Figure()
        for stock in selected_stocks:
            stock_data = melted_df[melted_df['Symbol'] == stock]
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Score'], mode='lines', name=stock))

        fig.update_layout(
            title=f'Top {top_n} Stocks Score Over Time ({ranking_type})',
            xaxis_title='Date',
            yaxis_title='Score',
            xaxis=dict(
                tickformat='%Y-%m-%d',  # Format x-axis ticks as YYYY-MM-DD
                tickmode='auto',
                nticks=10  # Adjust this number to control the density of x-axis labels
            )
        )

        st.plotly_chart(fig)

        # Display the dataframe
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("DataFrame structure:")
        st.write(display_df.head())
        st.write("DataFrame columns:")
        st.write(display_df.columns)







@st.cache_data(persist="disk")
def generate_top_20_table(top_ranked_symbols_last_day=None):
    if 'best_strategy' in st.session_state and st.session_state.best_strategy is not None and 'Top_Ranked_Symbols' in st.session_state.best_strategy:
        # Use the best strategy data
        ranking_metric = st.session_state.best_strategy['Settings']['Ranking Metric']
        max_date = st.session_state.best_strategy.get('Date')
        top_ranked_symbols = st.session_state.best_strategy['Top_Ranked_Symbols'][:20]
    elif top_ranked_symbols_last_day is not None:
        # Use the provided top_ranked_symbols_last_day
        ranking_metric = 'TstScr7_Top3ER'  # Adjust this if you use a different metric for initial simulation
        max_date = st.session_state.get('last_simulation_date')
        top_ranked_symbols = top_ranked_symbols_last_day[:20]
    else:
        return "No data available for top ranked symbols."

    # Ensure max_date is a valid datetime object
    if max_date is None or max_date == 'Unknown Date':
        max_date = pd.Timestamp.now().date()
    else:
        try:
            max_date = pd.to_datetime(max_date).date()
        except Exception as e:
            st.error(f"Error converting max_date to datetime: {e}. Using current date instead.")
            max_date = pd.Timestamp.now().date()

    top_symbols_data = {
        "Rank": list(range(1, 21)),
        "Symbol": [symbol['Symbol'] for symbol in top_ranked_symbols],
        "Score": [f"{symbol[ranking_metric]:.2f}" for symbol in top_ranked_symbols],
        "Best ER": [f"{symbol['TstScr7_Top3ER'] * 100:.2f}%" for symbol in top_ranked_symbols],
        "Best Period": [f"{int(symbol['Best_Period7'])}" for symbol in top_ranked_symbols]
    }

    html_table = f"""
    <h2>Top 20 Strategy for {(max_date + BDay(1)).strftime('%Y-%m-%d')}</h2>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Score</th>
            <th>Best ER</th>
            <th>Best Period</th>
        </tr>
    """

    for i in range(20):
        html_table += f"""
        <tr>
            <td>{top_symbols_data['Rank'][i]}</td>
            <td>{top_symbols_data['Symbol'][i]}</td>
            <td>{top_symbols_data['Score'][i]}</td>
            <td>{top_symbols_data['Best ER'][i]}</td>
            <td>{top_symbols_data['Best Period'][i]}</td>
        </tr>
        """

    html_table += "</table>"
    return html_table




def send_user_email(user_email):
    try:
        sender_email = st.secrets["GMAIL"]["GMAIL_ACCT"]
        sender_password = st.secrets["GMAIL"]["GMAIL_PASS"]
    except KeyError:
        st.error("Gmail credentials not found in secrets. Please check your configuration.")
        return

    recipient_email = user_email
    subject = "Your Top 20 Strategy (powered by Zoltar)"
    
    msg = MIMEMultipart()
    msg['From'] = f"ZF <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    top_ranked_symbols_last_day = st.session_state.get('top_ranked_symbols_last_day')
    top_20_table = generate_top_20_table(top_ranked_symbols_last_day)
    
    html_body = f"""
    <html>
      <body>
        <p>Establishing communication with ZF community (phase 1 complete).</p>
        {top_20_table}
        <p><img src="data:image/png;base64,{get_image_base64()}" alt="ZoltarSurf"></p>
        <p>May the riches be with you..</p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_body, 'html'))
 
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.success('Email sent successfully!')
    except Exception as e:
        st.error(f'Error sending email: {e}')
        
        
        

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


def run_streamlit_app(validate_df, start_date, end_date):
    # st.set_page_config(layout="wide")
    import requests
    # Initialize session state for iteration count and history
    if 'iteration' not in st.session_state:
        st.session_state.iteration = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'email' not in st.session_state:
        st.session_state.email = ""

    #7.24.24 - make results persistent in the session and Initialize session state variables if they don't exist
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
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = None

    if 'show_image' not in st.session_state:
        st.session_state.show_image = False
    
    if 'new_wisdom' not in st.session_state:
        st.session_state.new_wisdom = ""    
    if 'initial_simulation_run' not in st.session_state:
        st.session_state.initial_simulation_run=False

    # Initialize new DataFrames for rankings
    ranking_metric_rankings = pd.DataFrame(columns=['Symbol'])
    score_original_rankings = pd.DataFrame(columns=['Symbol'])

        
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
    st.title("Interactive Strategy Evaluation Engine powered by Zoltar Stock Ranking")
    

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
        if st.button("Submit"):
            if new_wisdom:
                st.session_state.wise_cracks.append(new_wisdom)
                st.session_state.new_wisdom = ""  # Clear the stored new wisdom
                st.rerun()  # Rerun the app to reflect changes

    st.write("IMPORTANT: For best experience please use in landscape mode on high-memory device (optimization under way to address lackluster mobile experience). Thank you for your patience!")
    st.write("Date range:", combined_validate_df['Week'].min().strftime('%m-%d-%Y'), "to", combined_validate_df['Week'].max().strftime('%m-%d-%Y'))
    st.write("Number of unique symbols:", combined_validate_df['Symbol'].nunique())

    # Instructions section
    st.subheader("Instructions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="instructions">
            <strong>Date Range:</strong><br>
            Data Load: Total Caps loaded by default (1,200 pre-filtered); can load other sets based on Market Cap size<br>
            - Use Pre-selected buttons: Select from data used for Training Ranks, Validation, or Out-of-Time Validation Ranges<br>
            <br>
            Narrow down selected ranges further with more precise selection (USE THIS OPTION TO LIMT DATE RANGE)<br>
            - Start Date: Select the start date for analysis<br>
            - End Date: Select the end date for analysis<br>
            <br>
            ATTENTION: Please limit date range to avoid significantly increased run-times and resource limits<br>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="instructions">
            <strong>Settings:</strong><br>
            - Initial Investment: Set the initial amount to invest<br>
            - Ranking Metric: Choose the pre-defined ranking metrics to use for strategies (all are driven by Zoltar Score Suite) <br>
             * Note: Updated Scores not available<br>
            - Skip Top N: Number of top ranked stocks to skip (remove possible outliers)<br>
            - Depth: Number of top ranked stocks in each purchase (this will be replaced with Score Percentile cut-off in the future)<br>
            <strong>Sell Rules:</strong><br>
            - Use sliders to adjust stop-loss and gain thresholds (Strategy 1 and 3 use annualized target; 2 uses flat target gain percent)<br>
            </div>
            """,
            unsafe_allow_html=True
        )

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

        
    # New section: Best Strategy Across All Iterations
    st.subheader("Best Strategy Across All Iterations")
    if 'best_strategy' not in st.session_state:
        st.session_state.best_strategy = None
    max_date = combined_validate_df['Week'].max()
    
    # 8.3.24 - add initial run to present something..
    # Run initial simulation on app load
    if not st.session_state.initial_simulation_run:
        st.write("Running initial simulation...")
        try:
            max_date = combined_validate_df['Week'].max()
            # Reset DataFrames in session state before each run
            st.session_state.ranking_metric_rankings = pd.DataFrame(columns=['Symbol'])
            st.session_state.score_original_rankings = pd.DataFrame(columns=['Symbol'])

            # Run simulation with default settings
            strategy_results, rankings_df, strategy_summaries, current_holdings_report, top_ranked_symbols_last_day, ranking_metric_rankings, score_original_rankings = generate_daily_rankings_strategies(
                combined_validate_df,
                None,  # select_portfolio_func
                None,  # models
                max_date,  # start_date
                max_date,  # end_date
                None,  # updated_models
                10000,  # initial_investment
                0.35,  # strategy_1_annualized_gain
                -0.07,  # strategy_1_loss_threshold
                0.015,  # strategy_2_gain_threshold
                -0.20,  # strategy_2_loss_threshold
                0.4,  # strategy_3_annualized_gain
                -0.20,  # strategy_3_loss_threshold
                2,  # skip
                15,  # depth
                'TstScr7_Top3ER'  # ranking_metric
            )            
    
            st.subheader(f"Top 20 {selected_category} Cap Strategy for {(max_date + pd.offsets.BDay(1)).strftime('%Y-%m-%d')}")
        
            ranking_metric = 'TstScr7_Top3ER'
            
            # Ensure top_ranked_symbols_last_day is a list of dictionaries
            if isinstance(top_ranked_symbols_last_day, pd.DataFrame):
                top_ranked_symbols_last_day = top_ranked_symbols_last_day.to_dict('records')
            
            top_symbols_data = {
                "Rank": list(range(1, 21)),
                "Symbol": [symbol['Symbol'] for symbol in top_ranked_symbols_last_day[:20]],
                "Score": [],
                "Best ER": [f"{symbol[ranking_metric] * 100:.2f}%" for symbol in top_ranked_symbols_last_day[:20]],
                "Best Period": [f"{int(symbol['Best_Period7'])}" for symbol in top_ranked_symbols_last_day[:20]]
            }
        
            # Check if the max score is less than 1 and multiply by 100 if needed
            scores = [symbol[ranking_metric] for symbol in top_ranked_symbols_last_day[:20]]
            if max(scores) < 1:
                scores = [score * 100 for score in scores]
            
            # Round the scores to 2 decimal places
            top_symbols_data["Score"] = [f"{score:.2f}" for score in scores]
        
            top_symbols_df = pd.DataFrame(top_symbols_data)
            
            # Display the table with sorting and scrolling functionality
            st.dataframe(top_symbols_df.style
                         .set_properties(**{'text-align': 'center'})
                         .set_table_styles([
                             {'selector': 'th', 'props': [('font-size', '12px'), ('text-align', 'center')]},
                             {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'center')]},
                         ]))
            
            st.session_state.initial_simulation_run = True
            # Store top_ranked_symbols_last_day in session state
            st.session_state.top_ranked_symbols_last_day = top_ranked_symbols_last_day
            st.session_state.last_simulation_date = max_date  # where max_date is the last date of your simulation
            st.session_state.ranking_metric_rankings = ranking_metric_rankings
            st.session_state.score_original_rankings = score_original_rankings
            st.write("Initial simulation completed.")
        except Exception as e:
            st.error(f"An error occurred during the initial simulation: {str(e)}")
    
    if st.session_state.best_strategy:
        best_strategy = st.session_state.best_strategy
        date_for_display = pd.to_datetime(best_strategy['Settings']['End Date'])
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Best Strategy", best_strategy['Strategy'])
            st.metric("Number of Transactions", best_strategy['Number of Transactions'])
            st.metric("Current Holdings", best_strategy['Current Holdings'])
            
            # Add a new section for displaying the top-ranked symbols
            st.subheader(f"Top 20 {selected_category} Cap Strategy for {(date_for_display + pd.offsets.BDay(1)).strftime('%Y-%m-%d')}")
    
            if 'Top_Ranked_Symbols' in st.session_state.best_strategy:
                ranking_metric = st.session_state.best_strategy['Settings']['Ranking Metric']
                
                top_symbols_data = {
                    "Rank": list(range(1, 21)),
                    "Symbol": [symbol['Symbol'] for symbol in st.session_state.best_strategy['Top_Ranked_Symbols']],
                    "Score": [],
                    "Best ER": [f"{symbol['TstScr7_Top3ER'] * 100:.2f}%" for symbol in st.session_state.best_strategy['Top_Ranked_Symbols']],
                    "Best Period": [f"{int(symbol['Best_Period7'])}" for symbol in st.session_state.best_strategy['Top_Ranked_Symbols']]
                }
        
                # Check if the max score is less than 1 and multiply by 100 if needed
                scores = [symbol[ranking_metric] for symbol in st.session_state.best_strategy['Top_Ranked_Symbols']]
                if max(scores) < 1:
                    scores = [score * 100 for score in scores]
                
                # Round the scores to 2 decimal places
                top_symbols_data["Score"] = [f"{score:.2f}" for score in scores]
        
                top_symbols_df = pd.DataFrame(top_symbols_data)
                
                # Display the table with sorting and scrolling functionality
                st.dataframe(top_symbols_df.style
                             .set_properties(**{'text-align': 'center'})
                             .set_table_styles([
                                 {'selector': 'th', 'props': [('font-size', '12px'), ('text-align', 'center')]},
                                 {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'center')]},
                             ]))
            else:
                st.write("Top ranked symbols information not available.")
    
        with col2:
            st.metric("Initial Investment", f"${best_strategy['Starting Value']:.2f}")
            st.metric("Final Value", f"${best_strategy['Final Value']:.2f}")
            st.metric("Total Return", f"{best_strategy['Total Return']:.2%}")
    
            st.subheader("Best Strategy Settings")
            settings_data = {
                "Setting": ["Initial Investment", "Ranking Metric", "Skip Top N", "Depth", "Start Date", "End Date"],
                "Value": [
                    f"${best_strategy['Settings']['Initial Investment']:.2f}",
                    best_strategy['Settings']['Ranking Metric'],
                    best_strategy['Settings']['Skip Top N'],
                    best_strategy['Settings']['Depth'],
                    best_strategy['Settings']['Start Date'],
                    best_strategy['Settings']['End Date']
                ]
            }
    
            # Add strategy-specific parameters
            strategy_params = best_strategy['Settings']['Strategy Parameters']
            strategy_name = best_strategy.get('Strategy Name', 'Unknown Strategy')
            for param, value in strategy_params.items():
                settings_data["Setting"].append(f"{strategy_name} - {param}")
                if isinstance(value, (int, float)):
                    settings_data["Value"].append(f"{value:.3f}")
                else:
                    settings_data["Value"].append(str(value))
    
            settings_df = pd.DataFrame(settings_data)
    
            # Display the table without index and with reduced width
            st.table(settings_df.style
                     .hide(axis="index")
                     .set_properties(**{'width': '300px'})
                     .set_table_styles([
                         {'selector': 'th', 'props': [('font-size', '12px')]},
                         {'selector': 'td', 'props': [('font-size', '12px')]},
                         {'selector': '', 'props': [('width', '300px')]}
                     ]))
    
        # Create and display the line chart
        if 'Daily_Value' in best_strategy:
            daily_values = pd.DataFrame(best_strategy['Daily_Value'])
            daily_values['Date'] = pd.to_datetime(daily_values['Date'])
            daily_values.set_index('Date', inplace=True)
    
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(daily_values.index, daily_values['Value'])
            ax.set_title(f"{best_strategy['Strategy']} Performance")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            plt.xticks(rotation=45)
            plt.tight_layout()
    
            # Convert plot to PNG image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()
            plt.close()
    
            # Display the image in Streamlit
            st.image(f"data:image/png;base64,{img_str}", use_column_width=True)
    else:
        st.write("Run strategies to see the best performing strategy across all iterations.")

    # 8.5 new section to display rankings over the period of simulation (will get messy, will need to clean up)
    # New section: Latest Iteration Ranks Research
    st.subheader("Latest Iteration Ranks Research")
    
    # Check if the rankings are available
    if 'ranking_metric_rankings' in st.session_state and 'score_original_rankings' in st.session_state:
        col1, col2 = st.columns(2)
        ranking_metric = getattr(st.session_state, 'ranking_metric', 'TstScr7_Top3ER')
        with col1:
            st.subheader(f"{ranking_metric} Rankings")
            display_interactive_rankings(st.session_state.ranking_metric_rankings, ranking_metric)
    
        with col2:
            st.subheader("Original Score Rankings")
            display_interactive_rankings(st.session_state.score_original_rankings, "Score")
    else:
        st.write("Rankings data not available. Please run a simulation first.")
        
    # 7.27 - new radio buttons to help select date range    
    # User inputs
    initial_investment = st.sidebar.number_input("Initial Investment", min_value=1000, max_value=1000000, value=10000, step=1000)
    ranking_metric = st.sidebar.selectbox("Ranking Metric", [
        "Score_Original", "Score_Updated", "Best_ER_Original", "Best_ER_Updated",
        "TstScr1_AvgWin", "TstScr2_AvgReturn", "TstScr3_AvgER", "TstScr4_OlympER",
        "TstScr5_Top3Win", "TstScr6_Top3Return", "TstScr7_Top3ER"
    ], index=10)    
    col1, col2 = st.sidebar.columns(2)
    skip = col1.selectbox("Skip Top N", options=[0, 1, 2, 3, 4, 5], index=2)
    depth = col2.selectbox("Depth", options=[5, 10, 15, 20, 25, 30, 35], index=2)
    
    # Date range selection section
    st.sidebar.subheader("Select Date Range for Analysis")
    
    # Extract date ranges for validate, validate_oot, and train
    validate_dates = combined_validate_df[combined_validate_df['source'] == 'validate']['Week']
    validate_oot_dates = combined_validate_df[combined_validate_df['source'] == 'validate_oot']['Week']
    train_dates = combined_validate_df[combined_validate_df['source'] == 'train']['Week']
    
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
        start_date = combined_validate_df['Week'].min()
        end_date = combined_validate_df['Week'].max()
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
    col3, col4 = st.sidebar.columns(2)
    start_date = col3.date_input("Start Date", start_date)
    end_date = col4.date_input("End Date", end_date)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    strategy_params = {}
    
    # Strategy 1
    centered_header("Strategy 1")
    strategy_params['Strategy_1'] = {
        'annualized_gain_threshold': st.sidebar.slider(
            "Annualized Gain Threshold", 
            10.0, 100.0, 35.0, 5.0,  # Changed step to 5.0
            format="%.0f%%", 
            key="strategy1_gain"
        ) / 100,  # Convert to decimal
        'loss_threshold': st.sidebar.slider(
            "Loss Threshold", 
            -20.0, 0.0, -7.0, 0.5, 
            format="%.1f%%", 
            key="strategy1_loss"
        ) / 100  # Convert to decimal
    }
    
    # Strategy 2
    centered_header("Strategy 2")
    strategy_params['Strategy_2'] = {
        'gain_threshold': st.sidebar.slider(
            "Gain Threshold", 
            0.5, 10.0, 1.5, 0.5, 
            format="%.1f%%", 
            key="strategy2_gain"
        ) / 100,  # Convert to decimal
        'loss_threshold': st.sidebar.slider(
            "Loss Threshold", 
            -20.0, 0.0, -20.0, 0.5, 
            format="%.1f%%", 
            key="strategy2_loss"
        ) / 100  # Convert to decimal
    }
    
    # Strategy 3
    centered_header("Strategy 3 [Optimized Sell Date]")
    strategy_params['Strategy_3'] = {
        'annualized_gain_threshold': st.sidebar.slider(
            "Annualized Gain Threshold", 
            10.0, 100.0, 40.0, 5.0,  # Changed step to 5.0
            format="%.0f%%", 
            key="strategy3_gain"
        ) / 100,  # Convert to decimal
        'loss_threshold': st.sidebar.slider(
            "Loss Threshold", 
            -20.0, 0.0, -20.0, 0.5, 
            format="%.1f%%", 
            key="strategy3_loss"
        ) / 100  # Convert to decimal
    }


 

    
    if st.sidebar.button("Run Strategies"):
        st.session_state.iteration += 1
        # Reset DataFrames in session state before each run
        st.session_state.ranking_metric_rankings = pd.DataFrame(columns=['Symbol'])
        st.session_state.score_original_rankings = pd.DataFrame(columns=['Symbol'])

        strategy_results, rankings_df, strategy_summaries, current_holdings_report, top_ranked_symbols_last_day, ranking_metric_rankings, score_original_rankings = generate_daily_rankings_strategies(
            validate_df, 
            None,  # select_portfolio_func
            None,  # models
            start_date, 
            end_date, 
            None,  # updated_models
            initial_investment,
            strategy_params['Strategy_1']['annualized_gain_threshold'], 
            strategy_params['Strategy_1']['loss_threshold'],
            strategy_params['Strategy_2']['gain_threshold'], 
            strategy_params['Strategy_2']['loss_threshold'],
            strategy_params['Strategy_3']['annualized_gain_threshold'], 
            strategy_params['Strategy_3']['loss_threshold'],
            skip, 
            depth,
            ranking_metric=ranking_metric
        )
        # Calculate average days held for each strategy
        for strategy, data in strategy_results.items():
            if data['Transactions']:
                hold_periods = [(t['Sell_Date'] - t['Buy_Date']).days for t in data['Transactions']]
                avg_days_held = sum(hold_periods) / len(hold_periods)
            else:
                avg_days_held = 0
            strategy_summaries[strategy]['Average Days Held'] = avg_days_held
        
        spy_data = validate_df[validate_df['Symbol'] == 'SPY'].copy()
        
        if spy_data.empty:
            st.error("Error: No SPY data found in validate_df")
            return
        
        spy_data['Return'] = spy_data['Close Price'].pct_change()
        spy_data = spy_data.set_index('Week')
        
        date_range = pd.date_range(start=start_date, end=end_date)
        spy_returns = spy_data['Return'].reindex(date_range).fillna(0)
        
        spy_values = [initial_investment]
        for ret in spy_returns:
            spy_values.append(spy_values[-1] * (1 + ret))
        
        strategy_results['SPY (Baseline)'] = {'Daily_Value': [{'Date': date, 'Value': value} for date, value in zip(date_range, spy_values[1:])]}
        
        strategy_values_df = create_strategy_values_df(strategy_results)
        strategy_values_df = fill_missing_dates(strategy_values_df, date_range)
        
        spy_values_df = pd.DataFrame({'Week': date_range, 'SPY (Baseline)': spy_values[1:]})
        combined_df = pd.merge(strategy_values_df, spy_values_df, on='Week', how='outer')
    
        if 'SPY (Baseline)_x' in combined_df.columns and 'SPY (Baseline)_y' in combined_df.columns:
            combined_df['SPY (Baseline)'] = combined_df['SPY (Baseline)_x'].combine_first(combined_df['SPY (Baseline)_y'])
            combined_df = combined_df.drop(columns=['SPY (Baseline)_x', 'SPY (Baseline)_y'])
    
        columns_to_fill = [col for col in combined_df.columns if col != 'Week']
        combined_df[columns_to_fill] = combined_df[columns_to_fill].ffill()
    
        st.subheader("Strategy Performance")
        melted_df = combined_df.melt('Week', var_name='Strategy', value_name='Value')
        chart = alt.Chart(melted_df).mark_line().encode(
            x='Week:T',
            y=alt.Y('Value:Q', scale=alt.Scale(zero=False)),
            color='Strategy:N'
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
        
        st.subheader("Strategy Values")
        st.dataframe(combined_df.style.format({col: "${:.2f}" for col in combined_df.columns if col != 'Week'}))
        
        st.subheader("Strategy Summary")
        
        strategy_summary_df = pd.DataFrame(strategy_summaries).T
        
        st.dataframe(strategy_summary_df.style.format({
            'Starting Value': "${:.2f}",
            'Final Value': "${:.2f}",
            'Total Return': "{:.2%}",
            'Average Days Held': "{:.1f}"  # Format to one decimal place
        }))
        
        st.subheader("Transactions")
        col1, col2, col3 = st.columns(3)
        for i, strategy in enumerate(['Strategy_1', 'Strategy_2', 'Strategy_3']):
            transactions_df = pd.DataFrame(strategy_results[strategy]['Transactions'])
            if not transactions_df.empty:
                if i == 0:
                    col1.dataframe(transactions_df)
                elif i == 1:
                    col2.dataframe(transactions_df)
                else:
                    col3.dataframe(transactions_df)

        # Display current holdings for each strategy
        st.subheader("Current Holdings")
        col1, col2, col3 = st.columns(3)
        for i, strategy in enumerate(['Strategy_1', 'Strategy_2', 'Strategy_3']):
            holdings = strategy_results[strategy]['Book']
            holdings_df = pd.DataFrame(holdings)
            if not holdings_df.empty:
                if i == 0:
                    col1.dataframe(holdings_df)
                elif i == 1:
                    col2.dataframe(holdings_df)
                else:
                    col3.dataframe(holdings_df)
            
        # Store results in session state
        st.session_state.strategy_results = strategy_results
        st.session_state.strategy_summary_df = strategy_summary_df
        st.session_state.combined_df = combined_df
        # Store the rankings in the session state
        st.session_state.ranking_metric_rankings = ranking_metric_rankings
        st.session_state.score_original_rankings = score_original_rankings

        # Update best strategy
        current_best = max(strategy_summaries.items(), key=lambda x: x[1]['Total Return'])
        st.session_state.best_strategy = {
            'Strategy': current_best[0],
            **current_best[1],
            'Settings': {
                'Initial Investment': initial_investment,
                'Ranking Metric': ranking_metric,
                'Skip Top N': skip,
                'Depth': depth,
                'Start Date': start_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d'),
                'Strategy Parameters': strategy_params[current_best[0]]
            },
            'Top_Ranked_Symbols': top_ranked_symbols_last_day
        }
        # st.session_state.best_strategy['Top_Ranked_Symbols'] = top_ranked_symbols_last_day
        
        # Record settings and summary
        history_entry = {
            'Iteration': st.session_state.iteration,
            'Settings': {
                'Initial Investment': initial_investment,
                'Ranking Metric': ranking_metric,
                'Skip Top N': skip,
                'Depth': depth,
                'Start Date': start_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d'),
                'Strategy Parameters': strategy_params
            },
            'Summary': strategy_summary_df.to_dict()
        }
        st.session_state.history.append(history_entry)
        
        #8.1.24 - first just display the top-ranked symbols for the last day in text - then we'll move on to having links/etc
        st.write("Top Ranked Symbols using Best Strategy for the Last Day:")
        st.write(st.session_state.best_strategy['Top_Ranked_Symbols'])
        
        
        # Display Best Strategy Across All Iterations
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
            st.metric("Current Holdings", best_strategy['Current Holdings'])
        
        # Add table with strategy settings
        st.subheader("Best Strategy Settings")
        settings_data = {
            "Setting": ["Initial Investment", "Ranking Metric", "Skip Top N", "Depth", "Start Date", "End Date"],
            "Value": [
                f"${best_strategy['Settings']['Initial Investment']:.2f}",
                best_strategy['Settings']['Ranking Metric'],
                best_strategy['Settings']['Skip Top N'],
                best_strategy['Settings']['Depth'],
                best_strategy['Settings']['Start Date'],
                best_strategy['Settings']['End Date']
            ]
        }
        
        settings_df = pd.DataFrame(settings_data)
        
        # Remove the first column (Index) and make the table narrower
        # st.table(settings_df.style.hide(axis='index').set_table_styles([{
        #     'selector': 'table',
        #     'props': [('width', '50%')]
        # }, {
        #     'selector': 'th',
        #     'props': [('text-align', 'center')]
        # }, {
        #     'selector': 'td',
        #     'props': [('text-align', 'center')]
        # }]))
        
        # Add strategy-specific parameters
        strategy_params = best_strategy['Settings']['Strategy Parameters']
        strategy_name = best_strategy.get('Strategy Name', 'Unknown Strategy')
        for param, value in strategy_params.items():
            settings_data["Setting"].append(f"{strategy_name} - {param}")
            if isinstance(value, (int, float)):
                settings_data["Value"].append(f"{value:.3f}")
            else:
                settings_data["Value"].append(str(value))
        
        settings_df = pd.DataFrame(settings_data)
        st.table(settings_df)
    # else:
    #     st.write("Run strategies to see the best performing strategy across all iterations.")
    #7.24.24pm Display persistent results
    if st.session_state.combined_df is not None:
        st.subheader("Strategy Performance")
        melted_df = st.session_state.combined_df.melt('Week', var_name='Strategy', value_name='Value')
        chart = alt.Chart(melted_df).mark_line().encode(
            x='Week:T',
            y=alt.Y('Value:Q', scale=alt.Scale(zero=False)),
            color='Strategy:N'
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
        
        st.subheader("Strategy Values")
        st.dataframe(st.session_state.combined_df.style.format({col: "${:.2f}" for col in st.session_state.combined_df.columns if col != 'Week'}))
        
        st.subheader("Strategy Summary")
        st.dataframe(st.session_state.strategy_summary_df.style.format({
            'Starting Value': "${:.2f}",
            'Final Value': "${:.2f}",
            'Total Return': "{:.2%}",
            'Cash Balance': "${:.2f}"
        }))
        
        st.subheader("Transactions")
        col1, col2, col3 = st.columns(3)
        for i, strategy in enumerate(['Strategy_1', 'Strategy_2', 'Strategy_3']):
            transactions_df = pd.DataFrame(st.session_state.strategy_results[strategy]['Transactions'])
            if not transactions_df.empty:
                if i == 0:
                    col1.dataframe(transactions_df)
                elif i == 1:
                    col2.dataframe(transactions_df)
                else:
                    col3.dataframe(transactions_df)
        # Clear Results button
        if st.sidebar.button("Clear Results"):
            st.session_state.strategy_results = None
            st.session_state.strategy_summary_df = None
            st.session_state.combined_df = None
            st.experimental_rerun()
            
                
    # Display Interactive Strategy Training History
    st.header("Strategy Training History")
    if st.session_state.history:
    # Display Interactive Strategy Training History
        for entry in st.session_state.history:
            st.subheader(f"Iteration {entry['Iteration']}")
            st.json(entry['Settings'])
            st.dataframe(pd.DataFrame(entry['Summary']).style.format({
                'Starting Value': "${:.2f}",
                'Final Value': "${:.2f}",
                'Total Return': "{:.2%}",
                'Cash Balance': "${:.2f}",
                'Number of Transactions': "{:.0f}",
                'Current Holdings': "{:.0f}"
            }))
            st.markdown("---")
    else:
        st.write("No iterations have been run yet. Use the 'Run Strategies' button to start.")


    # 8.3.24 - email yourself    
    # Sidebar input for email
    st.sidebar.markdown("---")  # Add a separator
    user_email = st.sidebar.text_input("Email your best strategy results:")
    if st.sidebar.button("Send Email"):
        if user_email:
            if 'top_ranked_symbols_last_day' in st.session_state or 'best_strategy' in st.session_state:
                send_user_email(user_email)
            else:
                st.sidebar.error("Strategy data not available. Please run a simulation first.")
        else:
            st.sidebar.error("Please enter a valid email address.")            

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
   
    
    # Assuming `combined_df` has a 'Week' column
    max_week = combined_validate_df['Week'].max().strftime('%m-%d-%Y')
    # For max_week
    max_week = pd.to_datetime(max_week)  # Ensure max_week is a datetime object
    next_bd = (max_week + BDay(1)).strftime('%m-%d-%Y')
    
    # For combined_validate_df
    max_date = pd.to_datetime(combined_validate_df['Week'].max())  # Ensure it's a datetime object
    next_bd_comb = (max_date + BDay(1)).strftime('%m-%d-%Y')


    # Display image when button is clicked
    if st.session_state.show_image:
        # Title of the Section
        st.markdown(f"<h2 style='text-align: center;'>Recommendations for {next_bd}</h2>", unsafe_allow_html=True)
    
        # Generate rankings_df for the last 3 days
        end_date = combined_validate_df['Week'].max()
        rankings_df = generate_last_week_rankings(
            validate_df=combined_validate_df,
            end_date=end_date,
            models=None,
            updated_models=None
        )
    
        # Calculate Market Rank Metrics
        avg_market_rank, std_dev, latest_market_rank, low_setting, high_setting = calculate_market_rank_metrics(rankings_df)
    
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
            title={'text': f"Market Gauge for {selected_category}"}
        ))
    
        st.plotly_chart(fig)

    
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
        st.markdown(f"<h2 style='text-align: center;'>Overall Zoltar Stock Picks - {next_bd_comb}</h2>", unsafe_allow_html=True)
    
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
    data_dir = '/mount/src/zoltarfinancial/data'  # Adjust this path as needed
    latest_files = get_latest_files(data_dir)
    
    # Sidebar elements
    with st.sidebar:
        # Dropdown for selecting data source
        selected_category = st.selectbox(
            "Choose a market cap category:",
            options=['Tot','Small', 'Mid', 'Large'],
            index=0,  # This ensures 'Tot' is always the default
            format_func=lambda x: f"{x} Cap ({latest_files[x]})"
        )

        # Button to load data with confirmation
        if st.button("Load Data"):
            st.session_state.show_confirmation = True
            st.session_state.start_time = time.time()

            # Load the selected file
            if latest_files[selected_category]:
                file_path = os.path.join(data_dir, latest_files[selected_category])
                combined_validate_df = pd.read_pickle(file_path)
                st.success(f"Loaded {selected_category} Cap data: {latest_files[selected_category]}")
            else:
                st.error(f"No data file found for {selected_category} Cap")
                st.stop()
            
            # Load SPY data
            spy_data = load_data("spy_data_Large")    
            if spy_data is None:
                st.error("Failed to load SPY data. Please check your data files.")
                st.stop()

    # Load the selected file
    if latest_files[selected_category]:
        file_path = os.path.join(data_dir, latest_files[selected_category])
        combined_validate_df = pd.read_pickle(file_path)
    else:
        st.error(f"No data file found for {selected_category} Cap")
        st.stop()
    
    # Load SPY data
    spy_data = load_data("spy_data_Large")    
    if combined_validate_df is not None and spy_data is not None:
        # Get start and end dates from the data
        full_start_date = combined_validate_df['Week'].min()
        full_end_date = combined_validate_df['Week'].max()
    
        # Call your main app function
        run_streamlit_app(combined_validate_df, full_start_date, full_end_date)
    else:
        st.error("Failed to load necessary data. Please check your data files.")
