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
    st.write("Hello, Streamlit!")
    
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

# Local imports
import sys
sys.path.append('C:/Users/apod7/StockPicker/scripts')
import robin_stocks as r
import os
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

import streamlit as st

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

import altair as alt



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

def generate_daily_rankings_strategies(validate_df, select_portfolio_func, models, start_date=None, stop_date=None, updated_models=None,
                                       initial_investment=20000,
                                       strategy_1_annualized_gain=0.7, strategy_1_loss_threshold=-0.07,
                                       strategy_2_gain_threshold=0.025, strategy_2_loss_threshold=-0.07,
                                       strategy_3_gain_threshold=0.04, strategy_3_loss_threshold=-0.07,
                                       skip=2, depth=20):
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
        return None, None, None
    
    print(f"SPY data shape: {spy_data.shape}")
    print(f"SPY data columns: {spy_data.columns}")
    print(f"spy_returns type: {type(spy_returns)}")
    print(f"spy_returns shape: {spy_returns.shape}")
    print(f"First few values of spy_returns:\n{spy_returns.head()}")
    
    # Initialize DataFrames to store rankings and daily gains/losses
    rankings_df = pd.DataFrame(columns=['Symbol'])
    
    # Initialize strategy tracking
    strategy_results = {
        'Strategy_1': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_2': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment},
        'Strategy_3': {'Book': [], 'Transactions': [], 'Daily_Value': [], 'Cash': initial_investment}
    }
    
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
            score_original, _, _, _, _, _, _, _, _ = calculate_roi_score(
                validate_df, current_data, symbol, spy_returns, models, updated_models
            )
            daily_rankings.append({'Symbol': symbol, 'Score': score_original, 'Close_Price': stock['Close Price']})
        
        daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('Score', ascending=False)
        daily_rankings_df['Rank'] = daily_rankings_df['Score'].rank(method='min', ascending=False).astype(int)
        daily_rankings_df['Close_Price'] = daily_rankings_df['Close_Price'].astype(float)

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
    
    return strategy_results, rankings_df, strategy_summaries


# @st.cache_data
# def load_data(file_path):
#     return pd.read_pickle(file_path)


@st.cache_data
def create_strategy_values_df(strategy_results):
    strategy_values = []
    for strategy, data in strategy_results.items():
        for daily_value in data['Daily_Value']:
            strategy_values.append({
                'Week': daily_value['Date'],
                'Strategy': strategy,
                'Value': daily_value['Value']
            })
    
    strategy_values_df = pd.DataFrame(strategy_values)
    
    # Drop duplicate entries if any
    strategy_values_df = strategy_values_df.drop_duplicates(subset=['Week', 'Strategy'])
    
    # Pivot the DataFrame
    strategy_values_df = strategy_values_df.pivot(index='Week', columns='Strategy', values='Value').reset_index()
    
    return strategy_values_df

@st.cache_data
def fill_missing_dates(strategy_values_df, _date_range):
    strategy_values_df = strategy_values_df.set_index('Week').reindex(_date_range, method='ffill').reset_index()
    strategy_values_df = strategy_values_df.rename(columns={'index': 'Week'})
    return strategy_values_df

# Set the page configuration at the very top
st.set_page_config(layout="wide")


#7.24.24 load skinny long file

@st.cache_data
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

# Load the combined data
combined_validate_df = load_data("combined_data")
spy_data = load_data("spy_data")

if combined_validate_df is not None and spy_data is not None:
    # Display some basic information
    # st.write("Data sources:", combined_validate_df['source'].unique())
    # st.write("Date range:", combined_validate_df['Week'].min(), "to", combined_validate_df['Week'].max())
    # st.write("Number of unique symbols:", combined_validate_df['Symbol'].nunique())

    full_start_date = combined_validate_df['Week'].min()
    full_end_date = combined_validate_df['Week'].max()
else:
    st.error("Failed to load necessary data. Please check data files and try again.")
    
    
    
    

def run_streamlit_app(validate_df, start_date, end_date):
    # st.set_page_config(layout="wide")

    # Initialize session state for iteration count and history
    if 'iteration' not in st.session_state:
        st.session_state.iteration = 0
    if 'history' not in st.session_state:
        st.session_state.history = []

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
            animation-duration: 180s;
        }
        .ticker-2 {
            animation-duration: 240s;
        }
        .ticker-3 {
            animation-duration: 300s;
        }
        .ticker-item {
            display: inline-block;
            padding: 0 1rem;
            font-size: 1.2rem;
        }
        @keyframes ticker {
            0% {
                transform: translate3d(0, 0, 0);
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
            font-size: 10px;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Define wise cracks
    wise_cracks = [
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
        "The common question that gets asked in business is, 'why?' That's a good question, but an equally valid question is, 'why not?'"
    ]

    wise_cracks.extend([
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
    ])
    
    # HTML for moving ribbons
    st.markdown(
        f"""
        <div class="ticker-wrapper">
            <div class="ticker ticker-1">
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in wise_cracks])}
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in wise_cracks])}
            </div>
        </div>
        <div class="ticker-wrapper">
            <div class="ticker ticker-2">
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in wise_cracks[20:] + wise_cracks[:20]])}
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in wise_cracks[20:] + wise_cracks[:20]])}
            </div>
        </div>
        <div class="ticker-wrapper">
            <div class="ticker ticker-3">
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in wise_cracks[40:] + wise_cracks[:40]])}
                {"".join([f'<span class="ticker-item">{crack}</span>' for crack in wise_cracks[40:] + wise_cracks[:40]])}
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

    st.title("Interactive Strategy Evaluation")
    
    st.write("Date range:", combined_validate_df['Week'].min(), "to", combined_validate_df['Week'].max())
    st.write("Number of unique symbols:", combined_validate_df['Symbol'].nunique())

    # Instructions section
    st.subheader("Instructions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="instructions">
            <strong>Settings:</strong><br>
            - Initial Investment: Set the initial amount to invest<br>
            - Ranking Metric: Choose the metric to rank strategies<br>
            - Skip Top N: Number of top ranked stocks to skip (possible outliers)<br>
            - Depth: Number of top ranked stocks in each purchase
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="instructions">
            <strong>Date Range:</strong><br>
            - Start Date: Select the start date for analysis<br>
            - End Date: Select the end date for analysis<br>
            <strong>Strategy Parameters:</strong><br>
            - Adjust thresholds for each strategy
            </div>
            """,
            unsafe_allow_html=True
        )

    # New section: Best Strategy Across All Iterations
    st.subheader("Best Strategy Across All Iterations")
    if 'best_strategy' not in st.session_state:
        st.session_state.best_strategy = None

    if st.session_state.best_strategy:
        best_strategy = st.session_state.best_strategy
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Strategy", best_strategy['Strategy'])
            st.metric("Number of Transactions", best_strategy['Number of Transactions'])
            st.metric("Current Holdings", best_strategy['Current Holdings'])
        with col2:
            st.metric("Initial Investment", f"${best_strategy['Starting Value']:.2f}")
            st.metric("Final Value", f"${best_strategy['Final Value']:.2f}")
            st.metric("Total Return", f"{best_strategy['Total Return']:.2%}")
        
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
        
        # Add strategy-specific parameters
        strategy_params = best_strategy['Settings']['Strategy Parameters']
        for param, value in strategy_params.items():
            settings_data["Setting"].append(f"{best_strategy['Strategy']}: {param}")
            settings_data["Value"].append(f"{value:.3f}")
        
        settings_df = pd.DataFrame(settings_data)
        st.table(settings_df)
    else:
        st.write("Run strategies to see the best performing strategy across all iterations.")
        
        
    # User inputs
    initial_investment = st.sidebar.number_input("Initial Investment", min_value=1000, max_value=1000000, value=10000, step=1000)
    ranking_metric = st.sidebar.selectbox("Ranking Metric", ["score_original", "score_updated", "expected_return", "best_er_original", "sharpe_ratio_original", "treynor_ratio_original"])
    
    col1, col2 = st.sidebar.columns(2)
    skip = col1.selectbox("Skip Top N", options=[0, 1, 2, 3, 4, 5], index=2)
    depth = col2.selectbox("Depth", options=[5, 10, 15, 20, 25, 30, 35], index=3)
    
    col3, col4 = st.sidebar.columns(2)
    start_date = col3.date_input("Start Date", start_date)
    end_date = col4.date_input("End Date", end_date)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    strategy_params = {
        'Strategy_1': {
            'annualized_gain_threshold': st.sidebar.slider("Strategy 1: Annualized Gain Threshold", 0.000, 2.000, 0.700, 0.100, format="%.3f"),
            'loss_threshold': st.sidebar.slider("Strategy 1: Loss Threshold", -0.200, 0.000, -0.070, 0.005, format="%.3f")
        },
        'Strategy_2': {
            'gain_threshold': st.sidebar.slider("Strategy 2: Gain Threshold", 0.000, 0.100, 0.025, 0.005, format="%.3f"),
            'loss_threshold': st.sidebar.slider("Strategy 2: Loss Threshold", -0.200, 0.000, -0.070, 0.005, format="%.3f")
        },
        'Strategy_3': {
            'gain_threshold': st.sidebar.slider("Strategy 3: Gain Threshold", 0.000, 0.100, 0.030, 0.005, format="%.3f"),
            'loss_threshold': st.sidebar.slider("Strategy 3: Loss Threshold", -0.200, 0.000, -0.070, 0.005, format="%.3f")
        }
    }
    
    if st.sidebar.button("Run Strategies"):
        st.session_state.iteration += 1
        
        strategy_results, rankings_df, strategy_summaries = generate_daily_rankings_strategies(
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
            strategy_params['Strategy_3']['gain_threshold'], 
            strategy_params['Strategy_3']['loss_threshold'],
            skip, 
            depth
        )
        
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
            'Starting Value': "${:.2f}",
            'Final Value': "${:.2f}",
            'Total Return': "{:.2%}",
            'Cash Balance': "${:.2f}"
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
            }
        }
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
        
        # Display Best Strategy Across All Iterations
        st.subheader("Best Strategy Across All Iterations")
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
        
        # Add strategy-specific parameters
        strategy_params = best_strategy['Settings']['Strategy Parameters']
        for param, value in strategy_params.items():
            settings_data["Setting"].append(f"{best_strategy['Strategy']}: {param}")
            settings_data["Value"].append(f"{value:.3f}")
        
        settings_df = pd.DataFrame(settings_data)
        st.table(settings_df)
    # else:
    #     st.write("Run strategies to see the best performing strategy across all iterations.")

    # Display Interactive Strategy Training History
    st.header("Interactive Strategy Training History")
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
        
# # Outside the button click handler, you can add:
# if 'best_strategy' not in st.session_state:
#     st.write("Run strategies to see the best performing strategy across all iterations.")
    
if __name__ == "__main__":
    run_streamlit_app(combined_validate_df, full_start_date, full_end_date) 
#7.21.24 - works

# def run_streamlit_app(validate_df, start_date, end_date):
#     st.title("Interactive Strategy Evaluation")
    
#     # User inputs
#     initial_investment = st.sidebar.number_input("Initial Investment", min_value=1000, max_value=1000000, value=20000, step=1000)
#     ranking_metric = st.sidebar.selectbox("Ranking Metric", ["score_original", "score_updated", "expected_return", "best_er_original", "sharpe_ratio_original", "treynor_ratio_original"])
#     skip = st.sidebar.slider("Skip Top N", 0, 10, 2)
#     depth = st.sidebar.slider("Depth", 1, 50, 20)
    
#     # Create initial DataFrames
#     rankings_df = create_rankings_df(validate_df, start_date, end_date, ranking_metric, skip, depth)
    
#     # Strategy parameters
#     strategy_params = {
#         'Strategy_1': {
#             'annualized_gain_threshold': st.sidebar.slider("Strategy 1: Annualized Gain Threshold", 0.0, 2.0, 0.7, 0.1),
#             'loss_threshold': st.sidebar.slider("Strategy 1: Loss Threshold", -0.2, 0.0, -0.07, 0.01)
#         },
#         'Strategy_2': {
#             'gain_threshold': st.sidebar.slider("Strategy 2: Gain Threshold", 0.0, 0.1, 0.025, 0.005),
#             'loss_threshold': st.sidebar.slider("Strategy 2: Loss Threshold", -0.2, 0.0, -0.07, 0.01)
#         },
#         'Strategy_3': {
#             'gain_threshold': st.sidebar.slider("Strategy 3: Gain Threshold", 0.0, 0.1, 0.04, 0.005),
#             'loss_threshold': st.sidebar.slider("Strategy 3: Loss Threshold", -0.2, 0.0, -0.07, 0.01)
#         }
#     }
    
#     # Update strategy results based on user inputs
#     strategy_results = update_strategy_results(rankings_df, initial_investment, strategy_params)
    
#     # Create strategy values DataFrame
#     strategy_values_df = create_strategy_values_df(strategy_results)
    
#     # Display results
#     st.subheader("Strategy Performance")
#     # st.line_chart(strategy_values_df.set_index('Date'))
#     import altair as alt

#     # Assuming strategy_values_df is your DataFrame with 'Date' and strategy columns
#     # Melt the DataFrame to create a long format suitable for Altair
#     melted_df = strategy_values_df.melt('Date', var_name='Strategy', value_name='Value')
    
#     # Create the Altair chart
#     chart = alt.Chart(melted_df).mark_line().encode(
#         x='Date:T',
#         y=alt.Y('Value:Q', scale=alt.Scale(zero=False)),  # True: This ensures the y-axis starts at zero
#         color='Strategy:N'
#     ).properties(
#         width=700,
#         height=400
#     )
    
#     # Display the chart in Streamlit
#     st.altair_chart(chart, use_container_width=True)
    
    
#     st.subheader("Rankings and Daily Gain/Loss")
#     st.dataframe(rankings_df)
    
#     st.subheader("Strategy Values")
#     st.dataframe(strategy_values_df)


# Run the Streamlit app
# if __name__ == "__main__":
#     # Load your validate_df here
#     validate_oot_df = pd.read_pickle(r'C:\Users\apod7\StockPicker\validate_oot_df_072024.pkl')
#     end_date = validate_oot_df['Week'].max() #- relativedelta(days=3)
#     start_date = end_date- relativedelta(days=29)
#     run_streamlit_app(validate_oot_df, start_date, end_date)
    

    # validate_df = pd.read_pickle(r'C:\Users\apod7\StockPicker\validate_df_072024.pkl')
    # end_date = validate_df['Week'].max() #- relativedelta(days=3)
    # start_date = end_date- relativedelta(days=29)
    # run_streamlit_app(validate_df, start_date, end_date)

    # Load your validate_df here
    # train_df = pd.read_pickle(r'C:\Users\apod7\StockPicker\train_df_072024.pkl')
    # end_date = train_df['Week'].max() - relativedelta(days=465)
    # start_date = end_date- relativedelta(days=89)
    # run_streamlit_app(train_df, start_date, end_date)

