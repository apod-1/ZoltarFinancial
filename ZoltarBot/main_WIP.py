# activate myflaskenv
# C:\Users\apod7\StockPicker\app\ZoltarFinancial\ZoltarBot>git add .
# git commit -m "main.py change - summary and riches phrase"
# git push -f heroku main

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import openai
import os
import pandas as pd
from datetime import datetime, timedelta, date, time

# import pandas as pd

# 2.16.25 - new functions to handle uptodate info on stocks from my git

def load_data2(file_path):
    return pd.read_pickle(file_path)

def get_available_versions(data_dir, selected_dates=None, selected_time_slots=None):
    files = os.listdir(data_dir)
    versions = set()
    for file in files:
        if file.startswith(('high_risk_rankings_', 'low_risk_rankings_')):
            version = '_'.join(file.split('_')[-2:]).split('.')[0]
            date_part = version[:8]
            time_slot_part = version.split('-')[1] if '-' in version else "FULL OVERNIGHT UPDATE"
            
            if (selected_dates is None or date_part in selected_dates) and \
               (selected_time_slots is None or time_slot_part in selected_time_slots):
                versions.add(version)
                
    return sorted(list(versions), reverse=True)

def convert_to_ranking_format(df, ranking_metric):
    df['Date'] = pd.to_datetime(df['Date'])
    pivot_df = df.pivot(index='Symbol', columns='Date', values=ranking_metric)
    pivot_df.reset_index(inplace=True)
    return pivot_df




def load_data2(file_path):
    return pd.read_pickle(file_path)
        # unique_time_slots = ["FULL OVERNIGHT UPDATE", "PREMARKET UPDATE", "AFTEROPEN UPDATE","MORNING UPDATE","AFTERNOON UPDATE","PRECLOSE UPDATE","AFTERCLOSE UPDATE","WEEKEND UPDATE"]  # Example slots

if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
    data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
else:
    data_dir = '/mount/src/zoltarfinancial/daily_ranks'
# @st.cache_data
available_versions = get_available_versions(data_dir)
default_time_slots = ["FULL OVERNIGHT UPDATE", "WEEKEND UPDATE"]
chronological_order = [
    "FULL OVERNIGHT UPDATE",
    "PREMARKET UPDATE",
    "MORNING UPDATE",
    "AFTEROPEN UPDATE",
    "AFTERNOON UPDATE",
    "PRECLOSE UPDATE",
    "AFTERCLOSE UPDATE",
    "WEEKEND UPDATE"
]

# unique_time_slots = high_risk_df_long['Time_Slot'].unique()

# Replace NaN values with "FULL OVERNIGHT UPDATE"
unique_time_slots = [slot if pd.notna(slot) else "FULL OVERNIGHT UPDATE" for slot in chronological_order]    

ordered_time_slots = sorted(unique_time_slots, key=lambda x: chronological_order.index(x) if x in chronological_order else len(chronological_order))

filtered_versions_intra = [v for v in available_versions if (v.split('-')[1] if '-' in v else "FULL OVERNIGHT UPDATE") in ordered_time_slots]  # replaced default_time_slots to make use of most recent
filtered_versions_daily = [v for v in available_versions if (v.split('-')[1] if '-' in v else "FULL OVERNIGHT UPDATE") in default_time_slots]  
filtered_versions = filtered_versions_intra[:15]



# latest_files = get_latest_files(data_dir)
# if latest_files is None:
# st.stop()  # Stop the app execution if files couldn't be loaded
def get_latest_files_spin(data_dir=None):
    if data_dir is None:
        # Determine the environment and set the appropriate data directory
        if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
            # Local environment
            data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
        else:
            # Cloud environment
            data_dir = '/mount/src/zoltarfinancial/daily_ranks'
   
    latest_files = {}
    loaded_successfully=False
    # 2.7.25 - change to continue loading until successful
    while not(loaded_successfully):
        try:
            for category in ['high_risk', 'low_risk']:
                files = [f for f in os.listdir(data_dir) if f.startswith(f"{category}_rankings_") and f.endswith(".pkl")]
                if files:
                    # Use the file's modification time to determine the latest file
                    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
                    latest_files[category] = latest_file
                    loaded_successfully=True
                else:
                    latest_files[category] = None
        except FileNotFoundError:
            # with st.spinner("New version of Zoltar Ranks is loading. The process usually takes ~1 min to complete. Please try again..."):
            #     sleep(30)  # Wait for 60 seconds
            print("Still loading. This may take another minute. Thank you for your patience.")
            # sleep(10)  # Wait for 60 seconds
            return None
   
    return latest_files

        # except FileNotFoundError:
        #     with st.spinner("New version of Zoltar Ranks is loading. Please wait..."):
        #         time.sleep(10)  # Wait for 10 seconds before trying again
        #     st.info("Still loading. This may take a few minutes. Thank you for your patience.")

# In your main code
# with st.spinner("Loading the latest Zoltar Ranks..."):
latest_files = get_latest_files_spin()
    # sleep(2)  # Wait for 60 seconds

# st.success("Latest Zoltar Ranks loaded successfully!")    

# Capture file_update_date
# Capture file_update_date with hours and minutes
file_update_date = datetime.fromtimestamp(os.path.getmtime(os.path.join(data_dir, latest_files['high_risk'])))
# high_risk_rankings_20241024_184311.pkl
# Load the data
# @st.cache_data
def load_data(file_path):
    return pd.read_pickle(file_path)

# 1.9.25 - ready to start using newly created product versions?     
high_risk_df = load_data(os.path.join(data_dir, latest_files['high_risk'])) if latest_files['high_risk'] else None
low_risk_df = load_data(os.path.join(data_dir, latest_files['low_risk'])) if latest_files['low_risk'] else None



high_risk_rankings = convert_to_ranking_format(high_risk_df, f"High_Risk_Score")
low_risk_rankings = convert_to_ranking_format(low_risk_df, f"Low_Risk_Score")

merged_df_low = pd.merge(low_risk_rankings, combined_fundamentals_df, on='Symbol', how='left')
merged_df_high = pd.merge(high_risk_rankings, combined_fundamentals_df, on='Symbol', how='left')

# Get all date columns
date_columns = [col for col in merged_df_high.columns if isinstance(col, pd.Timestamp)]

# # Filter date columns based on the selected date range
# date_columns = [col for col in date_columns]
# Filter date columns based on the selected date range
date_columns = [col for col in date_columns if full_start_date <= col <= full_end_date]

# if not date_columns:
#     st.error(f"No data available for the selected date range for rankings.")
#     return

# # Use the latest date column in the selected range for ranking
latest_date = max(date_columns)
ranking_column = latest_date

# Sort the filtered DataFrame
sorted_df_low = merged_df_low.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
sorted_df_high = merged_df_high.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
# Sort by the last column for merged_df_low
# sorted_df_low = merged_df_low.sort_values(by=merged_df_low.columns[-1], ascending=False).reset_index(drop=True)

# Sort by the last column for merged_df_high
# sorted_df_high = merged_df_high.sort_values(by=merged_df_high.columns[-1], ascending=False).reset_index(drop=True)
# Get the data for selected versions with filters applied
high_risk_df_long, low_risk_df_long = select_versions2(15, None, ordered_time_slots) #12.1.24 -  changed from default_time_slots to get most recent, uppped to 15 from 10

# Sort both DataFrames by 'Symbol', 'Version', and 'Date' in descending order
high_risk_df_long = high_risk_df_long.sort_values(by=['Symbol', 'Version', 'Date'], ascending=[True, True, False])
low_risk_df_long = low_risk_df_long.sort_values(by=['Symbol', 'Version', 'Date'], ascending=[True, True, False])

# Now, select the last record for each combination of 'Symbol' and 'Version' (most recent Date)
high_risk_df_long = high_risk_df_long.groupby(['Symbol', 'Version']).first().reset_index()
low_risk_df_long = low_risk_df_long.groupby(['Symbol', 'Version']).first().reset_index()
# # First, select the rows with the maximum 'Date' for each Symbol and Version
# high_risk_df_long = high_risk_df_long.loc[high_risk_df_long.groupby(['Symbol', 'Version'])['Date'].idxmax()]
# low_risk_df_long = low_risk_df_long.loc[low_risk_df_long.groupby(['Symbol', 'Version'])['Date'].idxmax()]

# Now, select the maximum score and Close_Price for each Symbol and Version for high_risk_df
high_risk_df_long = high_risk_df_long.groupby(['Symbol', 'Version']).agg({
    'High_Risk_Score': 'last',  # Get the maximum High_Risk_Score for each group
    'Close_Price': 'last'  # Ensure the Close_Price is from the latest Date by using 'last'
}).reset_index()

# For low_risk_df, get the max Low_Risk_Score and Close_Price from the latest record
low_risk_df_long = low_risk_df_long.groupby(['Symbol', 'Version']).agg({
    'Low_Risk_Score': 'last',  # Get the maximum Low_Risk_Score for each group
    'Close_Price': 'last'  # Ensure the Close_Price is from the latest Date by using 'last'
}).reset_index()

# Sort the dataframes by Version in descending order
high_risk_df_long = high_risk_df_long.sort_values('Version', ascending=False)
low_risk_df_long = low_risk_df_long.sort_values('Version', ascending=False)            
# Sort the dataframes by Version in descending order
high_risk_df_long = high_risk_df_long.sort_values('Version', ascending=False)
low_risk_df_long = low_risk_df_long.sort_values('Version', ascending=False)

# Create new columns for Date and Time Slot
high_risk_df_long['Date'] = high_risk_df_long['Version'].str[:8]
high_risk_df_long['Time_Slot'] = high_risk_df_long['Version'].str.split('-').str[1]

low_risk_df_long['Date'] = low_risk_df_long['Version'].str[:8]
low_risk_df_long['Time_Slot'] = low_risk_df_long['Version'].str.split('-').str[1]

# Create filters for Date and Time Slot
unique_dates = high_risk_df_long['Date'].unique()
unique_time_slots = high_risk_df_long['Time_Slot'].unique()

# Replace NaN values with "FULL OVERNIGHT UPDATE"
unique_time_slots = [slot if pd.notna(slot) else "FULL OVERNIGHT UPDATE" for slot in unique_time_slots]    
# Use top_x to limit the number of stocks displayed - selected to do top 20 (not top_x as it was before
display_df_low = sorted_df_low.head(50)
# display_df_low_all = sorted_df_low.head(1200)
print(display_df_low)
display_df_high = sorted_df_high.head(5)
unique_dates = sorted(set(version[:8] for version in filtered_versions), reverse=True)
# Extract unique time slots from available versions
unique_time_slots = sorted(set(version.split('-')[1] if '-' in version else "FULL OVERNIGHT UPDATE" for version in available_versions))

# Multi-select for stocks
# default_stocks_low_all = sorted_df_low['Symbol'].tolist()
# default_stocks_low = display_df_low['Symbol'].tolist()
# default_stocks_high = display_df_high['Symbol'].tolist()
# 1.9.25 - include spy
default_stocks_low = ['SPY'] + display_df_low['Symbol'].tolist()
default_stocks_high = ['SPY'] + display_df_high['Symbol'].tolist()
    # selected_stocks = st.multiselect(
    #     f"Select stocks to display ({ranking_type})",
    #     options=sorted_df['Symbol'].tolist(),
    #     default=default_stocks,
    #     key=f"{ranking_type}_stock_multiselect"
    # )





# Filter for custom stocks and get the latest date for each stock
# custom_df_low = sorted_df_low[sorted_df_low['Symbol'].isin(default_stocks_low)]
#12.1.24 -  trying to do all 
custom_df_low = sorted_df_low[sorted_df_low['Symbol'].isin(default_stocks_low)]


# custom_df_low = custom_df_low.sort_values('Date').groupby('Symbol').last().reset_index()

# Handle None values
custom_df_low['Fundamentals_Sector'] = custom_df_low['Fundamentals_Sector'].fillna('Unknown Sector')
custom_df_low['Fundamentals_Industry'] = custom_df_low['Fundamentals_Industry'].fillna('Unknown Industry')
    
# Filter for custom stocks and get the latest date for each stock
custom_df_high = merged_df_high[merged_df_high['Symbol'].isin(default_stocks_high)]
# custom_df_high = custom_df_high.sort_values('Date').groupby('Symbol').last().reset_index()

# Handle None values
custom_df_high['Fundamentals_Sector'] = custom_df_high['Fundamentals_Sector'].fillna('Unknown Sector')
custom_df_high['Fundamentals_Industry'] = custom_df_high['Fundamentals_Industry'].fillna('Unknown Industry')




# 11.24.24 PLACEHOLDER SECTION TO LOAD ALL TOP RANKS TO BE ABLE TO ANSWER ANY QUESTIONS ABOUT THEM IMMEDIATELY        
def generate_stock_data(custom_stocks, high_risk_df_long, low_risk_df_long):
    stock_data = []
    for stock in custom_stocks:
        stock_data.append(f"\n{stock}:")
        stock_data.append("| Version | Date | Time Slot | High Zoltar Rank | Low Zoltar Rank | Close Price | High Zoltar Rank Index to Avg | Low Zoltar Rank Index to Avg |")
        stock_data.append("|---------|------|-----------|-----------------|----------------|-------------|------------------------|------------------------|")
        
        high_risk_stock = high_risk_df_long[high_risk_df_long['Symbol'] == stock]
        low_risk_stock = low_risk_df_long[low_risk_df_long['Symbol'] == stock]
        
        # 11.24.24 - correct for negative  values
        # Calculate shifts for both High and Low Risk Scores
        shift_high = abs(min(high_risk_stock['High_Risk_Score'].min(), 0))
        shift_low = abs(min(low_risk_stock['Low_Risk_Score'].min(), 0))
        
        # Calculate averages with shift
        avg_high_score = (high_risk_stock['High_Risk_Score'] + shift_high).mean()
        avg_low_score = (low_risk_stock['Low_Risk_Score'] + shift_low).mean()
        
        for _, row in high_risk_stock.iterrows():
            low_risk_row = low_risk_stock[low_risk_stock['Version'] == row['Version']].iloc[0]
            
            # Calculate indices with shift
            high_risk_index = (row['High_Risk_Score'] + shift_high) / avg_high_score
            low_risk_index = (low_risk_row['Low_Risk_Score'] + shift_low) / avg_low_score
            
            # Calculate real scores
            high_risk_score_real = row['High_Risk_Score'] * 100
            low_risk_score_real = low_risk_row['Low_Risk_Score'] * 100

        # for _, row in high_risk_stock.iterrows():
        #     low_risk_row = low_risk_stock[low_risk_stock['Version'] == row['Version']].iloc[0]
        #     high_risk_index = row['High_Risk_Score'] / high_risk_stock['High_Risk_Score'].mean()
        #     low_risk_index = low_risk_row['Low_Risk_Score'] / low_risk_stock['Low_Risk_Score'].mean()
            
            stock_data.append(f"| {row['Version']} | {row['Date']} | {row['Time_Slot']} | {row['High_Risk_Score']*100:.2f}% | {low_risk_row['Low_Risk_Score']*100:.2f}% | ${row['Close_Price']:.2f} | {high_risk_index:.2f} | {low_risk_index:.2f} |")
        
        # Calculate and add averages
        avg_high_risk = high_risk_stock['High_Risk_Score'].mean() * 100
        avg_low_risk = low_risk_stock['Low_Risk_Score'].mean() * 100
        avg_close_price = high_risk_stock['Close_Price'].mean()
        stock_data.append(f"\nAverages: High Zoltar Rank: {avg_high_risk:.2f}%, Low Zoltar Rank: {avg_low_risk:.2f}%, Close Price: ${avg_close_price:.2f}")
        
        # Add trend information
        high_risk_trend = "increasing" if high_risk_stock['High_Risk_Score'].iloc[0] > high_risk_stock['High_Risk_Score'].iloc[-1] else "decreasing"
        low_risk_trend = "increasing" if low_risk_stock['Low_Risk_Score'].iloc[0] > low_risk_stock['Low_Risk_Score'].iloc[-1] else "decreasing"
        price_trend = "increasing" if high_risk_stock['Close_Price'].iloc[0] > high_risk_stock['Close_Price'].iloc[-1] else "decreasing"
        stock_data.append(f"Trends: High Zoltar Rank: {high_risk_trend}, Low Zoltar Rank: {low_risk_trend}, Price: {price_trend}")
    
    return "\n".join(stock_data)
# def generate_fundamentals_data(custom_df):
#     fundamentals_data = []
#     fundamentals_data.append("| Symbol | PE | PB | Dividends | Ex-Dividend Date | Market Cap | Sector | Industry | Best Hold Period (days) |")
#     fundamentals_data.append("|--------|----|----|-----------|-------------------|------------|--------|----------|------------------------------|")
    
#     for _, row in custom_df.iterrows():
#         fundamentals_data.append(f"| {row['Symbol']} | {row['Fundamentals_PE']:.2f} | {row['Fundamentals_PB']:.2f} | {row['Fundamentals_Dividends']:.2f} | {row['Fundamentals_ExDividendDate']} | {row['Fundamentals_MarketCap']:,.0f} | {row['Fundamentals_Sector']} | {row['Fundamentals_Industry']} | {row['High_Risk_Score_HoldPeriod']} |")   #{row['High_Risk_Score_HoldPeriod']}
def generate_fundamentals_data(custom_df):
    fundamentals_data = []
    fundamentals_data.append("| Symbol | PE | PB | Dividends | Ex-Dividend Date | Market Cap | Sector | Industry |")
    fundamentals_data.append("|--------|----|----|-----------|-------------------|------------|--------|----------|")
    
    for _, row in custom_df.iterrows():
        fundamentals_data.append(f"| {row['Symbol']} | {row['Fundamentals_PE']:.2f} | {row['Fundamentals_PB']:.2f} | {row['Fundamentals_Dividends']:.2f} | {row['Fundamentals_ExDividendDate']} | {row['Fundamentals_MarketCap']:,.0f} | {row['Fundamentals_Sector']} | {row['Fundamentals_Industry']} |")   #{row['High_Risk_Score_HoldPeriod']}
    
    return "\n".join(fundamentals_data)
def generate_fundamentals_data_l(custom_df):
    fundamentals_data = []
    fundamentals_data.append("| Symbol | PE | PB | Dividends | Ex-Dividend Date | Market Cap | Sector | Industry |")
    fundamentals_data.append("|--------|----|----|-----------|-------------------|------------|--------|----------|")
    
    for _, row in custom_df.iterrows():
        fundamentals_data.append(f"| {row['Symbol']} | {row['Fundamentals_PE']:.2f} | {row['Fundamentals_PB']:.2f} | {row['Fundamentals_Dividends']:.2f} | {row['Fundamentals_ExDividendDate']} | {row['Fundamentals_MarketCap']:,.0f} | {row['Fundamentals_Sector']} | {row['Fundamentals_Industry']} |")
    
    return "\n".join(fundamentals_data)
pre_prompt_low = f"""
The data below represents the top ranked stocks for the most recent data point using Low Zoltar Ranks that predict average expected returns from buying stock now and selling over the next 14 days; also included are corresponding stock prices for {len(default_stocks_low)} stocks: {', '.join(default_stocks_low)}.
The user may or may not be familiar with these stocks, and the stocks on this list should always be correlated against the user's portfolio section, if it exists.
If a stock the user is asking about is not on the list, recommend that the user adds the stock to their Research Portfolio, or runs the Simulation to reveal more stocks.
The user is particularly interested in finding undervalued stocks through looking for 1) the highest High and Low Zoltar Rank for the most recent data point, 2) with highest (and non-negative) average low Zoltar Ranks, 3) with higher index to average (also non-negative), and 3) preferably at a lower price than in prior data points for that stock.
Make sure that the final answer looks at the historical trends and addresses the user interest. If user is interested in high returns, then they are interested in highest High Zoltar Rank, if user is interested in consistent performance, then the user is interested in highest average Low Zoltar Rank; and together with those a higher index to average for the current data point, combined with deflated price for most recent data point could signal an undervalued stock.
When user is interested in diversification, they want the top Zoltar Ranks from multiple sectors.
When user wants to select stocks to improve their portfolio, this is the list to use to recommend stocks from - but don't mix it with their existing portfolio.  The stocks in this section aim for more stability in return prediction.
Together with data in this section, additional section with similar organization shows the user's current research portfolio, and stocks on this list could be recommended to replace some of the stocks in this portfolio expected to perform worse, especially in the same industries and sectors.

The data covers {len(unique_dates)} dates from {min(unique_dates)} to {max(unique_dates)}, with time slots: {', '.join(unique_time_slots)}.

Data for each stock:
{generate_stock_data(default_stocks_low, high_risk_df_long, low_risk_df_long)}

Fundamentals data for each stock:
{generate_fundamentals_data_l(custom_df_low)}

Historical ranges across all stocks:
- High Zoltar Rank: {high_risk_df_long['High_Risk_Score'].min()*100:.2f}% to {high_risk_df_long['High_Risk_Score'].max()*100:.2f}%
- Low Zoltar Rank: {low_risk_df_long['Low_Risk_Score'].min()*100:.2f}% to {low_risk_df_long['Low_Risk_Score'].max()*100:.2f}%
- Close Price: ${high_risk_df_long['Close_Price'].min():.2f} to ${high_risk_df_long['Close_Price'].max():.2f}

For each stock, we calculate:
1. Average of expected returns in prior versions
2. Current expected return
3. Index to average expected returns (current / average)

Based on these calculations, we provide indicators:
- Strong Buy: If average Low Zoltar Rank >= 70bps and Index to Avg > 1.3, or if average Low Zoltar Rank >= 0bps and Index to Avg > 1.5
- Hold & Trim: If average Low Zoltar Rank >= 70bps and Index to Avg <= 1.3, or if 0bps < average Low Zoltar Rank < 70bps and Index to Avg > 1
- Moderate Sell: If 0bps <= last Low Zoltar Rank < 70bps and Index to Avg <= 1
- Strong Sell: If last Low Zoltar Rank <= 0bps and index to Avg <= 1
- Promising: For other cases

The data shows the historical trend of High and Low Zoltar Ranks (expected 14-day returns) alongside the stock price for each stock. Additionally, fundamental data is provided to give context on each stock's valuation, dividend information, market capitalization, sector, and industry.
If information on a stock user is enquiring about is not found in any of the provided sections with the query, recommend that the user adds the stock to their Research Portfolio or Runs Simulation to for information on more custom stock preferences.
"""


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# HTML template for the chat interface
html_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Zoltar Chat Assistant</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      .chat-container { max-width: 600px; margin: 0 auto; }
      .chat-message { margin: 10px 0; font-size: 10px; } /* Smaller font size */
      .user { color: blue; }
      .assistant { color: green; }
      .input-container { margin-top: 20px; }
      table { width: 100%; border-collapse: collapse; margin-top: 10px; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
      th { background-color: #4CAF50; color: white; }
      tr:nth-child(even) { background-color: #f9f9f9; }
      tr:hover { background-color: #f1f1f1; }
    </style>
  </head>
  <body>
    <div class="chat-container">
      <h2>Zoltar Chat Assistant | Knowledge is your friend</h2>
      <div id="chat-history">
        {% for message in messages %}
          <div class="chat-message {{ message['role'] }}">
            <strong>{{ message['role'].capitalize() }}:</strong>
            <div>{{ message['content'] | safe }}</div>
          </div>
        {% endfor %}
      </div>
      <div class="input-container">
        <form action="/ask-zoltar" method="post">
          <input type="text" name="prompt" placeholder="Ask Zoltar a question..." required>
          <button type="submit">Send</button>
        </form>
      </div>
    </div>
  </body>
</html>
"""

# Initialize chat history
chat_history = []

def markdown_to_html_table(markdown):
    lines = markdown.strip().split('\n')
    if len(lines) < 3:
        return markdown  # Not enough lines to form a table

    # Extract headers
    headers = lines[0].strip().split('|')[1:-1]
    headers = [header.strip() for header in headers]

    # Extract rows
    rows = []
    for line in lines[2:]:
        row = line.strip().split('|')[1:-1]
        if row:  # Only add if the row is not empty
            rows.append([cell.strip() for cell in row])

    # Build HTML table
    html = '<table><thead><tr>'
    for header in headers:
        html += f'<th>{header}</th>'
    html += '</tr></thead><tbody>'
    
    for row in rows:
        html += '<tr>'
        for cell in row:
            html += f'<td>{cell}</td>'
        html += '</tr>'
    
    html += '</tbody></table>'
    return html

@app.route('/', methods=['GET'])
def home():
    # Render the chat interface with the current chat history
    return render_template_string(html_template, messages=chat_history)

@app.route('/ask-zoltar', methods=['POST'])
def ask_zoltar():

    # # Determine the data directory
    # if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
    #     data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
    # else:
    #     data_dir = '/mount/src/zoltarfinancial/daily_ranks'


    # def get_available_versions(data_dir, selected_dates=None, selected_time_slots=None):
    #     files = os.listdir(data_dir)
    #     versions = set()
    #     for file in files:
    #         if file.startswith(('high_risk_rankings_', 'low_risk_rankings_')):
    #             # Extract the full date and time string
    #             version = '_'.join(file.split('_')[-2:]).split('.')[0]
    #             date_part = version[:8]
    #             time_slot_part = version.split('-')[1] if '-' in version else "FULL OVERNIGHT UPDATE"
                
    #             # Apply filters
    #             if (selected_dates is None or date_part in selected_dates) and \
    #                (selected_time_slots is None or time_slot_part in selected_time_slots):
    #                 versions.add(version)
                    
    #     return sorted(list(versions), reverse=True)  # Sort versions in descending order
    # def load_data2(file_path):
    #     return pd.read_pickle(file_path)
    #         # unique_time_slots = ["FULL OVERNIGHT UPDATE", "PREMARKET UPDATE", "AFTEROPEN UPDATE","MORNING UPDATE","AFTERNOON UPDATE","PRECLOSE UPDATE","AFTERCLOSE UPDATE","WEEKEND UPDATE"]  # Example slots
    # # 9.3.24
    # def convert_to_ranking_format(df, ranking_metric):
    #     # Ensure the 'Date' column is in datetime format
    #     df['Date'] = pd.to_datetime(df['Date'])
        
    #     # Pivot the dataframe to have dates as columns and symbols as rows
    #     pivot_df = df.pivot(index='Symbol', columns='Date', values=ranking_metric)
        
    #     # Reset index to have 'Symbol' as a column
    #     pivot_df.reset_index(inplace=True)
        
    #     return pivot_df    
    
    # # @st.cache_data
    # def select_versions2(num_versions, selected_dates=None, selected_time_slots=None):
    #     if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
    #         data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
    #     else:
    #         data_dir = '/mount/src/zoltarfinancial/daily_ranks'
    
    #     # Get all available versions without filtering
    #     all_versions = get_available_versions(data_dir)
    
    #     # Filter versions based on selected dates and time slots
    #     versions = all_versions
    #     if selected_dates:
    #         versions = [v for v in versions if v[:8] in selected_dates]
    #     if selected_time_slots:
    #         versions = [v for v in versions if (v.split('-')[1] if '-' in v else "FULL OVERNIGHT UPDATE") in selected_time_slots]
        
    #     selected_versions = versions[:num_versions]
    
    #     all_high_risk_dfs = []
    #     all_low_risk_dfs = []
    
    #     for version in selected_versions:
    #         high_risk_file = f"high_risk_rankings_{version}.pkl"
    #         low_risk_file = f"low_risk_rankings_{version}.pkl"
    
    #         high_risk_path = os.path.join(data_dir, high_risk_file)
    #         low_risk_path = os.path.join(data_dir, low_risk_file)
    
    #         if os.path.exists(high_risk_path) and os.path.exists(low_risk_path):
    #             try:
    #                 high_risk_df = load_data2(high_risk_path)
    #                 low_risk_df = load_data2(low_risk_path)
    
    #                 high_risk_df['Version'] = version
    #                 low_risk_df['Version'] = version
    
    #                 all_high_risk_dfs.append(high_risk_df)
    #                 all_low_risk_dfs.append(low_risk_df)
    #             except Exception as e:
    #                 False
    #                 # st.warning(f"Error loading data for version {version}: {str(e)}")
    #         else:
    #             # st.warning(f"Data files for version {version} not found.")
    #             False
    
    #     if not all_high_risk_dfs or not all_low_risk_dfs:
    #         # st.error("No valid data found for the selected versions.")
    #         False
    #         return pd.DataFrame(), pd.DataFrame()
    
    #     return pd.concat(all_high_risk_dfs), pd.concat(all_low_risk_dfs)
    
    # available_versions = get_available_versions(data_dir)
    # default_time_slots = ["FULL OVERNIGHT UPDATE", "WEEKEND UPDATE"]
    # chronological_order = [
    #     "FULL OVERNIGHT UPDATE",
    #     "PREMARKET UPDATE",
    #     "MORNING UPDATE",
    #     "AFTEROPEN UPDATE",
    #     "AFTERNOON UPDATE",
    #     "PRECLOSE UPDATE",
    #     "AFTERCLOSE UPDATE",
    #     "WEEKEND UPDATE"
    # ]

    # # unique_time_slots = high_risk_df_long['Time_Slot'].unique()

    # # Replace NaN values with "FULL OVERNIGHT UPDATE"
    # unique_time_slots = [slot if pd.notna(slot) else "FULL OVERNIGHT UPDATE" for slot in chronological_order]    
    
    # ordered_time_slots = sorted(unique_time_slots, key=lambda x: chronological_order.index(x) if x in chronological_order else len(chronological_order))

    # filtered_versions_intra = [v for v in available_versions if (v.split('-')[1] if '-' in v else "FULL OVERNIGHT UPDATE") in ordered_time_slots]  # replaced default_time_slots to make use of most recent
    # filtered_versions_daily = [v for v in available_versions if (v.split('-')[1] if '-' in v else "FULL OVERNIGHT UPDATE") in default_time_slots]  
    # filtered_versions = filtered_versions_intra[:15]

    # def get_latest_files(data_dir=None):
    #     if data_dir is None:
    #         # Determine the environment and set the appropriate data directory
    #         if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
    #             # Cloud environment
    #             data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
    #         else:
    #             # Local environment
    #             data_dir = '/mount/src/zoltarfinancial/daily_ranks'
    
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
    # latest_files = get_latest_files(data_dir)

    # def load_data(file_path):
    #     return pd.read_pickle(file_path)
    
    # high_risk_df = load_data(os.path.join(data_dir, latest_files['high_risk'])) if latest_files['high_risk'] else None
    # low_risk_df = load_data(os.path.join(data_dir, latest_files['low_risk'])) if latest_files['low_risk'] else None


    # high_risk_rankings = convert_to_ranking_format(high_risk_df, "High_Risk_Score")
    # low_risk_rankings = convert_to_ranking_format(low_risk_df, "Low_Risk_Score")

    # # def get_data_directory():
    # #     if os.path.exists('/mount/src/zoltarfinancial'):
    # #         # Cloud environment
    # #         return '/mount/src/zoltarfinancial/data'
    # #     else:
    # #         # Local environment
    # #         return r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\data'

    # # 9.3.24 - need new one for fundamentals_df 
    # def find_most_recent_file(directory, prefix):
    #     # List all files in the directory
    #     files = os.listdir(directory)
    #     # Filter files that start with the given prefix
    #     files = [f for f in files if f.startswith(prefix)]
    #     # Sort files by modification time in descending order
    #     files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    #     # Return the most recent file
    #     return os.path.join(directory, files[0]) if files else None

    # def get_data_directory():
    #     if os.path.exists('/mount/src/zoltarfinancial'):
    #         # Cloud environment
    #         return '/mount/src/zoltarfinancial/data'
    #     else:
    #         # Local environment
    #         return r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\data'


    # # Load fundamentals data (kept as is from your original code)
    # output_dir_fund = get_data_directory()
    # fundamentals_file_prefix = 'fundamentals_df_'
    # most_recent_fundamentals_file = find_most_recent_file(output_dir_fund, fundamentals_file_prefix)

    # if most_recent_fundamentals_file:
    #     combined_fundamentals_df = pd.read_pickle(most_recent_fundamentals_file)
    #     # st.write(f"Loaded fundamentals data from {most_recent_fundamentals_file}")
    # else:
    #     # st.write("No fundamentals file found.")
    #     False



    # merged_df_low = pd.merge(low_risk_rankings, combined_fundamentals_df, on='Symbol', how='left')
    # merged_df_high = pd.merge(high_risk_rankings, combined_fundamentals_df, on='Symbol', how='left')

    # # Get all date columns
    # date_columns = [col for col in merged_df_high.columns if isinstance(col, pd.Timestamp)]
    # # Get start and end dates from the data
    # start_date = min(high_risk_df['Date'].min(), low_risk_df['Date'].min())
    # full_end_date = max(high_risk_df['Date'].max(), low_risk_df['Date'].max())
    
    # # # Filter date columns based on the selected date range
    # # date_columns = [col for col in date_columns]
    # # Filter date columns based on the selected date range
    # # date_columns = [col for col in date_columns if start_date <= col <= end_date]

    # # if not date_columns:
    # #     st.error(f"No data available for the selected date range for rankings.")
    # #     return
    
    # # # Use the latest date column in the selected range for ranking
    # latest_date = max(date_columns)
    # ranking_column = latest_date
    
    # # Sort the filtered DataFrame
    # sorted_df_low = merged_df_low.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
    # sorted_df_high = merged_df_high.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
    # # Sort by the last column for merged_df_low
    # # sorted_df_low = merged_df_low.sort_values(by=merged_df_low.columns[-1], ascending=False).reset_index(drop=True)
    
    # # Sort by the last column for merged_df_high
    # # sorted_df_high = merged_df_high.sort_values(by=merged_df_high.columns[-1], ascending=False).reset_index(drop=True)
    # # Get the data for selected versions with filters applied
    # high_risk_df_long, low_risk_df_long = select_versions2(15, None, ordered_time_slots) #12.1.24 -  changed from default_time_slots to get most recent, uppped to 15 from 10

    # # Sort both DataFrames by 'Symbol', 'Version', and 'Date' in descending order
    # high_risk_df_long = high_risk_df_long.sort_values(by=['Symbol', 'Version', 'Date'], ascending=[True, True, False])
    # low_risk_df_long = low_risk_df_long.sort_values(by=['Symbol', 'Version', 'Date'], ascending=[True, True, False])
    
    # # Now, select the last record for each combination of 'Symbol' and 'Version' (most recent Date)
    # high_risk_df_long = high_risk_df_long.groupby(['Symbol', 'Version']).first().reset_index()
    # low_risk_df_long = low_risk_df_long.groupby(['Symbol', 'Version']).first().reset_index()
    # # # First, select the rows with the maximum 'Date' for each Symbol and Version
    # # high_risk_df_long = high_risk_df_long.loc[high_risk_df_long.groupby(['Symbol', 'Version'])['Date'].idxmax()]
    # # low_risk_df_long = low_risk_df_long.loc[low_risk_df_long.groupby(['Symbol', 'Version'])['Date'].idxmax()]
    
    # # Now, select the maximum score and Close_Price for each Symbol and Version for high_risk_df
    # high_risk_df_long = high_risk_df_long.groupby(['Symbol', 'Version']).agg({
    #     'High_Risk_Score': 'last',  # Get the maximum High_Risk_Score for each group
    #     'Close_Price': 'last'  # Ensure the Close_Price is from the latest Date by using 'last'
    # }).reset_index()
    
    # # For low_risk_df, get the max Low_Risk_Score and Close_Price from the latest record
    # low_risk_df_long = low_risk_df_long.groupby(['Symbol', 'Version']).agg({
    #     'Low_Risk_Score': 'last',  # Get the maximum Low_Risk_Score for each group
    #     'Close_Price': 'last'  # Ensure the Close_Price is from the latest Date by using 'last'
    # }).reset_index()
    
    # # Sort the dataframes by Version in descending order
    # high_risk_df_long = high_risk_df_long.sort_values('Version', ascending=False)
    # low_risk_df_long = low_risk_df_long.sort_values('Version', ascending=False)            
    # # Sort the dataframes by Version in descending order
    # high_risk_df_long = high_risk_df_long.sort_values('Version', ascending=False)
    # low_risk_df_long = low_risk_df_long.sort_values('Version', ascending=False)

    # # Create new columns for Date and Time Slot
    # high_risk_df_long['Date'] = high_risk_df_long['Version'].str[:8]
    # high_risk_df_long['Time_Slot'] = high_risk_df_long['Version'].str.split('-').str[1]
    
    # low_risk_df_long['Date'] = low_risk_df_long['Version'].str[:8]
    # low_risk_df_long['Time_Slot'] = low_risk_df_long['Version'].str.split('-').str[1]

    # # Create filters for Date and Time Slot
    # unique_dates = high_risk_df_long['Date'].unique()
    # unique_time_slots = high_risk_df_long['Time_Slot'].unique()

    # # Replace NaN values with "FULL OVERNIGHT UPDATE"
    # unique_time_slots = [slot if pd.notna(slot) else "FULL OVERNIGHT UPDATE" for slot in unique_time_slots]    
    # # Use top_x to limit the number of stocks displayed - selected to do top 20 (not top_x as it was before
    # display_df_low = sorted_df_low.head(50)
    # # display_df_low_all = sorted_df_low.head(1200)
    # print(display_df_low)
    # display_df_high = sorted_df_high.head(5)
    # unique_dates = sorted(set(version[:8] for version in filtered_versions), reverse=True)
    # # Extract unique time slots from available versions
    # unique_time_slots = sorted(set(version.split('-')[1] if '-' in version else "FULL OVERNIGHT UPDATE" for version in available_versions))
    
    # # Multi-select for stocks
    # # default_stocks_low_all = sorted_df_low['Symbol'].tolist()
    # default_stocks_low = display_df_low['Symbol'].tolist()
    # default_stocks_high = display_df_high['Symbol'].tolist()
    #     # selected_stocks = st.multiselect(
    #     #     f"Select stocks to display ({ranking_type})",
    #     #     options=sorted_df['Symbol'].tolist(),
    #     #     default=default_stocks,
    #     #     key=f"{ranking_type}_stock_multiselect"
    #     # )




    
    # # Filter for custom stocks and get the latest date for each stock
    # # custom_df_low = sorted_df_low[sorted_df_low['Symbol'].isin(default_stocks_low)]
    # #12.1.24 -  trying to do all 
    # custom_df_low = sorted_df_low[sorted_df_low['Symbol'].isin(default_stocks_low)]


    # # custom_df_low = custom_df_low.sort_values('Date').groupby('Symbol').last().reset_index()
    
    # # Handle None values
    # custom_df_low['Fundamentals_Sector'] = custom_df_low['Fundamentals_Sector'].fillna('Unknown Sector')
    # custom_df_low['Fundamentals_Industry'] = custom_df_low['Fundamentals_Industry'].fillna('Unknown Industry')
        
    # # Filter for custom stocks and get the latest date for each stock
    # custom_df_high = merged_df_high[merged_df_high['Symbol'].isin(default_stocks_high)]
    # # custom_df_high = custom_df_high.sort_values('Date').groupby('Symbol').last().reset_index()
    
    # # Handle None values
    # custom_df_high['Fundamentals_Sector'] = custom_df_high['Fundamentals_Sector'].fillna('Unknown Sector')
    # custom_df_high['Fundamentals_Industry'] = custom_df_high['Fundamentals_Industry'].fillna('Unknown Industry')



    
    # # 11.24.24 PLACEHOLDER SECTION TO LOAD ALL TOP RANKS TO BE ABLE TO ANSWER ANY QUESTIONS ABOUT THEM IMMEDIATELY        
    # def generate_stock_data(custom_stocks, high_risk_df_long, low_risk_df_long):
    #     stock_data = []
    #     for stock in custom_stocks:
    #         stock_data.append(f"\n{stock}:")
    #         stock_data.append("| Version | Date | Time Slot | High Zoltar Rank | Low Zoltar Rank | Close Price | High Zoltar Rank Index to Avg | Low Zoltar Rank Index to Avg |")
    #         stock_data.append("|---------|------|-----------|-----------------|----------------|-------------|------------------------|------------------------|")
            
    #         high_risk_stock = high_risk_df_long[high_risk_df_long['Symbol'] == stock]
    #         low_risk_stock = low_risk_df_long[low_risk_df_long['Symbol'] == stock]
            
    #         # 11.24.24 - correct for negative  values
    #         # Calculate shifts for both High and Low Risk Scores
    #         shift_high = abs(min(high_risk_stock['High_Risk_Score'].min(), 0))
    #         shift_low = abs(min(low_risk_stock['Low_Risk_Score'].min(), 0))
            
    #         # Calculate averages with shift
    #         avg_high_score = (high_risk_stock['High_Risk_Score'] + shift_high).mean()
    #         avg_low_score = (low_risk_stock['Low_Risk_Score'] + shift_low).mean()
            
    #         for _, row in high_risk_stock.iterrows():
    #             low_risk_row = low_risk_stock[low_risk_stock['Version'] == row['Version']].iloc[0]
                
    #             # Calculate indices with shift
    #             high_risk_index = (row['High_Risk_Score'] + shift_high) / avg_high_score
    #             low_risk_index = (low_risk_row['Low_Risk_Score'] + shift_low) / avg_low_score
                
    #             # Calculate real scores
    #             high_risk_score_real = row['High_Risk_Score'] * 100
    #             low_risk_score_real = low_risk_row['Low_Risk_Score'] * 100

    #         # for _, row in high_risk_stock.iterrows():
    #         #     low_risk_row = low_risk_stock[low_risk_stock['Version'] == row['Version']].iloc[0]
    #         #     high_risk_index = row['High_Risk_Score'] / high_risk_stock['High_Risk_Score'].mean()
    #         #     low_risk_index = low_risk_row['Low_Risk_Score'] / low_risk_stock['Low_Risk_Score'].mean()
                
    #             stock_data.append(f"| {row['Version']} | {row['Date']} | {row['Time_Slot']} | {row['High_Risk_Score']*100:.2f}% | {low_risk_row['Low_Risk_Score']*100:.2f}% | ${row['Close_Price']:.2f} | {high_risk_index:.2f} | {low_risk_index:.2f} |")
            
    #         # Calculate and add averages
    #         avg_high_risk = high_risk_stock['High_Risk_Score'].mean() * 100
    #         avg_low_risk = low_risk_stock['Low_Risk_Score'].mean() * 100
    #         avg_close_price = high_risk_stock['Close_Price'].mean()
    #         stock_data.append(f"\nAverages: High Zoltar Rank: {avg_high_risk:.2f}%, Low Zoltar Rank: {avg_low_risk:.2f}%, Close Price: ${avg_close_price:.2f}")
            
    #         # Add trend information
    #         high_risk_trend = "increasing" if high_risk_stock['High_Risk_Score'].iloc[0] > high_risk_stock['High_Risk_Score'].iloc[-1] else "decreasing"
    #         low_risk_trend = "increasing" if low_risk_stock['Low_Risk_Score'].iloc[0] > low_risk_stock['Low_Risk_Score'].iloc[-1] else "decreasing"
    #         price_trend = "increasing" if high_risk_stock['Close_Price'].iloc[0] > high_risk_stock['Close_Price'].iloc[-1] else "decreasing"
    #         stock_data.append(f"Trends: High Risk Score: {high_risk_trend}, Low Risk Score: {low_risk_trend}, Price: {price_trend}")
        
    #     return "\n".join(stock_data)
    # # def generate_fundamentals_data(custom_df):
    # #     fundamentals_data = []
    # #     fundamentals_data.append("| Symbol | PE | PB | Dividends | Ex-Dividend Date | Market Cap | Sector | Industry | Best Hold Period (days) |")
    # #     fundamentals_data.append("|--------|----|----|-----------|-------------------|------------|--------|----------|------------------------------|")
        
    # #     for _, row in custom_df.iterrows():
    # #         fundamentals_data.append(f"| {row['Symbol']} | {row['Fundamentals_PE']:.2f} | {row['Fundamentals_PB']:.2f} | {row['Fundamentals_Dividends']:.2f} | {row['Fundamentals_ExDividendDate']} | {row['Fundamentals_MarketCap']:,.0f} | {row['Fundamentals_Sector']} | {row['Fundamentals_Industry']} | {row['High_Risk_Score_HoldPeriod']} |")   #{row['High_Risk_Score_HoldPeriod']}
    # def generate_fundamentals_data(custom_df):
    #     fundamentals_data = []
    #     fundamentals_data.append("| Symbol | PE | PB | Dividends | Ex-Dividend Date | Market Cap | Sector | Industry |")
    #     fundamentals_data.append("|--------|----|----|-----------|-------------------|------------|--------|----------|")
        
    #     for _, row in custom_df.iterrows():
    #         fundamentals_data.append(f"| {row['Symbol']} | {row['Fundamentals_PE']:.2f} | {row['Fundamentals_PB']:.2f} | {row['Fundamentals_Dividends']:.2f} | {row['Fundamentals_ExDividendDate']} | {row['Fundamentals_MarketCap']:,.0f} | {row['Fundamentals_Sector']} | {row['Fundamentals_Industry']} |")   #{row['High_Risk_Score_HoldPeriod']}
        
    #     return "\n".join(fundamentals_data)
    # def generate_fundamentals_data_l(custom_df):
    #     fundamentals_data = []
    #     fundamentals_data.append("| Symbol | PE | PB | Dividends | Ex-Dividend Date | Market Cap | Sector | Industry |")
    #     fundamentals_data.append("|--------|----|----|-----------|-------------------|------------|--------|----------|")
        
    #     for _, row in custom_df.iterrows():
    #         fundamentals_data.append(f"| {row['Symbol']} | {row['Fundamentals_PE']:.2f} | {row['Fundamentals_PB']:.2f} | {row['Fundamentals_Dividends']:.2f} | {row['Fundamentals_ExDividendDate']} | {row['Fundamentals_MarketCap']:,.0f} | {row['Fundamentals_Sector']} | {row['Fundamentals_Industry']} |")
        
    #     return "\n".join(fundamentals_data)
    # pre_prompt_low = f"""
    # The data below represents the top ranked stocks for the most recent data point using Low Zoltar Ranks that predict average expected returns from buying stock now and selling over the next 14 days; also included are corresponding stock prices for {len(default_stocks_low)} stocks: {', '.join(default_stocks_low)}.
    # The user may or may not be familiar with these stocks, and the stocks on this list should always be correlated against the user's portfolio section, if it exists.
    # If a stock the user is asking about is not on the list, recommend that the user adds the stock to their Research Portfolio, or runs the Simulation to reveal more stocks.
    # The user is particularly interested in finding undervalued stocks through looking for 1) the highest High and Low Zoltar Rank for the most recent data point, 2) with highest (and non-negative) average low Zoltar Ranks, 3) with higher index to average (also non-negative), and 3) preferably at a lower price than in prior data points for that stock.
    # Make sure that the final answer looks at the historical trends and addresses the user interest. If user is interested in high returns, then they are interested in highest High Zoltar Rank, if user is interested in consistent performance, then the user is interested in highest average Low Zoltar Rank; and together with those a higher index to average for the current data point, combined with deflated price for most recent data point could signal an undervalued stock.
    # When user is interested in diversification, they want the top Zoltar Ranks from multiple sectors.
    # When user wants to select stocks to improve their portfolio, this is the list to use to recommend stocks from - but don't mix it with their existing portfolio.  The stocks in this section aim for more stability in return prediction.
    # Together with data in this section, additional section with similar organization shows the user's current research portfolio, and stocks on this list could be recommended to replace some of the stocks in this portfolio expected to perform worse, especially in the same industries and sectors.
    
    # The data covers {len(unique_dates)} dates from {min(unique_dates)} to {max(unique_dates)}, with time slots: {', '.join(unique_time_slots)}.
    
    # Data for each stock:
    # {generate_stock_data(default_stocks_low, high_risk_df_long, low_risk_df_long)}
    
    # Fundamentals data for each stock:
    # {generate_fundamentals_data_l(custom_df_low)}
    
    # Historical ranges across all stocks:
    # - High Zoltar Rank: {high_risk_df_long['High_Risk_Score'].min()*100:.2f}% to {high_risk_df_long['High_Risk_Score'].max()*100:.2f}%
    # - Low Zoltar Rank: {low_risk_df_long['Low_Risk_Score'].min()*100:.2f}% to {low_risk_df_long['Low_Risk_Score'].max()*100:.2f}%
    # - Close Price: ${high_risk_df_long['Close_Price'].min():.2f} to ${high_risk_df_long['Close_Price'].max():.2f}
    
    # For each stock, we calculate:
    # 1. Average of expected returns in prior versions
    # 2. Current expected return
    # 3. Index to average expected returns (current / average)
    
    # Based on these calculations, we provide indicators:
    # - Strong Buy: If average Low Zoltar Rank >= 70bps and Index to Avg > 1.3, or if average Low Zoltar Rank >= 0bps and Index to Avg > 1.5
    # - Hold & Trim: If average Low Zoltar Rank >= 70bps and Index to Avg <= 1.3, or if 0bps < average Low Zoltar Rank < 70bps and Index to Avg > 1
    # - Moderate Sell: If 0bps <= last Low Zoltar Rank < 70bps and Index to Avg <= 1
    # - Strong Sell: If last Low Risk Score <= 0bps and index to Avg <= 1
    # - Promising: For other cases
    
    # The data shows the historical trend of High and Low Zoltar Ranks (expected 14-day returns) alongside the stock price for each stock. Additionally, fundamental data is provided to give context on each stock's valuation, dividend information, market capitalization, sector, and industry.
    # If information on a stock user is enquiring about is not found in any of the provided sections with the query, recommend that the user adds the stock to their Research Portfolio or Runs Simulation to for information on more custom stock preferences.
    # """
    
    # Define the pre_prompt_about variable

# 2.16.25 - og is below:
    pre_prompt_about = """
    Use this section to answer questions user may have on the company and methodology
    Founder and CEO of the company is Andrew N. Podosenov. A little from the CEO: For over 20 years, my passion has been to use Computational Math and Statistics to uncover knowledge about cause-effect relationships and use derived solutions to capitalize on this knowledge. When passion meets expertise, magic happens!
    What we are about:
    We created a self-service, AI-assisted/maintained software platform (web, mobile and desktop) that educates users on current stock market trends and enables more informed trading decisions. 
    We achieve outstanding results through our successful deployment of advanced analytical techniques from data and behavioral science, time series, machine learning and optimization to produce features, define objective functions, and systemically produce unbiased Zoltar Ranks.  These highly predictive and timely solutions, together with our simulation software, research toolkit, and an uber-helpful Zoltar AI Chat Assistant that has up-to-date knowledge, enable users to test execution levers, fine-tune trading strategies, and generate and share own custom curated BUY(and SELL) lists.
    With each daily iteration of Zoltar Model Suite we generate over 500 sub-models and score on live data every 30 minutes to empower Zoltar community with timely and reliable intraday prediction trends.  Additionally, our platform provides trend analysis of all prior Zoltar Rank versions for advanced Ensemble Modeling capability directly to the users.
    Mission statement:
    We strive to have our platform users form trading strategies that are uniquely theirs.  Our mission is to ensure they consistently outperform  S&P 500. 
    We use the power of advanced analytics to derive Zoltar Ranks, that together with strategy levers, simulation engine and the uber-helpful Zoltar Chat assistant, make it possible.
    Zoltar Financial Methodology:
    1. **Target Definition**: Clearly define the investment targets and objectives.
    2. **Sector and Industry Level Modeling and Feature Engineering**: Develop models at the sector and industry levels, incorporating advanced feature engineering techniques.
    3. **Segmentation**: Segment data to identify distinct market segments and tailor strategies accordingly.
    4. **Transparent, Repeatable Binning and Other Transformations**: Apply transparent and repeatable transformations to data for consistency and reliability.
    5. **A Suite of Machine Learning Algorithms**: Utilize a diverse set of machine learning algorithms to analyze data and predict market trends.
    6. **Optimization and Tuning of Portfolio**: Optimize portfolios using models that cater to varying levels of Zoltar Users' risk tolerance criteria.
    7. **Strategy Training and Validation**: Provide tools for Zoltar Users to customize, share, and validate their strategies, fostering a collaborative environment.
    8. **Live Daily Trading on Zoltar Corp**: Execute the leader strategy live daily, showcasing the strength of the Zoltar community and marking the start of ZF blockchain integration.
    """
    
    
    # data = request.json
    # prompt = data.get('prompt', '')  # Use request.json to access JSON data

    # # Add user message to chat history
    # chat_history.append({"role": "user", "content": prompt})

    # messages=[
    #     {"role": "system", "content": "You are a helpful assistant for a stock trading application named Zoltar. Provide a short summary, ALWAYS followed by a table with supporting details, and only use special characters like ':','<', '>', '|' or '&' in the response only for Table markdown, and never use ':' or '''; and conclude with 'May the riches be with you...'"},
    #     # {"role": "user", "content": prompt}
    # ]

    # # if 'pre_prompt_low' in locals() or 'pre_prompt_low' in globals():
    # #     messages.append({"role": "user", "content": pre_prompt_low})        

    # # Append pre_prompt_about to messages if it exists
    # if 'pre_prompt_about' in locals() or 'pre_prompt_about' in globals():
    #     messages.append({"role": "user", "content": pre_prompt_about})

    # messages.append({"role": "user", "content": prompt})


    # # Query OpenAI API
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o-mini",  # Ensure you are using the correct model
    #     messages=messages
    # )

    # # Extract the response text
    # response_text = response.choices[0].message['content']

    # # Split the response into summary, table, and closing phrase
    # parts = response_text.split('\n\n', 2)
    # summary = parts[0]
    # table_markdown = parts[1] if len(parts) > 1 else ''
    # closing_phrase = parts[2] if len(parts) > 2 else "May the riches be with you..."

    # # Convert markdown table to HTML table
    # table_html = markdown_to_html_table(table_markdown)

    # # Combine the parts into the final formatted response
    # formatted_response = f"<div>{summary}</div>{table_html}<div>{closing_phrase}</div>"

    # # Add assistant response to chat history
    # chat_history.append({"role": "assistant", "content": formatted_response})

    # # Return JSON response
    # return jsonify({"answer": formatted_response})
# 2.16.25 - new section
# def ask_zoltar():
    data = request.json
    prompt = data.get('prompt', '')

    chat_history.append({"role": "user", "content": prompt})

    messages = [
        {"role": "system", "content": "You are a helpful assistant for a stock trading application named Zoltar. Provide a short summary, ALWAYS followed by a table with supporting details, and only use special characters like ':','<', '>', '|' or '&' in the response only for Table markdown, and never use ':' or '''; and conclude with 'May the riches be with you...'"},
    ]

    if 'pre_prompt_low' in locals() or 'pre_prompt_low' in globals():
        messages.append({"role": "user", "content": pre_prompt_low})        

    if 'pre_prompt_about' in locals() or 'pre_prompt_about' in globals():
        messages.append({"role": "user", "content": pre_prompt_about})

    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages
    )

    response_text = response.choices[0].message['content']

    parts = response_text.split('\n\n', 2)
    summary = parts[0]
    table_markdown = parts[1] if len(parts) > 1 else ''
    closing_phrase = parts[2] if len(parts) > 2 else "May the riches be with you..."

    table_html = markdown_to_html_table(table_markdown)

    formatted_response = f"<div>{summary}</div>{table_html}<div>{closing_phrase}</div>"

    chat_history.append({"role": "assistant", "content": formatted_response})

    return jsonify({"answer": formatted_response})
if __name__ == '__main__':
    app.run(debug=True)