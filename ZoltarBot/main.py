# activate myflaskenv
# C:\Users\apod7\StockPicker\app\ZoltarFinancial\ZoltarBot>git add .
# git commit -m "main.py change - summary and riches phrase"
# git push -f heroku main

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import openai
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')


# 12.1.24 - add pertinent info from the app here.
# Define the pre-prompts
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
1. Target Definition: Clearly define the investment targets and objectives.
2. Sector and Industry Level Modeling and Feature Engineering: Develop models at the sector and industry levels, incorporating advanced feature engineering techniques.
3. Segmentation: Segment data to identify distinct market segments and tailor strategies accordingly.
4. Transparent, Repeatable Binning and Other Transformations: Apply transparent and repeatable transformations to data for consistency and reliability.
5. A Suite of Machine Learning Algorithms: Utilize a diverse set of machine learning algorithms to analyze data and predict market trends.
6. Optimization and Tuning of Portfolio: Optimize portfolios using models that cater to varying levels of Zoltar Users' risk tolerance criteria.
7. Strategy Training and Validation: Provide tools for Zoltar Users to customize, share, and validate their strategies, fostering a collaborative environment.
8. Live Daily Trading on Zoltar Corp: Execute the leader strategy live daily, showcasing the strength of the Zoltar community and marking the start of ZF blockchain integration.
"""

def get_pre_prompt_low():
    if os.path.exists(r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'):
        data_dir = r'C:\Users\apod7\StockPicker\app\ZoltarFinancial\daily_ranks'
    else:
        data_dir = '/mount/src/zoltarfinancial/daily_ranks'
    
    available_versions = [f for f in os.listdir(data_dir) if f.startswith('low_risk_rankings_')]
    latest_version = max(available_versions)
    
    low_risk_df = pd.read_pickle(os.path.join(data_dir, latest_version))
    sorted_df = low_risk_df.sort_values(by='Low_Risk_Score', ascending=False)
    top_5_stocks = sorted_df.head(5)['Symbol'].tolist()
    
    pre_prompt_low = f"The top 5 low risk stocks based on the latest data are: {', '.join(top_5_stocks)}"
    return pre_prompt_low

@app.route('/ask-zoltar', methods=['POST'])
def ask_zoltar():
    data = request.json
    prompt = data.get('prompt', '')

    chat_history.append({"role": "user", "content": prompt})

    pre_prompt_low = get_pre_prompt_low()

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a stock trading application named Zoltar. Provide a short summary, ALWAYS followed by a table with supporting details, and only use special characters like ':','<', '>', '|' or '&' in the response only for Table markdown, and never use ':' or '''; and conclude with 'May the riches be with you...'"},
            {"role": "user", "content": pre_prompt_about},
            {"role": "user", "content": pre_prompt_low},
            {"role": "user", "content": prompt}
        ]
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
    data = request.json
    prompt = data.get('prompt', '')  # Use request.json to access JSON data

    # Add user message to chat history
    chat_history.append({"role": "user", "content": prompt})

    # Query OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Ensure you are using the correct model
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a stock trading application named Zoltar. Provide a short summary, ALWAYS followed by a table with supporting details, and only use special characters like ':','<', '>', '|' or '&' in the response only for Table markdown, and never use ':' or '''; and conclude with 'May the riches be with you...'"},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the response text
    response_text = response.choices[0].message['content']

    # Split the response into summary, table, and closing phrase
    parts = response_text.split('\n\n', 2)
    summary = parts[0]
    table_markdown = parts[1] if len(parts) > 1 else ''
    closing_phrase = parts[2] if len(parts) > 2 else "May the riches be with you..."

    # Convert markdown table to HTML table
    table_html = markdown_to_html_table(table_markdown)

    # Combine the parts into the final formatted response
    formatted_response = f"<div>{summary}</div>{table_html}<div>{closing_phrase}</div>"

    # Add assistant response to chat history
    chat_history.append({"role": "assistant", "content": formatted_response})

    # Return JSON response
    return jsonify({"answer": formatted_response})

if __name__ == '__main__':
    app.run(debug=True)