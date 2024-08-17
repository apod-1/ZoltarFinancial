from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import openai
import os

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
      .chat-message { margin: 10px 0; }
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a stock trading application named Zoltar that prepares responses as a short summary followed by more details in table format for almost all requests and finishes response with May the riches be with you..."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the response text
    response_text = response.choices[0].message['content']

    # Convert markdown table to HTML table
    formatted_response = markdown_to_html_table(response_text)

    # Add assistant response to chat history
    chat_history.append({"role": "assistant", "content": formatted_response})

    # Return JSON response
    return jsonify({"answer": formatted_response})

if __name__ == '__main__':
    app.run(debug=True)