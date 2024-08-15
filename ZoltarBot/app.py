from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# Load OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/ask-zoltar', methods=['POST'])
def ask_zoltar():
    data = request.json
    question = data.get('question', '')

    # Query OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a stock trading application named Zoltar."},
            {"role": "user", "content": question}
        ]
    )

    response_text = response.choices[0].message['content']
    return jsonify({'answer': response_text})

if __name__ == '__main__':
    app.run(debug=True)