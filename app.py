from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from transformers import AutoTokenizer
import os
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.environ.get('TOKENIZER', 'gpt2'))

api_url = os.environ.get('API_URL', 'https://api.openai.com/v1')
api_key = os.environ.get('API_KEY')
api_model = os.environ.get('API_MODEL', 'gpt-3.5-turbo')
temperature = os.environ.get('TEMPERATURE', 0)

@app.route('/v1/tokenizer/count', methods=['POST'])
def token_count():
    data = request.json
    messages = data.get('messages', [])
    full_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    tokens = tokenizer.encode(full_text)
    token_count = len(tokens)
    return jsonify({"token_count": token_count})

@app.route('/v1/chat/completions', methods=['POST'])
@limiter.limit(os.environ.get('RATE_LIMIT', '20/minute'))
def proxy_chat_completions():
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    request_data = request.json

    request_data['model'] = api_model
    request_data['temperature'] = temperature

    request_data['stream'] = True

    response = requests.post(f"{api_url}/chat/completions",
                             json=request_data,
                             headers=headers,
                             stream=True)

    # Stream the response back to the client
    def generate():
        for chunk in response.iter_content(chunk_size=8):
            if chunk:
                yield chunk

    return Response(stream_with_context(generate()),
                    content_type=response.headers['content-type'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
