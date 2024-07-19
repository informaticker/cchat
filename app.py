import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import AutoTokenizer
import requests
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[os.getenv('RATE_LIMIT', '15 per minute')]
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.getenv('TOKENIZER', 'gpt2'))

# API configuration
API_URL = os.getenv('API_URL', 'https://api.openai.com/v1')
API_KEY = os.getenv('API_KEY')
API_MODEL = os.getenv('API_MODEL', 'gpt-3.5-turbo')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0))

logger.info(f"Chat initialized using endpoint: {API_URL}, model: {API_MODEL}, temperature: {TEMPERATURE}")

@app.route('/v1/tokenizer/count', methods=['POST'])
def token_count():
    try:
        data = request.json
        messages = data.get('messages', [])
        full_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        tokens = tokenizer.encode(full_text)
        return jsonify({"token_count": len(tokens)})
    except Exception as e:
        logger.error(f"Error in token_count: {str(e)}")
        return jsonify({"error": "Invalid request"}), 400

@app.route('/v1/chat/completions', methods=['POST'])
@limiter.limit(os.getenv('RATE_LIMIT', '15/minute'))
def proxy_chat_completions():
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        request_data = request.json
        request_data['model'] = API_MODEL
        request_data['temperature'] = TEMPERATURE
        request_data['stream'] = True

        response = requests.post(f"{API_URL}/chat/completions",
                                 json=request_data,
                                 headers=headers,
                                 stream=True)

        response.raise_for_status()

        def generate():
            for chunk in response.iter_content(chunk_size=8):
                if chunk:
                    yield chunk

        return Response(stream_with_context(generate()),
                        content_type=response.headers['content-type'])

    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return jsonify({"error": "Failed to connect to the API"}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

if __name__ == '__main__':
    app.run(debug=False, port=int(os.getenv('PORT', 5000)))
