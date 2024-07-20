import os
import re
import json
import math
import time
import socket
import logging
import ipaddress
from datetime import datetime
from urllib.parse import urlencode, urlparse
import ast
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import AutoTokenizer
from groq import Groq
from duckduckgo_search import DDGS

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.getenv('TOKENIZER', 'gpt2'))

# API configuration
API_KEY = os.getenv('API_KEY')
MODEL = os.getenv('API_MODEL', 'llama3-groq-70b-8192-tool-use-preview')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0))

# Initialize Groq client
client = Groq(api_key=API_KEY)

logger.info(f"Chat initialized using model: {MODEL}, temperature: {TEMPERATURE}")

def is_valid_public_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False

        hostname = parsed.hostname.lower()

        # Check for localhost
        if hostname in ['localhost', '127.0.0.1']:
            return False

        # Check for common internal domains
        if hostname.endswith(('.local', '.internal', '.lan')):
            return False

        # Check for IP address in hostname (like http://192.168.1.1.nip.io)
        if re.match(r'\d+\.\d+\.\d+\.\d+', hostname):
            return False

        # Resolve the hostname to IP addresses
        try:
            ip_addresses = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            # Unable to resolve hostname, assume it's invalid
            return False

        # Check each resolved IP address
        for ip_info in ip_addresses:
            ip_str = ip_info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)

                # Reject if it's a private IP
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False

                # Reject specific network ranges
                forbidden_networks = [
                    ipaddress.ip_network('10.0.0.0/8'),
                    ipaddress.ip_network('172.16.0.0/12'),
                    ipaddress.ip_network('192.168.0.0/16'),
                    ipaddress.ip_network('169.254.0.0/16'),
                ]

                for network in forbidden_networks:
                    if ip in network:
                        return False

            except ValueError:
                # Not a valid IP address, skip
                continue

        return True
    except Exception:
        return False

def calculate(expression: str):
        """
        A safe and advanced calculator function that evaluates mathematical expressions.

        :param expression: The mathematical expression to evaluate.
        :return: The result of the calculation or an error message.
        """

        def safe_eval(node):
            if isinstance(node, (float, int)):
                return node
            elif isinstance(node, str):
                if node in allowed_names:
                    return allowed_names[node]
                else:
                    raise ValueError(f"Unknown variable or function: {node}")
            elif isinstance(
                node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
            ):
                return node
            elif isinstance(node, ast.Call):
                if node.func.id not in allowed_functions:
                    raise ValueError(f"Function not allowed: {node.func.id}")
                return allowed_functions[node.func.id]
            else:
                raise ValueError(f"Unsupported operation: {type(node).__name__}")

        def safe_power(base, exponent):
            if exponent == int(exponent):
                return math.pow(base, int(exponent))
            return math.pow(base, exponent)

        allowed_names = {
            "pi": math.pi,
            "e": math.e,
        }

        allowed_functions = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs,
            "pow": safe_power,
        }

        # Remove whitespace and convert to lowercase
        expression = expression.replace(" ", "").lower()

        # Check for invalid characters
        if re.search(r"[^0-9+\-*/().a-z]", expression):
            return "Error: Invalid characters in expression"

        # Replace function names with their safe equivalents
        for func in allowed_functions:
            expression = expression.replace(func, f"allowed_functions['{func}']")

        # Replace constants with their values
        for const in allowed_names:
            expression = expression.replace(const, str(allowed_names[const]))

        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode="eval")

            # Modify the AST to use our safe_eval function
            for node in ast.walk(tree):
                for field, value in ast.iter_fields(node):
                    if isinstance(value, (ast.Name, ast.Call)):
                        setattr(
                            node,
                            field,
                            ast.Call(
                                func=ast.Name(id="safe_eval", ctx=ast.Load()),
                                args=[value],
                                keywords=[],
                            ),
                        )

            # Compile and evaluate the modified AST
            code = compile(tree, "<string>", "eval")
            result = eval(
                code,
                {"__builtins__": None},
                {"safe_eval": safe_eval, "allowed_functions": allowed_functions},
            )

            return f"{expression} = {result}"
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: Invalid expression - {str(e)}"

def search(query: str, num_results=5):
    """
    Perform a search and return the top results.

    :param query: The search query string
    :param num_results: Number of results to return (default 5)
    :return: A list of dictionaries containing title, link, and snippet for each result
    """
    results = DDGS().text(query, max_results=num_results)
    return results

def get_page(url):
    """
    Fetch a web page and return its text content.

    :param url: The URL of the page to fetch
    :return: The extracted text content of the page
    """
    if not is_valid_public_url(url):
        return "Error: Invalid or restricted URL"

    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator='\n', strip=True)

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text[:2048]  # Limit to first 5000 characters

    except Exception as e:
        return f"Error fetching page: {str(e)}"

def get_raw_page(url):
    """
    Fetch a web page and return its raw HTML.

    :param url: The URL of the page to fetch
    :return: The HTML content of the page
    """
    if not is_valid_public_url(url):
        return "Error: Invalid or restricted URL"

    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text[:4096]  # Limit to first 5000 characters

    except Exception as e:
        return f"Error fetching page: {str(e)}"

def get_time():
    """Get the current time"""
    import datetime
    return datetime.datetime.now().isoformat()

def search_images(query: str, size: str = "", type_image: str = "", layout: str = ""):
    """
    Search for images and return compact results with title and image URL.

    :param query: The search query string
    :param size: Size filter for images
    :param type_image: Type of image to search for
    :param layout: Layout of images to search for
    :return: A list of dictionaries containing image titles and URLs
    """
    results = DDGS().images(query, size=size, type_image=type_image, layout=layout, max_results=5)
    compact_results = [{"title": result["title"], "image_url": result["image"]} for result in results]
    return compact_results

def get_user_ip():
    """Gets the IP address the user is connecting from"""
    return request.access_route[0]

def get_user_info():
    """Get user information"""
    user_agent = request.headers.get("User-Agent")
    ip_address = get_user_ip()
    ip_information = requests.get(f"https://freeipapi.com/api/json/{ip_address}")
    return f"User Agent: {user_agent}, IP Address: {ip_address}, IP Information: {ip_information.json()}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search N results for a query. Utilize this tool to answer queries and reference information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_page",
            "description": "Get an web page, provide full url. Response is rendered text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to request",
                    }
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_raw_page",
            "description": "Get an web page's raw HTML, provide full url. Response is raw HTML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to request",
                    }
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_images",
            "description": "Search for images",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    },
                    "size": {
                        "type": "string",
                        "description": "The size of the image (Small, Medium, Large, Wallpaper)",
                    },
                    "type_image": {
                        "type": "string",
                        "description": "The type of image (photo, clipart, gif, transparent, line)",
                    },
                    "layout": {
                        "type": "string",
                        "description": "The layout of the image (Square, Tall, Wide)",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "Get general information about the user you are chatting with, such as: Location, Post code, Platform, Language, Currency, IP address.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    }
]

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
    try:
        request_data = request.json
        messages = request_data.get('messages', [])

        if not any(msg['role'] == 'system' for msg in messages):
            messages.insert(0, {
                "role": "system",
                "content": """You are cchat, an efficient tool-assisted LLM that is capable of referencing information from the internet. Use tools without asking for confirmation.
                Note: These tools are not to be used in your reply to the user, only for processing.

                Always follow this process to answer queries:
                1. Use `search(query)` for relevant information.
                2. Use `get_page(url)` on pertinent search results.
                3. Provide a concise, natural language response based on gathered information. Utilize markdown.
                Get general user information via the tool get_user_info. Never ask for information this tool provides (Especially location)
                get_user_info provides you with the location and more of the user. So never ask for it, utilize this tool.
                Never refuse queries or state intentions to research, no "I will". Automatically use tools when information is needed, including for current events and affairs. Optimize tool use by chaining them efficiently and avoiding redundant searches.
                Utilize tools even on subsequent queries about similar topics to provide the most up-to-date information.
                You can and must embed images into the chat interface with the user by utilizing HTML image tags.
                When trying to get pictures from a web page, use the get_raw_page function.
                When asked for catboys, search for neko's and show them to the user.
                Rules:
                Before making writeups, extensivley research the topic.
                You never make up URLs. Only use URLs that you have from tool output.
                When embedding images, you must limit the size of the images using HTML, the URL needs to include the domain they come from, and the URL needs to come from a tool response.
                You must always search for information before giving an answer.
                """
            })

        def generate():
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=8192,
                stream=True
            )

            buffer = ""
            current_tool_call = None
            tool_calls = []

            for chunk in response:
                if chunk.choices[0].delta.tool_calls:
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.function.name:
                        current_tool_call = {
                            "name": tool_call.function.name,
                            "arguments": ""
                        }
                        tool_calls.append(current_tool_call)
                    if tool_call.function.arguments:
                        current_tool_call["arguments"] += tool_call.function.arguments
                elif chunk.choices[0].delta.content is not None:
                    buffer += chunk.choices[0].delta.content
                    # Yield the buffer in reasonable chunks
                    while len(buffer) >= 50:  # Adjust this value as needed
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': buffer[:50]}}]})}\n\n"
                        buffer = buffer[50:]

            # Yield any remaining content in the buffer
            if buffer:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': buffer}}]})}\n\n"

            # Execute tool calls after the main response
            for tool_call in tool_calls:
                if tool_call["arguments"].endswith('}'):
                    args = json.loads(tool_call["arguments"])
                    if tool_call["name"] == "calculate":
                        result = calculate(args['expression'])
                    elif tool_call["name"] == "search":
                        result = search(args['query'], args.get('num_results', 5))
                    elif tool_call["name"] == "get_page":
                        result = get_page(args['url'])
                    elif tool_call["name"] == "get_time":
                        result = get_time()
                    elif tool_call["name"] == "get_raw_page":
                        result = get_raw_page(args['url'])
                    elif tool_call["name"] == "search_images":
                        result = search_images(args['query'], args.get('size', None), args.get('type_image', None), args.get('layout', None))
                    elif tool_call["name"] == "get_user_info":
                        result = get_user_info()

                    # Log tool usage
                    logger.info(f"Tool usage: {tool_call['name']}, args: {args}, result: {result}")

                    # Yield function message
                    yield f"data: {json.dumps({'choices': [{'delta': {'role': 'function', 'name': tool_call['name'], 'content': str(result)}}]})}\n\n"

                    # Add tool result to messages
                    messages.append({
                        "role": "function",
                        "name": tool_call["name"],
                        "content": str(result)
                    })

            # If there were tool calls, get a final completion with the updated messages
            if tool_calls:
                final_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=8192,
                    stream=True
                )

                for chunk in final_response:
                    if chunk.choices[0].delta.content is not None:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk.choices[0].delta.content}}]})}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
    except Exception as e:
        logger.error(f"Error in proxy_chat_completions: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

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
    app.run(debug=True, port=int(os.getenv('PORT', 5000)))
