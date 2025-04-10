import logging
import os
import zipfile
import io
import csv
import json
import requests
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch your AI Proxy API key from the environment variables
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    logger.error("AIPROXY_TOKEN not found in environment variables")
else:
    logger.info("AIPROXY_TOKEN loaded successfully")
    logger.info(AIPROXY_TOKEN)

app = Flask(__name__)

def get_llm_answer(question, file_data):
    AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
    if not AIPROXY_TOKEN:
        logger.error("AIPROXY_TOKEN not found in environment variables")
        return "Missing API token"

    prompt = f"""You're helping a data science student solving an objective assignment.
    Return only the final correct answer, nothing else — do not include explanation or formatting.

    Question: {question}
    Data (if any):
    {file_data}
    Answer:"""

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 600
    }

    try:
        logger.info("Sending request to AI Proxy...")
        response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=payload)
        logger.info("Status Code: %s", response.status_code)
        logger.info("Response Text: %s", response.text)

        response.raise_for_status()
        json_resp = response.json()
        answer = json_resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("Error calling AI Proxy API: %s", e)
        if response is not None:
            logger.error("Response content: %s", response.text)
        answer = "LLM generation failed"

    return answer

@app.route('/')
def home():
    return "Welcome to the TDS Solver API! Use the /api/ endpoint to interact with the service."

@app.route('/docs')
def docs():
    return """
    <h1>TDS Solver API Documentation</h1>
    <p>Use the <code>/api/</code> endpoint to send questions and files.</p>
    <p>Example:</p>
    <ul>
        <li>POST to <code>/api/</code> with a "question" parameter and an optional "file" parameter (ZIP, CSV, JSON, Excel, or other supported files).</li>
    </ul>
    """

@app.route('/api/', methods=['POST'])
def solve_assignment():
    question = request.form.get('question', '')
    uploaded_file = request.files.get('file', None)
    logger.info("Received question: %s", question)

    file_data = ""

    def try_decode(content):
        for encoding in ['utf-8', 'utf-16', 'cp1252']:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("Unable to decode with utf-8, utf-16, or cp1252")

    def process_file(file_content, fname):
        ext = fname.split('.')[-1].lower()
        if ext == 'csv':
            return try_decode(file_content)
        elif ext == 'json':
            return json.dumps(json.loads(try_decode(file_content)))
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(io.BytesIO(file_content))
            return df.to_csv(index=False)
        elif ext in ['txt', 'md']:
            return try_decode(file_content)
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

    def parse_uploaded_file(file_storage):
        nonlocal file_data
        filename = file_storage.filename.lower()
        file_bytes = file_storage.read()

        if filename.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
                for fname in zip_ref.namelist():
                    if fname.lower().endswith(('.csv', '.json', '.xls', '.xlsx', '.txt', '.md')):
                        with zip_ref.open(fname) as f:
                            content = f.read()
                            file_data += f"\n\n--- FILE: {fname} ---\n\n"
                            file_data += process_file(content, fname)
        else:
            file_data += f"\n\n--- FILE: {filename} ---\n\n"
            file_data += process_file(file_bytes, filename)

    try:
        if uploaded_file:
            parse_uploaded_file(uploaded_file)
    except Exception as e:
        logger.error("Error processing file: %s", e)
        return jsonify({"error": f"Error processing file: {e}"}), 400

    answer = get_llm_answer(question, file_data)
    return jsonify({"answer": answer})

@app.route('/ui')
def ui():
    return """
    <html>
    <head>
        <title>TDS Solver UI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            label, input, textarea, button { display: block; margin-top: 10px; }
            textarea { width: 100%; height: 100px; }
        </style>
    </head>
    <body>
        <h1>TDS Solver – Interactive UI</h1>
        <form id="solverForm" enctype="multipart/form-data">
            <label for="question">Enter Question:</label>
            <textarea name="question" id="question" required></textarea>

            <label for="file">Upload File (optional):</label>
            <input type="file" name="file" id="file">

            <button type="submit">Submit</button>
        </form>
        <h3>Answer:</h3>
        <div id="result"></div>

        <script>
            document.getElementById("solverForm").onsubmit = async function (e) {
                e.preventDefault();
                const form = document.getElementById("solverForm");
                const formData = new FormData(form);
                const resultBox = document.getElementById("result");
                resultBox.innerText = "Computing...";
                const response = await fetch("/api/", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();
                resultBox.innerText = result.answer || result.error;
                form.reset();
            };
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
