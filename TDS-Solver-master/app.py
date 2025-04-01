import logging
import os
import zipfile
import io
import csv
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch your AI Proxy API key from the environment variables
AI_PROXY_API_KEY = os.environ.get("AI_PROXY_API_KEY")
if not AI_PROXY_API_KEY:
    logger.error("AI_PROXY_API_KEY not found in environment variables")
else:
    logger.info("AI_PROXY_API_KEY loaded successfully")

app = Flask(__name__)

def get_llm_answer(question, csv_data):
    """
    Call AI Proxy API to generate an answer based on the question and CSV data.
    """
    prompt = f"Question: {question}\nCSV Data: {csv_data}\nProvide a clear answer:"
    headers = {
        "Authorization": f"Bearer {AI_PROXY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7
    }
    response = None
    try:
        response = requests.post("https://api.aiproxy.com/generate", headers=headers, json=payload)
        response.raise_for_status()
        json_resp = response.json()
        logger.info("AI Proxy response: %s", json_resp)
        answer = json_resp.get("answer", "No answer generated")
    except Exception as e:
        logger.error("Error calling AI Proxy API: %s", e)
        if response is not None:
            logger.error("Response status code: %s", response.status_code)
            logger.error("Response content: %s", response.text)
        answer = "LLM generation failed"
    return answer

@app.route('/api/', methods=['POST'])
def solve_assignment():
    question = request.form.get('question', '')
    file = request.files.get('file', None)
    logger.info("Received question: %s", question)

    answer = None
    csv_data = None

    if file:
        try:
            file_bytes = file.read()
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
                file_list = zip_ref.namelist()
                if not file_list:
                    logger.error("Zip file is empty")
                    return jsonify({"error": "Zip file is empty"}), 400

                # Find the first CSV file in the zip archive
                csv_filename = None
                for fname in file_list:
                    if fname.lower().endswith('.csv'):
                        csv_filename = fname
                        break

                if not csv_filename:
                    logger.error("No CSV file found in the zip archive")
                    return jsonify({"error": "No CSV file found in the zip archive"}), 400

                with zip_ref.open(csv_filename) as csv_file:
                    csv_content = csv_file.read().decode('utf-8')
                    csv_data = csv_content  # Keep CSV data for potential LLM integration
                    reader = csv.DictReader(io.StringIO(csv_content))
                    for row in reader:
                        if 'answer' in row and row['answer']:
                            answer = row['answer']
                            break

                    if not answer:
                        logger.warning("Answer column not found or empty in CSV")
                        return jsonify({"error": "Answer column not found or empty in CSV"}), 400

        except zipfile.BadZipFile:
            logger.error("Uploaded file is not a valid zip file")
            return jsonify({"error": "File is not a valid zip file"}), 400
        except Exception as e:
            logger.error("Error processing file: %s", e)
            return jsonify({"error": f"Error processing file: {e}"}), 400
    else:
        # No file provided; generate answer dynamically using AI Proxy.
        logger.info("No file provided; generating answer using AI Proxy")
        csv_data = ""
        answer = get_llm_answer(question, csv_data)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
