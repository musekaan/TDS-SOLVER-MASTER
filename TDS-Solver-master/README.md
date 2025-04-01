
# TDS Solver

Welcome to **TDS Solver**! This is a simple project that creates an API (a special program that talks over the internet) to answer questions in two ways:
1. **If you send a ZIP file with a CSV file inside:**  
   It opens the ZIP, finds the CSV file, and picks out the answer from a column named "answer".
2. **If you send just a question (with no file):**  
   It uses an AI service (called AI Proxy) to generate an answer.

This guide will help you set up, run, and test the project step-by-step.

---

## What You Need

- **A computer** (Windows, macOS, or Linux)
- **Python 3:** Download from [python.org](https://www.python.org/downloads/)
- **Git (optional):** Helps with downloading the project from the internet ([git-scm.com](https://git-scm.com/))
- **Node.js (optional):** For sharing your project with others using LocalTunnel ([nodejs.org](https://nodejs.org/))

---

## Step 1: Download the Project

### Option A: Using Git (Recommended)
1. Open Command Prompt (or Terminal).
2. Run these commands:
   ```bash
   git clone <repository_url>
   cd TDS-Solver
   ```
   Replace `<repository_url>` with the link to the project repository.

### Option B: Without Git
1. Download the project ZIP file from your web browser.
2. Extract the ZIP file to a folder called `TDS-Solver`.

---

## Step 2: Set Up a Python Virtual Environment

A virtual environment keeps this project’s settings and libraries separate from other projects.

1. Open Command Prompt (or Terminal).
2. Change to your project folder. For example, on Windows:
   ```bash
   cd C:\Users\YourName\Documents\TDS-Solver
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - **On Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

---

## Step 3: Install Project Dependencies

With your virtual environment active and inside your project folder, run:
```bash
pip install -r requirements.txt
```
This command installs all the tools your project needs.

---

## Step 4: Set Up Your Environment Variables

This project uses an API key to talk to AI Proxy. We keep the key safe in a file called `.env`.

1. Create a file named **`.env`** in the project folder (the same folder as `app.py`).
2. Open the `.env` file with a text editor (like Notepad) and add:
   ```
   AI_PROXY_API_KEY=your_actual_api_key_here
   ```
   Replace `your_actual_api_key_here` with your real API key.

---

## Step 5: Run the API

Now, start your API by running:
```bash
python app.py
```
You should see a message like:
```
 * Running on http://127.0.0.1:5000/
```
This means your API is ready!

---

## Step 6: Test the API

### A. Testing with a Question Only (No File)
Open another Command Prompt (or use a tool like Postman) and run:
```bash
curl -X POST "http://127.0.0.1:5000/api/" -F "question=What is the capital of France?"
```

### B. Testing with a ZIP File
Make sure you have a ZIP file (for example, `abcd.zip`) that contains a CSV file with an "answer" column. Then run:
```bash
curl -X POST "http://127.0.0.1:5000/api/" -F "question=Download and unzip file abcd.zip which has a single extract.csv file inside. What is the value in the 'answer' column?" -F "file=@C:/path/to/abcd.zip"
```

---

## Optional: Share Your API with LocalTunnel

If you want to show your API to friends or run it on the internet, you can use LocalTunnel.

1. **Install LocalTunnel** (requires Node.js):
   ```bash
   npm install -g localtunnel
   ```
2. **Expose your local server:**
   ```bash
   lt --port 5000
   ```

---

## Step 7: Run Tests

The project includes tests to check if everything works correctly.

1. Make sure your virtual environment is active.
2. Run:
   ```bash
   pytest
   ```

---

## Troubleshooting

- **No API Key Loaded:**  
  Make sure your `.env` file is in the project folder and formatted correctly.
- **Server Errors:**  
  Look at the Command Prompt/Terminal where you ran `python app.py` for error messages.
- **Internet Issues:**  
  Ensure you have a good internet connection to talk to AI Proxy.

---

## Summary

- **What It Does:**  
  - Reads a question and, optionally, a ZIP file containing a CSV.
  - Extracts an answer from the CSV if available.
  - If no file is provided, asks AI Proxy for an answer.
  
- **How to Use It:**  
  - Set up your environment.
  - Install dependencies.
  - Add your API key in a `.env` file.
  - Run the app and test it with cURL or Postman.

By following these instructions step-by-step, you should be able to set up and run the TDS Solver project easily. Happy coding and exploring!

---

*If you have any questions, feel free to ask for help!*
