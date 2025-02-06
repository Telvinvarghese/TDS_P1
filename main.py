import numpy as np
from datetime import datetime
from urllib.parse import urlparse
import urllib.request
import httpx
import os
import json
import re
import sys
import subprocess
from typing import Dict, Optional
import sys
from fastapi import FastAPI, HTTPException, Request  # type: ignore
from fastapi.responses import JSONResponse, Response  # type: ignore
from fastapi.middleware.cors import CORSMiddleware
import ffmpeg
import pytesseract  # type: ignore
from PIL import Image, ImageEnhance, ImageFilter
import base64
import requests
import sqlite3
from typing import List
import csv
import markdown2
from pydantic import BaseModel  # type: ignore
from bs4 import BeautifulSoup
import wave
import json
from vosk import Model, KaldiRecognizer

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET","POST"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# run app
@app.get("/")
async def root():
    print("Successfully rendering app")
    return JSONResponse(content={"message": "Successfully rendering app"})

root_path = os.getcwd()

# 🔹 OpenAI API Endpoint
BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1"

openai_api_key = os.getenv('AIPROXY_TOKEN')
if not openai_api_key:
    print("openai_api_key missing")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
}

# 🔹 Task Definitions
TASKS = {
    "A1": "Install uv (if required) and run a Python script to generate data.",
    "A2": "Format a Markdown file using Prettier.",
    "A3": "Count the number of specific weekdays in a list of dates.",
    "A4": "Sort a JSON file by parameters.",
    "A5": "Extract the lines from log files.",
    "A6": "Extract tags from Markdown files and create an index file.",
    "A7": "Extract the sender's email from an email file.",
    "A8": "Extract a number from an image.",
    "A9": "Find the most similar pair of texts using embeddings.",
    "A10": "Run a query in an SQLite database.",
}

# 🔹 Step 1: Task Classification (LLM-Based)

async def classify_task_async(client, user_input: str) -> str:
    """
    Uses GPT to classify the task based on user input.
    """
    try:
        response = await client.post(
            BASE_URL + "/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Classify the user's request as one of the following tasks: " +
                     ", ".join([f"{k}: {v}" for k, v in TASKS.items()]) +
                     " Return only the task ID (A1, A2, ..., A10), nothing else."},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return f"Error: {response.status_code}, {response.text}"

        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "Unknown").strip()

    except httpx.RequestError as e:
        return f"Request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# 🔹 Step 2: Parameter Extraction

def extract_parameters(task_id: str, query: str) -> Dict[str, Optional[str]]:
    """
    Extracts relevant parameters from the task description.
    """
    parameters = {}

    if task_id == "A1":
        parameters["script_url"] = re.search(r"(https?://[^\s]+)", query).group(1) if "http" in query else None
        # Use regex to find an email address in the query
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", query)
        # Assign the found email to parameters, or None if not found
        parameters["user_email"] = email_match.group(0) if email_match else None
        # parameters["user_email"] = "${user.email}" if "${user.email}" in query else None

    elif task_id == "A2":
        parameters["prettier_version"] = re.search(r"prettier@([\d.]+)", query).group(1) if "prettier@" in query else "latest"
        match = re.search(r"(/data/[\w./-]+\.md)", query)
        if match:
            parameters["markdown_file"] = match.group(1)
        else:
            raise HTTPException(status_code=400,detail="Invalid file path. Data outside /data can't be accessed or exfiltrated.")
        
    elif task_id == "A3":
        # 1️⃣ Detect day of the week (English, Hindi, Tamil)
        day_pattern = r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|" \
                      r"रविवार|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|" \
                      r"ஞாயிறு|திங்கள்|செவ்வாய்|புதன்|வியாழன்|வெள்ளி|சனி)"
        match_day = re.search(day_pattern, query, re.IGNORECASE)

        # Hindi & Tamil to English conversion
        day_translation = {
            "रविवार": "Sunday", "सोमवार": "Monday", "मंगलवार": "Tuesday",
            "बुधवार": "Wednesday", "गुरुवार": "Thursday", "शुक्रवार": "Friday", "शनिवार": "Saturday",
            "ஞாயிறு": "Sunday", "திங்கள்": "Monday", "செவ்வாய்": "Tuesday",
            "புதன்": "Wednesday", "வியாழன்": "Thursday", "வெள்ளி": "Friday", "சனி": "Saturday"
        }

        detected_day = match_day.group(0) if match_day else None
        parameters["day_of_week"] = day_translation.get(
            detected_day, detected_day)  # Convert to English

        # 2️⃣ Extract all file paths from the query
        file_pattern = r"(/data/[\w./-]+\.(?:txt|log|dates))"
        matches = re.findall(file_pattern, query)

        # Set input_file as the first match (if exists)
        parameters["input_file"] = matches[0] if len(matches) > 0 else None

        # 3️⃣ Extract output file only if "write" (or Hindi/Tamil equivalent) appears before it
        output_pattern = r"(?:लिखो|எழுது|write)[^\n]*?\s(/data/[\w./-]+\.(?:txt|log|dates))"
        match_output = re.search(output_pattern, query)
        parameters["output_file"] = match_output.group(1) if match_output else (matches[1] if len(matches) > 1 else None)

    elif task_id == "A4":
        parameters["input_file"] = re.search(
            r"(/data/[\w./-]+\.json)", query).group(1)

        # print(re.search(
        #     r"(/data/[\w./-]+\.json)$", query))
        # Extract output file (if present)
        output_match = re.search(r"(write|store).*to\s+(/data/[\w./-]+\.json)", query)
        parameters["output_file"] = output_match.group(
            2) if output_match else None
        # parameters["sort_keys"] = ["last_name", "first_name"]
        # Extract sorting keys, handling "and" and "then"
        sort_match = re.search(r"by\s+([\w\s,]+?)(?=\s+and (write|store)\s*)", query)
        # print(sort_match)
        if sort_match:
            parameters["sort_keys"] = [key.strip() for key in re.split(
                r",|\bthen\b|\band\b", sort_match.group(1)) if key.strip()]
        else:
            parameters["sort_keys"] =[]
    
    elif task_id == "A5":
        parameters["input_dir"] = re.findall(
            r"(/data/[\w.-]+(?:/[\w.-]+)*/)", query)[0]
        parameters["output_file"] = re.search(r"\s(/data/[\w./-]+\.(?:txt|log|dates))", query).group(1)

    elif task_id == "A6":
        parameters["input_dir"] = re.findall(r"(/data/[\w.-]+(?:/[\w.-]+)*/)", query)[0]
        parameters["output_file"] = re.search(r"\s+(/data/[\w./-]+\.json)", query).group(1)

    elif task_id == "A7":
        parameters["input_file"] = re.search(
            r"(/data/[\w./-]+\.(?:txt|log))", query).group(1)
        parameters["output_file"] = re.search(
            r"\s+(/data/[\w./-]+\.(?:txt|log))", query).group(1)
        
    elif task_id == "A8":
        parameters["image_file"] = re.search(
            r"(/data/[\w./-]+\.(?:png|jpg|jpeg))", query).group(1)
        parameters["output_file"] = re.search(
            r"\s+(/data/[\w./-]+\.(?:txt|log))", query).group(1)
        
    elif task_id == "A9":
        parameters["input_file"] = re.search(
            r"(/data/[\w./-]+\.(?:txt|log|dates))", query).group(1)
        parameters["output_file"] = re.search(
            r"\s+(/data/[\w./-]+\.(?:txt|log|dates))", query).group(1)
        
    elif task_id == "A10":
        parameters["database_file"] = re.search(
            r"(/data/[\w./-]+\.db)", query).group(1)
        parameters["ticket_type"] = "Gold"
        parameters["output_file"] = re.search(
            r"\s+(/data/[\w./-]+\.(?:txt|log|dates))", query).group(1)

    return parameters

# 🔹 Step 3: Task Execution

async def execute_task(task_id: str, parameters: Dict[str, str]):
    """
    Executes the task based on its category and parameters.
    """
    if task_id == "A1":
        print("parameters: ", "script_url:", parameters["script_url"], "and", "user_email:", parameters["user_email"])
        a1(parameters["script_url"], parameters["user_email"])

    elif task_id == "A2":
        print("parameters: ","prettier_version:", parameters["prettier_version"],"and","markdown_file:",parameters["markdown_file"])
        a2(parameters["prettier_version"], parameters["markdown_file"])

    elif task_id == "A3":
        print("parameters: ", "input_file:", parameters["input_file"],",","day_of_week:",parameters["day_of_week"], "and", "output_file:", parameters["output_file"])
        await a3(parameters["input_file"],parameters["day_of_week"], parameters["output_file"])

    elif task_id == "A4":
        print("parameters: ", "input_file:", parameters["input_file"], ",", "sort_keys:",parameters["sort_keys"], "and", "output_file:", parameters["output_file"])
        await a4(parameters["input_file"], parameters["sort_keys"],parameters["output_file"])
        
    elif task_id == "A5":
        print("parameters: ", "input_dir:",parameters["input_dir"], "and", "output_file:", parameters["output_file"])
        await a5(parameters["input_dir"], parameters["output_file"])

    elif task_id == "A6":
        print("parameters: ", "input_dir:",parameters["input_dir"], "and", "output_file:", parameters["output_file"])
        await a6(parameters["input_dir"], parameters["output_file"])

    elif task_id == "A7":
        print("parameters: ", "input_file:",parameters["input_file"], "and", "output_file:", parameters["output_file"])
        await a7(parameters["input_file"], parameters["output_file"])
    
    elif task_id == "A8":
        print("parameters: ", "image_file:",parameters["image_file"], "and", "output_file:", parameters["output_file"])
        await a8(parameters["image_file"], parameters["output_file"])

    elif task_id == "A9":
        print("parameters: ", "input_file:",parameters["input_file"], "and", "output_file:", parameters["output_file"])
        await a9(parameters["input_file"], parameters["output_file"])

    elif task_id == "A10":
        print("parameters: ", "database_file:", parameters["database_file"], ",", "ticket_type:",parameters["ticket_type"], "and", "output_file:", parameters["output_file"])
        await a10(parameters["database_file"],parameters["ticket_type"], parameters["output_file"])

# 🔹 Step 4: Orchestrate Everything

# run task API
async def query_llm(task):
    task = task.replace("`", "")
    async with httpx.AsyncClient() as client:
        task_id = await classify_task_async(client, task)
    if "Unknown" != task_id:
        parameters = extract_parameters(task_id, task)
        print(task_id, parameters)
        await execute_task(task_id, parameters)
        print(f"✅ Task {task_id} executed successfully!")
        return Response(f"✅ Task {task_id} executed successfully!", status_code=200)
    else:
        return Response(f"Unknown Task", status_code=404)

@app.post("/run")
async def run_task(request: Request):
    task = request.query_params.get('task', '').replace("`", "")
    print("task: ", task)
    if not task:
        raise HTTPException(
            status_code=400, detail="Task description is missing.")
    try:
        response = await query_llm(task)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def a1(script_url: str, user_email: str):
    # script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    """Install uvicorn, download script if missing, and run it."""
    script_name = os.path.basename(urlparse(script_url).path)
    try:
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "uvicorn"], check=True)
        if not os.path.exists(script_name):
            urllib.request.urlretrieve(
                script_url, script_name)
        subprocess.run([sys.executable, script_name,user_email, "--root", "./data"], check=True)
        return {"message": "Success!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
def a2(prettier_version: str, markdown_file: str):
    # markdown_file = "/data/format.md"
    # prettier_version = "3.4.2"
    markdown_file = root_path+markdown_file
    if not os.path.exists(markdown_file):
        print(f"File not found: {markdown_file}")
        raise HTTPException(
            status_code=404, detail=f"File not found: {markdown_file}")
    else:
        print(f"✅ File found: {markdown_file}")
    try:
        # Use npx to run a specific version of Prettier (e.g., 3.4.2)
        result=subprocess.run(
            ["npx", f"prettier@{prettier_version}", "--write", markdown_file, "--parser", "markdown"], check=True, text=True, capture_output=True)
        print(f"✅ File formatted successfully: {markdown_file}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error formatting the file: {e}")
    
# List of possible date formats
DATE_FORMATS = [
    "%Y-%m-%d",          # 2014-12-29
    "%d-%b-%Y",          # 07-Jun-2004
    "%Y/%m/%d %H:%M:%S",  # 2006/02/05 20:46:44
    "%Y/%m/%d",          # 2023/01/04
    "%b %d, %Y"          # Jan 31, 2022
]

# Mapping days of the week to numbers (Monday = 0, ..., Sunday = 6)
WEEKDAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

def parse_date(date_str):
    """Try parsing a date string with multiple formats."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue  # Try next format
    raise ValueError(f"Unsupported date format: {date_str}")

async def a3(input_file: str, day_of_week: str, output_file: str):
    # input_file, output_file = "/data/dates.txt", "/data/dates-wednesdays.txt"
    # day_of_week = "Wednesday"
    """ Count occurrences of a specific weekday in a file and optionally write to output. """
    await b1(input_file)
    await b1(output_file)
    input_file, output_file = root_path+input_file, root_path+output_file
    # Convert day_of_week to a number (validate input)
    if day_of_week not in WEEKDAY_MAP:
        raise HTTPException(
            status_code=400, detail=f"Invalid day_of_week: {day_of_week}")

    day_number = WEEKDAY_MAP[day_of_week]

    # Ensure the input file exists
    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=404, detail=f"File not found: {input_file}")

    try:
        # Read input file and count occurrences of the given weekday
        with open(input_file, 'r') as file:
            count = sum(1 for line in file if parse_date(
                line.strip()).weekday() == day_number)

        # Write to output file (if provided)
        if output_file:
            with open(output_file, 'w') as file:
                file.write(str(count))
                print(f"✅ File written successfully: {output_file}")

        return str(count)

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid date format: {str(e)}")

async def a4(input_file: str, sort_keys: list, output_file: str):
    # input_file, output_file = "/data/contacts.json", "/data/contacts-sorted.json"
    # sort_keys=['last_name', 'first_name']
    await b1(input_file)
    await b1(output_file)
    input_file, output_file = root_path+input_file, root_path+output_file
    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=404, detail=f"File not found: {input_file}")
    try:
        with open(input_file, 'r') as input_file:
            json_input = json.load(input_file)
           # Sort dynamically based on provided keys
            sorted_json = sorted(json_input, key=lambda x: tuple(
                x.get(key, "") for key in sort_keys))
        with open(output_file, 'w') as output_file:
            json.dump(sorted_json, output_file, indent=4)
            print(f"✅ File written successfully: {output_file}")
            # return str(sorted_json)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid file content: {str(e)}")

async def a5(input_dir: str, output_file: str):
    # input_dir, output_file = '/data/logs', '/data/logs-recent.txt'
    await b1(input_dir)
    await b1(output_file)
    input_dir, output_file = root_path+input_dir, root_path+output_file
    if not os.path.isdir(input_dir):
        raise HTTPException(
            status_code=404, detail=f"Directory not found: {input_dir}")
    try:
        log_files = sorted(
            (file for file in os.listdir(input_dir) if file.endswith('.log')),
            key=lambda file: os.path.getmtime(os.path.join(input_dir, file)),
            reverse=True)
        recent_lines = []
        for log_file in log_files[:10]:
            log_path = os.path.join(input_dir, log_file)
            with open(log_path, 'r') as file:
                first_line = file.readline().strip()
                recent_lines.append(first_line)
        with open(output_file, 'w') as file:
            file.write('\n'.join(recent_lines))
            print(f"✅ File written successfully: {output_file}")
        return str(output_file)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error occurred: {str(e)}")

async def a6(input_dir: str, output_file: str):
    # input_dir, output_file = "/data/docs/", "/data/docs/index.json"
    await b1(input_dir)
    await b1(output_file)
    input_dir, output_file = root_path+input_dir, root_path+output_file
    file_titles = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        # Extract the first H1 title
                        for line in f:
                            if line.startswith("# "):
                                title = line.strip("# ").strip()
                                file_titles[os.path.relpath(
                                    file_path, input_dir)] = title
                                break
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    try:
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(file_titles, json_file,
                      ensure_ascii=False, indent=2)
        print(f"✅ Index file created at: {output_file}")
        return True
    except Exception as e:
        print(f"Error while writing to index file {output_file}: {e}")

async def a7(input_file: str, output_file: str):
    # input_file, output_file = "/data/email.txt", "/data/email-sender.txt"
    await b1(input_file)
    await b1(output_file)
    input_file, output_file = root_path+input_file, root_path+output_file
    sender_email = None
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            email_content = file.readlines()
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": f"Extract the sender's email address from the following text : {email_content} and return only the sender's email address, nothing else"
                }
            ]
        }
        response = requests.post(
            BASE_URL + "/chat/completions", headers=headers, json=payload)
        sender_email = response.json()["choices"][0]["message"]["content"]
        if sender_email:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(sender_email)
            print(f"✅ File written successfully: {output_file}")
        else:
            print("Sender's email not found.")
    except Exception as e:
        print(f"Error while writing to out file {output_file}: {e}")

async def a8(image_file: str, output_file: str):
    # image_file = "/data/credit_card.png"
    # output_file = "/data/credit-card.txt"
    await b1(image_file)
    await b1(output_file)
    image_file, output_file = root_path+image_file, root_path+output_file
    with open(image_file, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    """Call the LLM (GPT-4o-Mini) to process a given prompt."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "You are given an image containing a text. Extract the number from the image"},
                        {"type": "image_url", "image_url": {"url":  f"data:image/png;base64,{base64_image}"}}]
        }]
    }
    response = requests.post(
        BASE_URL + "/chat/completions", headers=headers, json=payload)
    # Print the response
    if response.status_code == 200:
        response_data = response.json()["choices"][0]["message"]["content"]
        pattern = r"(?:\d{4}[-\s]?){2}\d{4}[-\s]?\d{1,3}|\d{13,19}"
        match = re.search(pattern, response_data)
        cc_number = match.group(0).replace(" ", "")
        # print("Response:", cc_number)
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(cc_number)
            print(f"✅ File written successfully: {output_file}")
    else:
        print("Request failed with status code",
              response.status_code, response.text)

async def a9(input_file: str, output_file: str):
    # input_file = "/data/comments.txt"
    # output_file = "/data/comments-similar.txt"
    await b1(input_file)
    await b1(output_file)
    input_file, output_file = root_path+input_file, root_path+output_file
    # Read texts from file
    try:
        with open(input_file, 'r', encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return False
    if len(texts) < 2:
        print("Not enough texts to compare.")
        return False
    # Get embeddings for each text using OpenAI's API
    try:
        data = b3("post", BASE_URL+"/embeddings",
                  headers, {"model": "text-embedding-3-small", "input": texts}, )
        embeddings = np.array([emb["embedding"]
                              for emb in data.get("data", [])])
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return False
    if embeddings.shape[0] < 2:
        print("Not enough embeddings retrieved.")
        return False
    # Compute similarity matrix
    similarity_matrix = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarity_matrix, -np.inf)  # Ignore self-similarity
    # Find the most similar pair
    i, j = np.unravel_index(
        np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
    # print(f"{texts[i]}\n{texts[j]}\n")
    # Write to output file
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(f"{texts[i]}\n{texts[j]}\n")
            print(f"✅ File written successfully: {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False
    return True

async def a10(database_file: str, ticket_type: str, output_file: str):
    # database_file = "/data/ticket-sales.db"
    # output_file = "/data/ticket-sales-gold.txt"
    await b1(database_file)
    await b1(output_file)
    database_file, output_file = root_path+database_file, root_path+output_file
    if not os.path.exists(database_file):
        raise FileNotFoundError(f"Database file not found: {database_file}")
    try:
        conn = sqlite3.connect(database_file)
        cur = conn.cursor()
        total_sales = cur.execute(
            "SELECT SUM(units * price) FROM tickets WHERE type like '%"+ticket_type+"%'").fetchone()[0]
        print("Total sales of Gold tickets : ", total_sales)
        conn.close()
        with open(output_file, 'w') as file:
            file.write(str(total_sales))
            print(f"✅ File written successfully: {output_file}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error occurred: {str(e)}")

async def b1(file_path: str):
    # Prevent directory traversal attacks
    if not file_path.startswith("/data"):
        raise HTTPException(
            status_code=400, detail="Invalid file path. Data outside /data can't be accessed or exfiltrated.")
    
@app.delete("/delete")
async def b2(request: Request):
    return JSONResponse({"error": "File deletion is not allowed."}, status_code=400)

def b3(method: str, url: str, headers: dict = None, json_input: dict = None):
    method = method.lower()

    try:
        if method == "get":
            response = requests.get(
                url, headers=headers, json=json_input, timeout=30)
        elif method == "post":
            response = requests.post(
                url, headers=headers, json=json_input, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return response.json()  # Return parsed JSON data

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# read file API
@app.get("/read")
async def read_file(request: Request):
    file_path = request.query_params.get('path')
    # Ensure file path is safe
    await b1(file_path=file_path)
    full_file_path = root_path+file_path
    print("reading file at ",full_file_path)
    if not full_file_path:
        return JSONResponse({"error": "File path is missing."}, status_code=400)
    if not os.path.exists(full_file_path):
        return Response("File not found", status_code=404)
    try:
        with open(full_file_path, 'r') as file:
            content = file.read()
        return Response(content, media_type="text/plain", status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
