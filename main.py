import git
import mimetypes
from pathlib import Path
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
import urllib.request
import httpx
import asyncio
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
from typing import List, Dict, Any
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
    allow_methods=["GET", "POST"],  # Allows all methods
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
else:
    print("openai_api_key : ",openai_api_key)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
}

# 🔹 Task Definitions
TASKS = {
    "A1": "Install uv (if required) and run a Python script to generate data.",
    "A2": "Format a Markdown file using Prettier.",
    "A3": "Count the number of specific weekdays in a list of dates.",
    "A4": "Sort the array in a file by parameters.",
    "A5": "Extract the lines from files inside a directory.",
    "A6": "Extract tags from files inside a directory and create an index file.",
    "A7": "Extract the sender's email from an file contains an email message.",
    "A8": "Extract a number from an image file.",
    "A9": "Find the most similar pair of texts using embeddings.",
    "A10": "The task describes a database operation in natural language without including an explicit SQL query. The ticket type (e.g., 'Gold') is mentioned separately in the description.",
    "B3": "If the request involves retrieving data from a structured API (e.g., REST, GraphQL) using HTTP requests to an endpoint that returns JSON or XML and save it",
    "B4": "Clone a git repo and make a commit",
    "B5": "The task explicitly contains a full SQL query (e.g., 'SELECT ... FROM ... WHERE ...'). If the task only describes SQL without including a full query, classify it as A10.",
    "B6": "If the request involves extracting data from a webpage by parsing its HTML structure, typically using tools like BeautifulSoup, Selenium, or Puppeteer and save it",
    "B7": "Compress or resize an image",
    "B8": "Transcribe audio from an MP3 file",
    "B9": "Convert Markdown to HTML and save it to output file",
    "B10": "API endpoint that filters a CSV file and returns JSON data",
}

# 🔹 Step 1: Task input to English (LLM-Based)


async def translate_to_english(client, user_input: str) -> str:
    """
    Uses GPT to translate user input to English.
    """
    try:
        response = await client.post(
            BASE_URL + "/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Translate the following text to English. Return only the translated text, nothing else."},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return f"Error: {response.status_code}, {response.text}"

        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    except httpx.RequestError as e:
        return f"Request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# 🔹 Step 2: Task Classification (LLM-Based)


async def classify_task_async(client, user_input: str) -> str:
    """
    Uses GPT to classify the task based on user input.
    """
    try:
        response = await client.post(
            BASE_URL + "/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Classify the user's request as one of the following tasks: " +
                     ", ".join([f"{k}: {v}" for k, v in TASKS.items()]) +
                     " Return only the task ID (A1, A2, ..., A10, B3, B4,...., B10), nothing else."},
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

# 🔹 Step 3: Parameter Extraction


async def extract_parameters_with_llm(client, task_id: str, query: str) -> Dict[str, Any]:
    """
    Uses GPT-4o-mini to extract relevant parameters for any given task.
    """
    try:
        response = await client.post(
            BASE_URL + "/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI that extracts structured parameters from a user's query based on a task ID. "
                            "Return a valid JSON object with extracted parameters. If a parameter is missing, return null. "
                            "Strictly return only the JSON object, without explanations. Here are examples:\n"

                            "Task ID: A1\nQuery: 'Download the script from https://example.com/script.py and email it to user@example.com'\n"
                            "Output: {\"script_url\": \"https://example.com/script.py\", \"user_email\": \"user@example.com\"}\n"

                            "Task ID: A2\nQuery: 'Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place'\n"
                            "Output: {\"input_file\": \"/data/format.md\", \"prettier_version\": \"3.4.2\"}\n"

                            "Task ID: A3\nQuery: 'The file /data/dates.txt contains a list of dates. Count the number of Wednesdays and write it to /data/dates-wednesdays.txt.'\n"
                            "Output: {\"input_file\": \"/data/dates.txt\", \"day_of_week\": \"Wednesday\", \"output_file\": \"/data/dates-wednesdays.txt\"}\n"

                            "Task ID: A4\nQuery: 'Sort /data/employees.json by last_name, first_name and save to /data/sorted.json'\n"
                            "Output: {\"input_file\": \"/data/employees.json\", \"sort_keys\": [\"last_name\", \"first_name\"], \"output_file\": \"/data/sorted.json\"}\n"

                            "Task ID: A5\nQuery: 'Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first'\n"
                            "Output: {\"input_dir\": \"/data/logs/\", \"input_file_type\": \".log\", \"output_file\": \"/data/logs-recent.txt\"}\n"

                            "Task ID: A6\nQuery: 'Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title'\n"
                            "Output: {\"input_dir\": \"/data/docs/\", \"input_file_type\": \".md\",\"output_file\": \"/data/docs/index.json\"}\n"

                            "Task ID: A7\nQuery: '/data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender’s email address, and write just the email address to /data/email-sender.txt'\n"
                            "Output: {\"input_file\": \"/data/email.txt\", \"output_file\": \"/data/email-sender.txt\"}\n"

                            "Task ID: A8\nQuery: '/data/credit_card.png contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to /data/credit-card.txt'\n"
                            "Output: {\"image_file\": \"/data/credit_card.png\", \"output_file\": \"/data/credit-card.txt\"}\n"

                            "Task ID: A9\nQuery: '/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line'\n"
                            "Output: {\"input_file\": \"/data/comments.txt\", \"output_file\": \"/data/comments-similar.txt\"}\n"

                            "Task ID: A10\nQuery: 'The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt'\n"
                            "Output: {\"database_file\": \"/data/ticket-sales.db\", \"ticket_type\": \"Gold\",\"output_file\": \"/data/ticket-sales-gold.txt\"}\n"

                            "Task ID: B3\nQuery: 'Use get to fetch data from https://tds.s-anand.net/#/project-1 and write to /data/api_result.txt'\n"
                            "Output: {\"method\": \"get\", \"url\": \"https://tds.s-anand.net/#/project-1\", \"output_file\": \"/data/api_result.txt\"}\n"

                            "Task ID: B4\nQuery: 'Clone https://github.com/ and add new_file.txt file with 'This is a new file created through Python.' as file content and commit with 'Added a new file' message.'\n"
                            "Output: {\"repo_url\": \"https://github.com/\", \"file_name\": \"new_file.txt\", \"content\": \"This is a new file created through Python.\", \"commit_message\": \"Added a new file\"}\n"

                            "Task ID: B5\nQuery: 'Run 'SELECT SUM(units * price) FROM tickets WHERE type like '%Gold%'' in The SQLite database file /data/ticket-sales.db and write the number in /data/ticket-sales-gold.txt'\n"
                            "Output: {\"query\": \"SELECT SUM(units * price) FROM tickets WHERE type like '%Gold%'\",\"database_file\": \"/data/ticket-sales.db\",\"output_file\": \"/data/ticket-sales-gold.txt\"}\n"

                            "Task ID: B6\nQuery: 'Extract all h1 tag data from 'https://www.iitm.ac.in/''\n"
                            "Output: {\"url\": \"/data/format.md\", \"tag\": \"h1\"}\n"

                            "Task ID: B7\nQuery: 'Compress /data/credit_card.png by 85% quality to /data/compressed_credit_card.png'\n"
                            "Output: {\"input_file\": \"/data/credit_card.png\", \"quality\": \"85\", \"output_file\": \"/data/compressed_credit_card.png\"}\n"

                            "Task ID: B8\nQuery: 'Transcribe audio from /data/transcribing_1.mp3'\n"
                            "Output: {\"audio_file_path\": \"/data/transcribing_1.mp3\"}\n"

                            "Task ID: B9\nQuery: 'Convert /data/format.md to /data/format.html'\n"
                            "Output: {\"markdown_file\": \"/data/format.md\", \"html_file\": \"/data/format.html\"}\n"

                            "Task ID: B10\nQuery: 'filter a csv file /data/city.csv by New York city'\n"
                            "Output: {\"csv_file\": \"/data/city.csv\",\"column\": \"city\",\"value\": \"New York\"}\n"

                            "Now extract parameters for the following query."
                        )
                    },
                    {"role": "user", "content": f"Task ID: {task_id}\nQuery: {query}"}
                ],
                "temperature": 0,  # Keep responses deterministic
            },
            timeout=10,
        )

        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}, {response.text}"}

        # Extract JSON response
        response_data = response.json()
        extracted_text = response_data.get("choices", [{}])[0].get(
            "message", {}).get("content", "{}").strip()

        # Parse JSON response
        parameters = json.loads(extracted_text)
        return parameters

    except httpx.RequestError as e:
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from LLM"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# 🔹 Step 4: Task Execution


async def execute_task(task_id: str, parameters: Dict[str, str]):
    """
    Executes the task based on its category and parameters.
    """
    if task_id == "A1":
        a1(script_url=parameters["script_url"], user_email=parameters["user_email"])

    elif task_id == "A2":
        a2(prettier_version=parameters["prettier_version"], input_file=parameters["input_file"])

    elif task_id == "A3":
        await a3(input_file=parameters["input_file"], day_of_week=parameters["day_of_week"], output_file=parameters["output_file"])

    elif task_id == "A4":
        await a4(input_file=parameters["input_file"], sort_keys=parameters["sort_keys"], output_file=parameters["output_file"])

    elif task_id == "A5":
        await a5(input_dir=parameters["input_dir"], input_file_type=parameters["input_file_type"], output_file=parameters["output_file"])

    elif task_id == "A6":
        await a6(input_dir=parameters["input_dir"], input_file_type=parameters["input_file_type"], output_file=parameters["output_file"])

    elif task_id == "A7":
        await a7(input_file=parameters["input_file"], output_file=parameters["output_file"])

    elif task_id == "A8":
        await a8(image_file=parameters["image_file"], output_file=parameters["output_file"])

    elif task_id == "A9":
        await a9(input_file=parameters["input_file"], output_file=parameters["output_file"])

    elif task_id == "A10":
        await a10(database_file=parameters["database_file"], ticket_type=parameters["ticket_type"], output_file=parameters["output_file"])

    elif task_id == "B3":
        await b3(method=parameters["method"], url=parameters["url"], output_file=parameters["output_file"])

    elif task_id == "B4":
        await b4(repo_url=parameters["repo_url"], file_name=parameters["file_name"], content=parameters["content"], commit_message=parameters["commit_message"])

    elif task_id == "B5":
        await b5(query=parameters["query"], database_file=parameters["database_file"], output_file=parameters["output_file"])

    elif task_id == "B6":
        await b6(url=parameters["url"], tag=parameters["tag"])

    elif task_id == "B7":
        await b7(input_file=parameters["input_file"], output_file=parameters["output_file"])

    elif task_id == "B8":
        await b8(audio_file_path=parameters["audio_file_path"])

    elif task_id == "B9":
        await b9(markdown_file=parameters["markdown_file"], html_file=parameters["html_file"])

    elif task_id == "B10":
        await b10(csv_file=parameters["csv_file"], column=parameters["column"], value=parameters["value"])

# 🔹 Step 5: Orchestrate Everything
async def is_english_string(text):
    # Matches only ASCII characters (English letters, numbers, symbols)
    pattern = r"^[\x00-\x7F]+$"
    return bool(re.match(pattern, text))

async def query_llm(task: str):
    """
    Processes a user query by:
    1. Translating it to English (if needed)
    2. Classifying the task
    3. Extracting relevant parameters
    4. Executing the task

    Returns a response indicating success or failure.
    """

    task = task.replace("`", "").replace('"', "")  # Clean task input
    print(f"Received Task: {task}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 🔹 Step 1: Translate query to English
            if not await is_english_string(task):
                eng_task = await translate_to_english(client, task)
                print(f"📝 Task in English: {eng_task}")
            else:
                eng_task = task
                print(f"📝 Task is Already in English: {eng_task}")

            # 🔹 Step 2: Classify the task
            task_id = await classify_task_async(client, eng_task)
            print(f"📌 Task ID: {task_id}")

            if task_id == "Unknown":
                return Response("❌ Unknown Task", status_code=404)

            # 🔹 Step 3: Extract parameters using LLM
            parameters = await extract_parameters_with_llm(client, task_id, eng_task)
            print(f"📦 Extracted Parameters: {parameters}")

            # 🔹 Step 4: Execute the task
            execution_result = await execute_task(task_id, parameters)

            print(f"✅ Task {task_id} executed successfully!")
            return Response(f"✅ Task {task_id} executed successfully!", status_code=200)

    except httpx.HTTPError as e:
        print(f"❌ HTTP error occurred: {e}")
        return Response("❌ Failed due to a network error", status_code=500)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return Response("❌ An unexpected error occurred", status_code=500)
        
# run task API
@app.post("/run")
async def run_task(request: Request):
    task = request.query_params.get('task', '')
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

# task A1 to A10
def a1(script_url: str, user_email: str):
    # script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    """Install uvicorn, download script if missing, and run it."""
    script_name = os.path.basename(urlparse(script_url).path)
    try:
        subprocess.run([sys.executable, "-m", "pip","install", "uvicorn"], check=True)
        if not os.path.exists(script_name):
            urllib.request.urlretrieve(script_url, script_name)
        # Ensure the script has execute permissions (for Linux/macOS)
        os.chmod(script_name, 0o755)
        subprocess.run([sys.executable, script_name, user_email,"--root=./data"], check=True)
        return {"message": "Success!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")


def get_prettier_parser(file_path):
    """Determine the appropriate Prettier parser based on file type."""
    ext_to_parser = {
        ".js": "babel",
        ".jsx": "babel",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".json": "json",
        ".css": "css",
        ".scss": "scss",
        ".less": "less",
        ".html": "html",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml"
    }

    ext = Path(file_path).suffix  # Get file extension
    return ext_to_parser.get(ext, None)  # Return parser or None if not found


async def a2(prettier_version: str, input_file: str):
    """Run Prettier on the given file with the appropriate parser."""
    await b1(input_file)
    parser = get_prettier_parser(input_file)
    # input_file = "/data/format.md"
    # prettier_version = "3.4.2"
    input_file = root_path+input_file
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        raise HTTPException(
            status_code=404, detail=f"File not found: {input_file}")
    else:
        print(f"✅ File found: {input_file}")

    if parser is None:
        print(f"Unsupported file type: {input_file}")
        return
    try:
        # Use npx to run a specific version of Prettier (e.g., 3.4.2)
        result = subprocess.run(
            ["npx", f"prettier@{prettier_version}", "--write", input_file, "--parser", parser], check=True, text=True, capture_output=True)
        print(f"✅ File formatted successfully: {input_file}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"Error formatting the file: {e}")


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
    output_file_name = output_file
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
            print(f"✅ File written successfully: {output_file_name}")
            # return str(sorted_json)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid file content: {str(e)}")


async def a5(input_dir: str, input_file_type: str, output_file: str):
    # input_dir, output_file = '/data/logs', '/data/logs-recent.txt'
    await b1(input_dir)
    await b1(output_file)
    input_dir, output_file = root_path+input_dir, root_path+output_file
    if not os.path.isdir(input_dir):
        raise HTTPException(
            status_code=404, detail=f"Directory not found: {input_dir}")
    try:
        log_files = sorted(
            (file for file in os.listdir(input_dir)
             if file.endswith(input_file_type)),
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


async def a6(input_dir: str, input_file_type: str, output_file: str):
    # input_dir, output_file = "/data/docs/", "/data/docs/index.json"
    # input_file_type=".md"
    await b1(input_dir)
    await b1(output_file)
    input_dir, output_file = root_path+input_dir, root_path+output_file
    file_titles = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(input_file_type):
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
        response = requests.post(
            BASE_URL+"/embeddings",
            headers=headers,
            json={"model": "text-embedding-3-small", "input": texts},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
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


# task B1 to B10
async def b1(file_path: str):
    # Prevent directory traversal attacks
    if not file_path.startswith("/data"):
        raise HTTPException(
            status_code=400, detail="Invalid file path. Data outside /data can't be accessed or exfiltrated.")


@app.delete("/delete")
async def b2(request: Request):
    return JSONResponse({"error": "File deletion is not allowed."}, status_code=400)


async def b3(method: str = "get", url: str = "", headers: dict = None, json_input: dict = None, output_file: str = None, ):
    """
    Asynchronous function to make an HTTP request.

    Parameters:
        method (str): HTTP method (e.g., "get", "post", "put", "delete").
        url (str): Target URL for the request.
        headers (dict, optional): Headers to include in the request.
        json_input (dict, optional): JSON payload for POST/PUT requests.
        output_file (str, optional): If provided, response content is saved to this file.

    Returns:
        dict or str: JSON response if possible; otherwise, raw text response.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method.upper(),  # Ensure method is uppercase (GET, POST, etc.)
                url=url,
                headers=headers,
                json=json_input
            )
            response.raise_for_status()  # Raise error for non-2xx status codes

            await b1(output_file)
            # Try to parse response as JSON
            try:
                content = response.json()
            except json.JSONDecodeError:
                content = response.text

            # Save response to a file if requested
            if output_file:
                output_file_path = root_path+output_file

                with open(output_file_path, "w", encoding="utf-8") as f:
                    if isinstance(content, dict):
                        json.dump(content, f, indent=4)
                    else:
                        f.write(content)

                print(f"Response saved to {output_file_path}")

            return content  # Return response content

    except httpx.HTTPStatusError as e:
        return f"HTTP error: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


async def b4(repo_url: str, file_name: str, content: str, commit_message: str):
    # repo_url="https://github.com/username"
    # file_name="new_file.txt"
    # content="This is a new file created through Python."
    # commit_message="Added a new file"
    try:
        # Clone the repository
        repo_dir = repo_url.split('/')[-1].replace('.git', '')
        print(f"Cloning repo from {repo_url} into {repo_dir}...")
        repo = git.Repo.clone_from(repo_url, repo_dir)

        # Create the new file with content
        file_path = os.path.join(repo_dir, file_name)
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"Created file: {file_path}")

        # Stage the new file
        repo.git.add(file_name)
        print(f"Staged file: {file_name}")

        # Commit the changes
        repo.index.commit(commit_message)
        print(f"Committed changes with message: '{commit_message}'")

        # Push the changes to the remote repository
        origin = repo.remote(name='origin')
        origin.push()
        print("Pushed changes to remote repository.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


async def b5(query: str, database_file: str = "/data/ticket-sales.db", output_file: str = "/data/query_result.txt"):
    await b1(database_file)
    await b1(output_file)
    """Executes an SQLite query, returns the first result, and saves it to a file."""
    database_file = root_path + database_file
    output_file = root_path + output_file
    try:
        conn = sqlite3.connect(database_file)
        cur = conn.cursor()
        cur.execute(query)  # Execute query
        result = cur.fetchall()  # Fetch all results
        conn.close()  # Ensure connection is closed
        if result:
            first_result = result[0]  # Get the first row
            with open(output_file, "w") as file:
                file.write(str(first_result))  # Save to file
            return first_result  # Return the first result
        else:
            with open(output_file, "w") as file:
                file.write("No results found")  # Write empty result message
            return None

    except sqlite3.Error as e:
        raise HTTPException(
            status_code=400, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error occurred: {str(e)}")


async def b6(url: str, tag: str):
    # url = "https://www.iitm.ac.in/"
    # tag = "h1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract specific data (modify as needed)
        tags = soup.find_all(tag)  # Example: Extracting all tags
        if tag == "a":
            for idx, link in enumerate(tags, 1):
                href = link.get("href")  # Extract href attribute
            if href:  # Skip empty href values
                print(f"{idx}. {href}")
        else:
            for idx, tag in enumerate(tags, 1):
                print(f"{idx}. {tag.get_text(strip=True)}")
    else:
        print(f"Failed to retrieve page: {response.status_code}")
    return True


async def b7(input_file: str, output_file: str, quality: int = 85, resize: tuple = None) -> None:
    await b1(input_file)
    await b1(output_file)
    """Compress or resize an image, or both, while maintaining reasonable quality across multiple formats."""

    input_file, output_file = root_path + input_file, root_path + output_file

    try:
        with Image.open(input_file) as img:
            # Resize image if resize parameter is provided
            if resize:
                img = img.resize(resize, Image.ANTIALIAS)
                # img = img.resize(resize, Image.LANCZOS)

            # Determine output format from file extension
            output_file_ext = Path(output_file).suffix.lower()
            format_map = {
                ".jpg": "JPEG", ".jpeg": "JPEG",
                ".png": "PNG",
                ".webp": "WEBP",
                ".bmp": "BMP",
                ".tiff": "TIFF", ".tif": "TIFF",
                ".gif": "GIF"
            }
            format = format_map.get(output_file_ext)

            if format is None:
                raise ValueError(f"Unsupported file format: {output_file_ext}")

            # Convert RGBA to RGB for formats that don't support transparency
            if img.mode == 'RGBA' and format in ["JPEG", "BMP"]:
                img = img.convert('RGB')

            # Save image with appropriate format-specific settings
            save_args = {"format": format}

            if format == "JPEG":
                save_args.update(quality=quality, optimize=True)
            elif format == "PNG":
                save_args.update(optimize=True)  # PNG doesn't use quality
            elif format == "WEBP":
                save_args.update(quality=quality)  # WEBP supports quality
            elif format in ["BMP", "TIFF", "GIF"]:
                pass  # These formats don’t require quality/optimization

            img.save(output_file, **save_args)
            print(f"File written successfully: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


async def b8(audio_file_path: str):
    await b1(audio_file_path)
    # You can use "tiny", "base", "small", "medium", or "large"
    # model = whisper.load_model("base")
    # result = model.transcribe(mp3_file_path)
    # transcription=result["text"]

    # Load Vosk Model (Make sure to download a model and replace "model_path")
    audio_file_path = root_path+audio_file_path
    model_path = root_path+"/vosk-model-small-en-us-0.15"
    model = Model(model_path)
    audio_path_wav = audio_file_path.replace("mp3", "wav")
    ffmpeg.input(audio_file_path).output(audio_path_wav, format="wav",
                                    acodec="pcm_s16le", ar="16000", ac=1).run(overwrite_output=True)
    wf = wave.open(audio_path_wav, "rb")
    # Recognizer
    rec = KaldiRecognizer(model, wf.getframerate())
    # Process audio
    transcription = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcription += result["text"] + " "
    # Print the final transcription
    print("transcription")
    print(transcription)
    return transcription


async def b9(markdown_file: str = "/data/markdown_file.md", html_file: str = "/data/html_file.html"):
    await b1(markdown_file)
    await b1(html_file)
    markdown_file, html_file = root_path+markdown_file, root_path+html_file
    # Read the markdown content from the input file
    with open(markdown_file, 'r', encoding='utf-8') as md_file:
        markdown_content = md_file.read()
    # Convert the markdown content to HTML
    html_content = markdown2.markdown(markdown_content)
    # Write the HTML content to the output file
    with open(html_file, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)
    print(f"HTML content has been written to {html_file}")
    return True

# Pydantic model to define the structure of the filtered CSV rows


class Row(BaseModel):
    id: int
    name: str
    age: int
    city: str

# Function to handle CSV filtering logic


def filter_csv(csv_path: str, column: str, value: str) -> List[Row]:
    # Check if the file exists
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    filtered_data = []
    # Open the CSV file and filter data based on the column and value
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            # Filter the rows based on the provided column and value
            for row in csv_reader:
                if column in row and row[column] == value:
                    # Convert string to appropriate types as per the Row model
                    filtered_data.append(Row(id=int(row['id']), name=row['name'], age=int(row['age']), city=row['city']
                                             ))
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid column name")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # If no data matches, raise an exception
    if not filtered_data:
        raise HTTPException(
            status_code=404, detail="No data found for given filter")
    return filtered_data

# curl "http://127.0.0.1:8000/filter_csv?csvpath=/data/city.csv&column=city&value=New%20York"
# Endpoint to filter CSV file by CSV path and filter conditions


@app.get("/filter_csv", response_model=List[Row])
async def b10(csv_file: str, column: str, value: str):
    await b1(csv_file)
    return filter_csv(csv_file, column, value)

# read file API


@app.get("/read")
async def read_file(request: Request):
    file_path = request.query_params.get('path')
    # Ensure file path is safe
    await b1(file_path=file_path)
    full_file_path = root_path+file_path
    print("reading file at ", full_file_path)
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

# FastAPI startup event


@app.on_event("startup")
async def startup_event():
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    """Install uvicorn, download script if missing"""
    script_name = os.path.basename(urlparse(script_url).path)
    try:
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "uvicorn"], check=True)
        if not os.path.exists(script_name):
            urllib.request.urlretrieve(script_url, script_name)
        # Ensure the script has execute permissions (for Linux/macOS)
        os.chmod(script_name, 0o755)
        return {"message": "Success!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
