# import whisper
import ffmpeg
from fastapi import FastAPI, HTTPException, Request # type: ignore
from fastapi.responses import JSONResponse, Response # type: ignore
import os
import numpy as np
import json
import re
import sys
import subprocess
import httpx # type: ignore
import urllib.request
from dotenv import load_dotenv # type: ignore
from prompts import system_prompt
from datetime import datetime
import pytesseract # type: ignore
from PIL import Image, ImageEnhance, ImageFilter
import base64
import requests
import sqlite3
from typing import List
import csv
import markdown2
from pydantic import BaseModel # type: ignore
from bs4 import BeautifulSoup
import wave
import json
from vosk import Model, KaldiRecognizer


load_dotenv()
root_path = os.getcwd()
app = FastAPI()

openai_api_key = os.getenv('AIPROXY_TOKEN')
if not openai_api_key:
    print("openai_api_key missing")

BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
}

# LLM function call using GPT-4o-Mini
async def query_gpt(prompt: str):
    """Call the LLM (GPT-4o-Mini) to process a given prompt."""
    data = {"model": "gpt-4o-mini", "messages": [
        {"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(BASE_URL, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()["choices"][0]["message"]["content"]
            result = json.loads(response_data)
            func_name = result["func_name"]
            args = result["arguments"]
            if func_name in globals() and args:
                globals()[func_name](*args)
                return f"Function {func_name} executed successfully"
            elif func_name in globals():
                globals()[func_name]()
                return f"Function {func_name} executed successfully"
            elif func_name not in globals():
               return f"Function {func_name} not found."
    except httpx.RequestError as e:
        print(e)
        raise HTTPException(
            status_code=400, detail="Error contacting the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")
      
async def b1(file_path: str):
    # Prevent directory traversal attacks
    if not file_path.startswith("/data"):
        raise HTTPException(
            status_code=400, detail="Invalid file path. Data outside /data can't be accessed or exfiltrated.")

@app.delete("/delete")
async def b2(request: Request):
    return JSONResponse({"error": "File deletion is not allowed."}, status_code=400)
  
# run task API
@app.post("/run")
async def run_task(request: Request):
    task = request.query_params.get('task', '')
    if not task:
        raise HTTPException(
            status_code=400, detail="Task description is missing.")
    try:
        result = await query_gpt(task)
        if "successfully" in result:
            return Response(result, status_code=200)
        else:
            return Response(result, status_code=404)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# read file API
@app.get("/read")
async def read_file(request: Request):
    file_path = request.query_params.get('path')
    print(file_path)
    # Ensure file path is safe
    await b1(file_path=file_path)
    full_file_path = root_path+file_path
    print(full_file_path)
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

def a1(user_email: str):
    """Install uvicorn, download datagen.py if missing, and run it."""
    try:
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "uvicorn"], check=True)
        if not os.path.exists("datagen.py"):
            urllib.request.urlretrieve("https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py", "datagen.py")
        subprocess.run([sys.executable, "datagen.py",user_email, "--root", "./data"], check=True)
        return {"message": "Success!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

def a2():
    file_path = "/data/format.md"
    file_path = root_path+file_path
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {file_path}")
    try:

        # Use npx to run a specific version of Prettier (e.g., 3.4.2)
        result = subprocess.run(["npx", "prettier@3.4.2", file_path, "--write",
                                "--parser", "markdown"], check=True, text=True, capture_output=True)
        print(f"File formatted successfully: {file_path}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"Error formatting the file: {e}")

# List of possible date formats
DATE_FORMATS = [
    "%Y-%m-%d",          # 2014-12-29
    "%d-%b-%Y",          # 07-Jun-2004
    "%Y/%m/%d %H:%M:%S", # 2006/02/05 20:46:44
    "%Y/%m/%d",          # 2023/01/04
    "%b %d, %Y"          # Jan 31, 2022
]


def parse_date(date_str):
    """Try parsing a date string with multiple formats."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue  # Try next format
    raise ValueError(f"Unsupported date format: {date_str}")


def a3():
    input_path, output_path = "/data/dates.txt", "/data/dates-wednesdays.txt"
    input_path, output_path = root_path+input_path, root_path+output_path
    if not os.path.exists(input_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {input_path}")
    try:
        with open(input_path, 'r') as file:
            wednesday_count = sum(1 for line in file if parse_date(line.strip()).weekday() == 2)
        with open(output_path, 'w') as file:
            file.write(str(wednesday_count))
            print(f"File written successfully: {output_path}")
            return str(wednesday_count)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid date format: {str(e)}")


def a4():
    input_path, output_path = "/data/contacts.json", "/data/contacts-sorted.json"
    input_path, output_path = root_path+input_path, root_path+output_path
    if not os.path.exists(input_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {input_path}")
    try:
        with open(input_path, 'r') as file:
            sorted_contacts = sorted(json.load(file), key=lambda x: (x['last_name'], x['first_name']))
        with open(output_path, 'w') as file:
            json.dump(sorted_contacts, file, indent=4)
            print(f"File written successfully: {output_path}")
            return str(sorted_contacts)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid file content: {str(e)}") 


def a5():
    log_dir, recent_log_file = '/data/logs', '/data/logs-recent.txt'
    log_dir, recent_log_file = root_path+log_dir, root_path+recent_log_file
    if not os.path.isdir(log_dir):
        raise HTTPException(
            status_code=404, detail=f"Directory not found: {log_dir}")
    try:
        log_files = sorted(
            (file for file in os.listdir(log_dir) if file.endswith('.log')),
            key=lambda file: os.path.getmtime(os.path.join(log_dir, file)),
            reverse=True)
        recent_lines = []
        for log_file in log_files[:10]:
            log_path = os.path.join(log_dir, log_file)
            with open(log_path, 'r') as file:
                first_line = file.readline().strip()
                recent_lines.append(first_line)
        with open(recent_log_file, 'w') as file:
            file.write('\n'.join(recent_lines))
            print(f"File written successfully: {recent_log_file}")
        return str(recent_log_file)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error occurred: {str(e)}")


def a6():
    docs, index_file_path = "/data/docs/", "/data/docs/index.json"
    docs, index_file_path = root_path+docs, root_path+index_file_path
    file_titles = {}
    for root, _, files in os.walk(docs):
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
                                    file_path, docs)] = title
                                break
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    try:
        with open(index_file_path, "w", encoding="utf-8") as json_file:
            json.dump(file_titles, json_file,
                      ensure_ascii=False, indent=2)
        print(f"Index file created at: {index_file_path}")
        return True
    except Exception as e:
        print(f"Error while writing to index file {index_file_path}: {e}")

def a7():
    input_file, output_file = "/data/email.txt", "/data/email-sender.txt"
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
        response = requests.post(BASE_URL, headers=headers, json=payload)
        sender_email = response.json()["choices"][0]["message"]["content"]
        if sender_email:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(sender_email)
            # print(f"Sender's email extracted: {sender_email}")
            print(f"File written successfully: {output_file}")
        else:
            print("Sender's email not found.")
    except Exception as e:
        print(f"Error while writing to out file {output_file}: {e}")

def a8():
    image_path = "/data/credit_card.png"
    output_path = "/data/credit-card.txt"
    image_path, output_path = root_path+image_path, root_path+output_path
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    """Call the LLM (GPT-4o-Mini) to process a given prompt."""
    payload = {
        "model": "gpt-4o-mini", 
        "messages": [{
        "role": "user", 
            "content": [{"type": "text", "text": "You are given an image containing a text. Extract the number from the image"},
                        {"type": "image_url", "image_url":{"url":  f"data:image/png;base64,{base64_image}"}}]
        }]
        }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    # Print the response
    if response.status_code == 200:
        response_data = response.json()["choices"][0]["message"]["content"]
        pattern = r"(?:\d{4}[-\s]?){2}\d{4}[-\s]?\d{1,3}|\d{13,19}"
        match = re.search(pattern, response_data)
        cc_number = match.group(0).replace(" ", "")
        # print("Response:", cc_number)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(cc_number)
            print(f"File written successfully: {output_path}")
    else:
        print("Request failed with status code",response.status_code, response.text)

def b3(method: str, url: str, headers: json, json_input: json):
    if method=="get":
        response = requests.get(url, headers=headers, json=json_input, timeout=30)
    if method == "post":
        response = requests.post(url, headers=headers, json=json_input, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data
  
def a9():
    input_file = "/data/comments.txt"
    output_file = "/data/comments-similar.txt"
    input_file, output_file = root_path+input_file, root_path+output_file

    # Read comments from file
    try:
        with open(input_file, 'r', encoding="utf-8") as f:
            comments = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return False

    if len(comments) < 2:
        print("Not enough comments to compare.")
        return False

    # Get embeddings for each comment using OpenAI's API
    try:
        data = b3("post","http://aiproxy.sanand.workers.dev/openai/v1/embeddings",
                  headers, json={"model": "text-embedding-3-small", "input": comments}, )
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

    # print(f"{comments[i]}\n{comments[j]}\n")

    # Write to output file
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(f"{comments[i]}\n{comments[j]}\n")
            print(f"File written successfully: {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False
    return True

def b5(query: str,db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        result = cur.execute(query).fetchall()[0]
        # print("Query Result : ", result)
        conn.close()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error occurred: {str(e)}")

def a10():
    input_db_path = "/data/ticket-sales.db"
    output_file = "/data/ticket-sales-gold.txt"
    input_db_path, output_file = root_path+input_db_path, root_path+output_file
    try:
        total_sales = b5("SELECT SUM(units * price) FROM tickets WHERE type like '%Gold%'",input_db_path)[0]
        with open(output_file, 'w') as file:
            file.write(str(total_sales))
            print(f"File written successfully: {output_file}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error occurred: {str(e)}")

