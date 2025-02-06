# import whisper
import git
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
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

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

root_path = os.getcwd()

openai_api_key = os.getenv('AIPROXY_TOKEN')
if not openai_api_key:
    print("openai_api_key missing")
else:
    print("API key set successfully")
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
        async with httpx.AsyncClient(timeout=60) as client:
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
        data = b3("post","http://aiproxy.sanand.workers.dev/openai/v1/embeddings",headers, {"model": "text-embedding-3-small", "input": comments}, )
        embeddings = np.array([emb["embedding"] for emb in data.get("data", [])])
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

def a10():
    input_db_path = "/data/ticket-sales.db"
    output_file = "/data/ticket-sales-gold.txt"
    input_db_path, output_file = root_path+input_db_path, root_path+output_file
    print(input_db_path, output_file)
    if not os.path.exists(input_db_path):
        raise FileNotFoundError(f"Database file not found: {input_db_path}")
    try:
        conn = sqlite3.connect(input_db_path)
        cur = conn.cursor()
        total_sales = cur.execute("SELECT SUM(units * price) FROM tickets WHERE type like '%Gold%'").fetchone()[0]
        print("Total sales of Gold tickets : ", total_sales)
        conn.close()
        with open(output_file, 'w') as file:
            file.write(str(total_sales))
            print(f"File written successfully: {output_file}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error occurred: {str(e)}")

# run app
@app.get("/")
async def read_root():
    print("Successfully rendering app")
    return JSONResponse(content={"message": "Successfully rendering app"})

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
            response = requests.post(url, headers=headers, json=json_input, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return response.json()  # Return parsed JSON data

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def b4(repo_url: str, file_name: str, content: str, commit_message: str):
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

def b5(query: str,db_path: str):
    db_path = root_path+db_path
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

def b6(url: str, tag: str):
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

def b7(input_path: str, output_path: str, quality: int = 85, resize: tuple = None) -> None:
    input_path, output_path=root_path+input_path,root_path+output_path
    """Compress or resize an image, or both, while maintaining reasonable quality."""
    try:
        with Image.open(input_path) as img:
            # Resize image if resize parameter is provided
            if resize:
                img = img.resize(resize, Image.ANTIALIAS)

            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # Save with optimization for PNG, or use quality for JPEG
            if input_path.lower().endswith('.png'):
                img.save(output_path, 'PNG', optimize=True)
            elif input_path.lower().endswith('.jpg') or input_path.lower().endswith('.jpeg'):
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
            else:
                img.save(output_path)

            print(f"File written successfully: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def b8(audio_path: str):

    # You can use "tiny", "base", "small", "medium", or "large"
    # model = whisper.load_model("base")
    # result = model.transcribe(mp3_file_path)
    # transcription=result["text"]

    # Load Vosk Model (Make sure to download a model and replace "model_path")
    audio_path=root_path+audio_path
    model_path = root_path+"/vosk-model-small-en-us-0.15"
    model = Model(model_path)
    audio_path_wav = audio_path.replace("mp3", "wav")
    ffmpeg.input(audio_path).output(audio_path_wav, format="wav",acodec="pcm_s16le", ar="16000", ac=1).run(overwrite_output=True)
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

def b9(input_markdown_file: str, output_html_file: str):
    input_markdown_file, output_html_file = root_path+input_markdown_file, root_path+output_html_file
    # Read the markdown content from the input file
    with open(input_markdown_file, 'r', encoding='utf-8') as md_file:
        markdown_content = md_file.read()
    # Convert the markdown content to HTML
    html_content = markdown2.markdown(markdown_content)
    # Write the HTML content to the output file
    with open(output_html_file, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)
    print(f"HTML content has been written to {output_html_file}")
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
                    filtered_data.append(Row(id=int(row['id']),name=row['name'],age=int(row['age']),city=row['city']
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

# curl "http://127.0.0.1:8000/filter_csv?csvpath=city.csv&column=city&value=New%20York"
# Endpoint to filter CSV file by CSV path and filter conditions
@app.get("/filter_csv", response_model=List[Row])
async def b10(csvpath: str, column: str, value: str):
    return filter_csv(csvpath, column, value)

# Entry point for running the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
