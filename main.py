from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse
import urllib.request
from urllib.parse import urlparse
import subprocess
import os
import json
import csv
import sys
import httpx
import uuid
import re
import asyncio
from prompts import system_prompts
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def home():
    return JSONResponse(content={"message": "Successfully rendering app"})

API_KEY = os.getenv("AIPROXY_TOKEN")
if not API_KEY:
    raise ValueError("API key missing")

BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

FORBIDDEN_TASKS = [
    "hack", "exploit", "bypass", "steal", "ddos", "ransomware", "malware", "spy",
    "delete system files", "shutdown server", "access unauthorized data",
    "generate fake identity", "scrape personal data", "open ports", "send spam emails"
]

def is_valid_task(task_description: str) -> bool:
    """
    Check if the task is valid and does not contain restricted keywords.
    """
    task_description_lower = task_description.lower()

    # Check for forbidden keywords
    for forbidden_word in FORBIDDEN_TASKS:
        if forbidden_word in task_description_lower:
            return False

    # Check for nonsense or weird tasks
    if len(task_description.split()) < 2:  # Too short task
        return False
    if not any(c.isalpha() for c in task_description):  # No letters, only symbols/numbers
        return False

    return True

FORBIDDEN_PATTERNS = [
    # File deletion + shell execution
    r"(rm\s|shutil\.rmtree|os\.remove|os\.rmdir|subprocess\.)",
    r"(eval|exec)\(",  # Prevent arbitrary code execution
    r"(\.\./|/etc/|/var/|/home/|/root/)",  # Directory traversal
    # Sensitive file access
    r"open\s*\(\s*['\"]?(/etc/passwd|/etc/shadow|/var/log)['\"]?\s*\)",
]

def is_script_safe(script_code: str) -> bool:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, script_code):
            return False
    return True

def write_cost_response(response):
    try:
        response_json = response.json()
        with open("cost_response.json", "w") as cost_file:
            json.dump(response_json, cost_file, indent=4)
    except Exception as e:
        print(f"Failed to write response: {e}")

async def run_script(filename: str):
    try:
        process = await asyncio.create_subprocess_exec(sys.executable, filename,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        return {"status": "success" if process.returncode == 0 else "error", 
                "output": stdout.decode().strip() if process.returncode == 0 else stderr.decode().strip()}
    except asyncio.TimeoutError:
        process.kill()
        return {"status": "error", "output": "Execution timed out!"}
    except Exception as e:
        return {"status": "error", "output": str(e)}

async def download_and_run_script(script_url: str, user_email: str):
    print(f"Downloading and running script: {script_url}")
    script_name = os.path.basename(urlparse(script_url).path)
    script_path = os.path.abspath(os.path.join(script_name))

    try:
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "uvicorn"], check=True)
        if not os.path.exists(script_path):
            urllib.request.urlretrieve(script_url, script_path)
        os.chmod(script_path, 0o755)
        result = subprocess.run([sys.executable, script_path, user_email,
                                "--root=./data"], capture_output=True, text=True, check=True)
        return {"message": "Success!", "output": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"Script execution failed: {e.stderr.strip()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def is_english_string(text: str) -> bool:
    return bool(re.match(r"^[\x00-\x7F]+$", text))

async def translate_to_english(user_input: str) -> dict:
    print(f"Translating text to English: {user_input}")
    if is_english_string(user_input):
        return {"status": "success", "output": user_input}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Translate the following text to English. Return only the translated text, nothing else."},
                        {"role": "user", "content": user_input}
                    ],
                    "temperature": 0,
                }
            )
            write_cost_response(response)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text)
            response_data = response.json()
            translated_text = response_data.get("choices", [{}])[0].get(
                "message", {}).get("content", "").strip()
            return {"status": "success", "output": translated_text}
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError:
            raise HTTPException(
                status_code=503, detail="Translation service unavailable")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}")

response_format = {
    "type": "json_schema",
    "json_schema":
    {
        "name": "generate_python_script",
        "schema":
        {
            "type": "object",
            "required": ["python_dependencies", "python_code"],
            "properties":
            {
                "python_code":
                {
                    "type": "string",
                    "description": "Python code to perform the task."
                },
                "python_dependencies":
                {
                    "type": "array",
                    "items":
                    {
                        "type": "object",
                        "properties":
                        {
                            "module_name":
                            {
                                "type": "string",
                                "description": "Name of the Python module."
                            },
                        },
                        "required": ["module_name"],
                        "additionalProperties": False
                    }
                }
            }
        }
    }
}

async def generate_python_script(task_description: str) -> str:
    # Always start with a new conversation history
    conversation_history = [
        {"role": "system", "content": system_prompts},  # Keep system prompt
        {"role": "user", "content": task_description}   # Fresh user input
    ]
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": conversation_history,
                    "temperature": 0,
                    "response_format": response_format,
                }
            )
            write_cost_response(response)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text)
            response_data = response.json()
            response_content = json.loads(response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip())
            python_dependencies = response_content['python_dependencies']
            python_code = response_content['python_code']
            dependencies_str = ''.join(f'# "{dependency.get("module_name", "")}",\n'for dependency in python_dependencies if dependency.get("module_name"))
            inline_metadata_script =f"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
{dependencies_str}# ]
# ///
"""
            script_path = save_script(inline_metadata_script, python_code)
            return [response_data,script_path]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError:
            raise HTTPException(
                status_code=503, detail="Code generation service unavailable")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}")

async def resend_request(task_description: str, python_code: str, error: str) -> str:
    update_task="""
Update the Python code
{python_code}
-----
for below task
{task_description}
---
Based on Error encountered while running task
{error}
"""
    # Always start with a new conversation history
    conversation_history = [
        {"role": "system", "content": system_prompts},  # Keep system prompt
        {"role": "user", "content": update_task}   # Fresh user input
    ]
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": conversation_history,
                    "temperature": 0,
                    "response_format": response_format,
                }
            )
            write_cost_response(response)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text)
            response_data = response.json()
            response_content = json.loads(response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip())
            python_dependencies = response_content['python_dependencies']
            python_code = response_content['python_code']
            dependencies_str = ''.join(
                f'# "{dependency.get("module_name", "")}",\n'for dependency in python_dependencies if dependency.get("module_name"))
            inline_metadata_script = f"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
{dependencies_str}# ]
# ///
"""
            script_path = save_script(inline_metadata_script, python_code)
            return [response_data, script_path]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError:
            raise HTTPException(
                status_code=503, detail="Code generation service unavailable")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}")

def save_script(inline_metadata_script: str, python_code: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    script_path = os.path.abspath(f"script_{timestamp}.py")
    try:
        with open(script_path, "w") as f:
            f.write(inline_metadata_script)
            f.write(python_code)
        return script_path
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save script: {str(e)}")

def execute_script(script_path: str) -> dict:
    """
    Executes a Python script from the given path and returns a dictionary 
    containing 'output' and 'error'.
    """
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=15  # Prevent infinite loops
        )

        return {
            "output": result.stdout.strip() if result.stdout else None,
            "error": result.stderr.strip() if result.returncode != 0 else None,
            "exit_code": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {"output": None, "error": "Execution timed out!", "exit_code": -1}
    except Exception as e:
        return {"output": None, "error": str(e), "exit_code": -1}

@app.post("/run")
async def run_task(task: str = Query(..., description="Task description")):
    translated_task = await translate_to_english(task)
    task_description = translated_task["output"].strip().lower()

    greetings = ["hi", "hello", "hey", "good morning","good afternoon", "good evening"]

    if task_description in greetings:
        return {"status": "success", "message": "Hello! How can I assist you?"}

    # Check if the task is valid
    if not is_valid_task(task_description):
        raise HTTPException(
            status_code=400, detail="❌ Invalid or unsafe task. Please provide a meaningful and safe request.")

    # Extract script URL and email (if present)
    url_match = re.search(r"https?://[^\s]+\.py", task_description)
    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", task_description)

    script_url = url_match.group(0) if url_match else None
    email = email_match.group(0) if email_match else None

    # If a Python script URL is found, execute it with or without email
    if script_url and email:
        return await download_and_run_script(script_url, email)

    # Continue with normal task execution
    try:
        # Normalize spaces (remove extra spaces and replace multiple spaces with a single space)
        task_description = re.sub(r'\s+', ' ', task_description.strip())
        task_description = task_description.replace("```", "").replace("`", "").replace('"', "")
        instructions_for_task = await call_gpt(task_description)
        response, script_path = await generate_python_script(instructions_for_task)
        execution_output = execute_script(script_path)
        retry_limit = 1  # Allow up to 1 retries
        for _ in range(retry_limit):
            execution_error = execution_output.get('error')
            if not execution_error:
                response =  {
                    "status": "Success",
                    "task": task_description,
                    "instructions": instructions_for_task,
                    "script_path": script_path,
                    "message": "Task executed successfully"
                }
                return response
            # Retry if an error occurs
            with open(script_path, 'r') as f:
                python_code = f.read()

            response, script_path = await resend_request(
                task_description=instructions_for_task,
                python_code=python_code,
                error=execution_error
            )
            execution_output = execute_script(script_path)  # Retry execution

        response= {
            "status": "Fail",
            "task": task_description,
            "instructions": instructions_for_task,
            "script_path": script_path,
            "error": execution_output.get('error'),
            "message": "Task Execution Failed"
        }
        raise HTTPException(status_code=400, detail=response)

    except HTTPException as e:
        raise e  # Keep HTTP exception intact
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/read")
async def read_file(path: str = Query(..., description="Path to the file to read")):
    # Define the root directory we allow
    allowed_directory = Path("/data").resolve(strict=True)
    
    # Ensure we correctly form the path, stripping only the leading slash once
    requested_path = allowed_directory / path.lstrip("/")  # Strip only the leading slash
    
    logger.info(f"Requested file path: {requested_path}")
    
    try:
        # Resolve the file to its absolute path
        requested_path = requested_path.resolve(strict=True)
        logger.info(f"Resolved file path: {requested_path}")

        # Check if the resolved path is within the /data directory
        if not requested_path.startswith(allowed_directory):
            logger.error(f"Path is outside of allowed directory: {requested_path}")
            raise HTTPException(status_code=400, detail="Invalid file path, outside of /data/ directory.")
        
        if not requested_path.exists():
            logger.error(f"File not found: {requested_path}")
            raise HTTPException(status_code=404, detail="File not found.")
        
        # Check if it's a file (not a directory)
        if not requested_path.is_file():
            logger.error(f"Path is not a file: {requested_path}")
            raise HTTPException(status_code=400, detail="Path is not a file.")

        # Get the file extension
        file_extension = requested_path.suffix.lower()
        logger.info(f"File extension: {file_extension}")
        
        # Handle different file types
        if file_extension in [".txt", ".log", ".md", ".xml", ".yaml", ".yml", ".ini", ".conf", ".sql", ".bat", ".sh"]:
            with open(requested_path, "r", encoding="utf-8") as file:
                return PlainTextResponse(content=file.read())

        elif file_extension == ".json":
            with open(requested_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return JSONResponse(content=data)

        elif file_extension == ".csv":
            with open(requested_path, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                data = [row for row in reader]
            return JSONResponse(content={"data": data})

        elif file_extension in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            return FileResponse(requested_path, media_type=f"image/{file_extension.lstrip('.')}")

        else:
            return FileResponse(requested_path)

    except UnicodeDecodeError as e:
        logger.error(f"Unicode decoding error: {e}")
        raise HTTPException(status_code=500, detail="File encoding error.")
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise HTTPException(status_code=404, detail="File not found.")
    except OSError as e:
        logger.error(f"OS error: {e}")
        raise HTTPException(status_code=500, detail="OS error occurred while reading the file.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

async def call_gpt(task_description: str) -> str:
    os.makedirs("task_description", exist_ok=True)
    payload = {
    "model": "gpt-4o-mini",  # Ensure this model is valid
    "messages": [
        {"role": "system","content": f"Rewrite the task description by replacing 'LLM' with 'gpt-4o-mini.' Also, check for phrases like 'Only write' or 'Just write' and refine them for better clarity. Finally, simplify the task description into clear and concise English while preserving its original meaning.: '{task_description}'"},
        {"role": "user","content": task_description}]
            ,"temperature": 0}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
            response.raise_for_status()
            content = response.json().get("choices", [{}])[
                0].get("message", {}).get("content", "")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            write_to_file(f"task_description/task_instructions_{timestamp}.txt", content)
            return content
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
