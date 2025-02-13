from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import urllib.request
from urllib.parse import urlparse
import subprocess
import os
import json
import sys
import httpx
import uuid
import re
import asyncio
from prompts1 import system_prompts
from pathlib import Path

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
async def root():
    return JSONResponse(content={"message": "Successfully rendering app"})

# ✅ API Key Validation
API_KEY = os.getenv("AIPROXY_TOKEN")
if not API_KEY:
    raise ValueError("API key missing")

BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

FORBIDDEN_PATTERNS = [
    r"(rm\s|shutil\.rmtree|os\.remove|os\.rmdir|subprocess\.)",  # File deletion + shell execution
    r"(eval|exec)\(",  # Prevent arbitrary code execution
    r"(\.\./|/etc/|/var/|/home/|/root/)",  # Directory traversal
    r"open\s*\(\s*['\"]?(/etc/passwd|/etc/shadow|/var/log)['\"]?\s*\)",  # Sensitive file access
]

# ✅ Ensuring security of generated scripts
def is_script_safe(script_code: str) -> bool:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, script_code):
            return False
    return True

# ✅ Writing cost response (No need for async)
def write_cost_response(response):
    try:
        response_json = response.json()
        with open("cost_response.json", "w") as cost_file:
            json.dump(response_json, cost_file, indent=4)
    except Exception as e:
        print(f"❌ Failed to write response: {e}")

async def run_script(filename: str):
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, filename,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return {"status": "success", "output": stdout.decode().strip()}
        else:
            return {"status": "error", "output": stderr.decode().strip()}
    except Exception as e:
        return {"status": "error", "output": str(e)}

async def download_and_run_script(script_url: str, user_email: str):
        script_name = os.path.basename(urlparse(script_url).path)
        script_path = os.path.abspath(os.path.join(script_name))
    
    try:
        try:
            import uvicorn
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn"], check=True)
        if not os.path.exists(script_path):
            urllib.request.urlretrieve(script_url, script_path)
        os.chmod(script_path, 0o755)
        result = subprocess.run([sys.executable, script_path, user_email,"--root=./data"], capture_output=True, text=True, check=True)
        return {"message": "Success!", "output": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script execution failed: {e.stderr.strip()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def is_english_string(text: str) -> bool:
    return bool(re.match(r"^[\x00-\x7F]+$", text))

async def translate_to_english(user_input: str) -> dict:
    if is_english_string(user_input):
        return {"status": "success", "output": user_input}
    async with httpx.AsyncClient(timeout=20.0) as client:
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
                raise HTTPException(status_code=response.status_code, detail=response.text)
            response_data = response.json()
            translated_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return {"status": "success", "output": translated_text}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Translation service unavailable")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def generate_python_script(task_description: str) -> str:
    """
    Uses GPT-4 to generate a Python script based on the task description.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_prompts},{"role": "user", "content": task_description}],
                    "temperature": 0.5,
                }
            )
            write_cost_response(response)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            response_data = response.json()
            script_code = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return script_code
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Code generation service unavailable")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def save_script(script_code: str) -> str:
        script_path = os.path.abspath(os.path.join(f"script_{uuid.uuid4().hex}.py"))
    try:
        with open(script_path, "w") as f:
            f.write(script_code)
        return script_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save script: {str(e)}")

def execute_script(script_path: str) -> str:
    """
    Executes a Python script from the given path and returns its output.
    """
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=20  # Prevent infinite loops
        )
        return (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return "Execution timed out!"
    except Exception as e:
        return f"Execution error: {str(e)}"
    
@app.post("/run")
async def run_task(task: str = Query(..., description="Task description in plain English")):
    url_match = re.search(r"https?://[^\s]+\.py", task)
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", task)
    
    if url_match and email_match:
        return await download_and_run_script(url_match.group(0), email_match.group(0))

    try:
        translated_task = await translate_to_english(task)
        task_description = translated_task["output"]
        
        max_retries = 3  # Maximum retries if script execution fails
        for attempt in range(max_retries):
            script_code = await generate_python_script(task_description)
            script_path = save_script(script_code)
            execution_output = execute_script(script_path)
            
            if "error" not in execution_output.lower():  # Check if execution was successful
                return {
                    "status": "Success",
                    "script_path": script_path,
                    "output": execution_output
                }
            
            print(f"⚠️ Execution failed, retrying... ({attempt + 1}/{max_retries})")

        # If all retries fail, return an error response
        raise HTTPException(status_code=500, detail="Script execution failed after multiple attempts.")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
        
@app.get("/read")
async def read_file(path: str = Query(..., description="Path to the file to read")):
    if not path:
        raise HTTPException(status_code=400, detail="Path is empty.")
    if not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Access to files outside /data is not allowed.")
    path = os.path.abspath(os.path.join("./data", path.lstrip("/data/")))
    print(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        with open(path, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
