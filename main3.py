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

# ✅ Parse task with LLM (Corrected `async` handling)
async def parse_task_with_llm(task_description: str) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.post(
                    BASE_URL + "/chat/completions",
                    headers=HEADERS,
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "Generate a Python script that strictly operates only in /data/."},
                            {"role": "user", "content": task_description}
                        ],
                        "temperature": 0.2
                    }
                )
                response.raise_for_status()
                
                if response.status_code == 429:  # Too Many Requests
                    wait_time = (2 ** attempt)  # Exponential backoff (2, 4, 8 sec)
                    await asyncio.sleep(wait_time)
                    continue  # Retry

                response_json = response.json()
                if 'choices' not in response_json or not response_json['choices']:
                    raise ValueError("Invalid response: 'choices' key is missing or empty")

                script_code = response_json['choices'][0].get('message', {}).get('content', "").strip()
                if not script_code or not is_script_safe(script_code):
                    raise HTTPException(status_code=400, detail="Generated script contains unsafe operations.")

                return script_code

            except httpx.HTTPStatusError as http_err:
                if http_err.response.status_code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue  # Retry
                raise HTTPException(status_code=http_err.response.status_code, detail=f"HTTP error: {http_err}")
            except httpx.TimeoutException:
                raise HTTPException(status_code=408, detail="Request timed out")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {str(e)}")

    raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")


# ✅ Save script securely
import tempfile
def save_script(script_code: str) -> str:
    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, f"task_{uuid.uuid4().hex}.py")
    with open(filename, "w") as script_file:
        script_file.write(script_code)
    return filename

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
    """Download a script from a URL, save it in /tmp, and execute it."""
    
    script_name = os.path.basename(urlparse(script_url).path)
    script_path = os.path.abspath(os.path.join("/tmp", script_name))

    try:
        # ✅ Ensure `uvicorn` is installed
        try:
            import uvicorn
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn"], check=True)

        # ✅ Download the script safely
        if not os.path.exists(script_path):
            urllib.request.urlretrieve(script_url, script_path)

        # ✅ Ensure script has execute permissions
        os.chmod(script_path, 0o755)

        # ✅ Run script with email argument
        result = subprocess.run([sys.executable, script_path, user_email], capture_output=True, text=True, check=True)
        
        return {"message": "Success!", "output": result.stdout.strip()}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script execution failed: {e.stderr.strip()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ✅ `/run` endpoint (Fixed missing `await`)
@app.post("/run")
async def run_task(task: str = Query(..., description="Task description in plain English")):
    url_match = re.search(r"https?://[^\s]+\.py", task)
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", task)

    if url_match and email_match:
        return await download_and_run_script(url_match.group(0), email_match.group(0))

    try:
        script_code = await parse_task_with_llm(task)
        script_filename = save_script(script_code)
        result = await run_script(script_filename)

        if result["status"] == "success":
            return result

        raise HTTPException(status_code=400, detail=result["output"])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

# ✅ `/read` endpoint (Unchanged)
@app.get("/read")
async def read_file(path: str = Query(..., description="Path to the file to read")):
    if not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Access to files outside /data is not allowed.")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        with open(path, "r") as file:
            content = file.read()
        return {"status": "success", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
