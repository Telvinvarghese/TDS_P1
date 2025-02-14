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
    if is_english_string(user_input):
        return {"status": "success", "output": user_input}
    async with httpx.AsyncClient(timeout=10.0) as client:
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
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_prompts}, {"role": "user", "content": task_description}],
                    "temperature": 0.5,
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
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_prompts}, {"role": "user", "content": update_task}],
                    "temperature": 0.5,
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

async def generate_python_script1(task_description: str) -> str:
    """
    Uses GPT-4 to generate a Python script based on the task description.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_prompts}, {"role": "user", "content": task_description}],
                    "temperature": 0.5,
                }
            )
            write_cost_response(response)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text)
            response_data = response.json()
            inline_metadata_script = ""
            python_code = response_data.get("choices", [{}])[0].get(
                "message", {}).get("content", "").strip()
            script_path = save_script(inline_metadata_script, python_code)
            return [python_code, script_path]
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
    script_path = os.path.abspath(
        os.path.join(f"script_{uuid.uuid4().hex}.py"))
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
async def run_task(task: str = Query(..., description="Task description in plain English")):
    url_match = re.search(r"https?://[^\s]+\.py", task)
    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", task)
    if url_match and email_match:
        return await download_and_run_script(url_match.group(0), email_match.group(0))
    try:
        translated_task = await translate_to_english(task)
        task_description = translated_task["output"]
        response, script_path = await generate_python_script(task_description)
        execution_output = execute_script(script_path)
        retry_limit = 0
        while retry_limit < 2:
            execution_error = execution_output['error']
            if execution_error is None:  # Script executed successfully
                return {"status": "Success","response": response,"script_path": script_path,}
            # If there's an error, retry after extracting script contents
            with open(script_path, 'r') as f:
                python_code = f.read()
                response, script_path = await resend_request(task_description=task_description, python_code=python_code, error=execution_error)
            retry_limit += 1
            execution_output = execute_script(script_path)  # Retry execution
        # If it still fails after retries, return failure
        return {"status": "Failed","error": execution_output['error'],"script_path": script_path,}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/read")
async def read_file(path: str = Query(..., description="Path to the file to read")):
    if not path:
        raise HTTPException(status_code=400, detail="Path is empty.")
    if not path.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Access to files outside /data is not allowed.")
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
        raise HTTPException(
            status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
