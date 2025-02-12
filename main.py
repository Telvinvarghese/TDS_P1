from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import subprocess
import os
import json
import sys
import urllib.request
from urllib.parse import urlparse
import httpx
import uuid
import re

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
    try:
        print("Successfully rendering app")
        return JSONResponse(content={"message": "Successfully rendering app"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

API_KEY = os.getenv("AIPROXY_TOKEN")
BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1"

if not API_KEY:
    raise ValueError("API key missing")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

FORBIDDEN_PATTERNS = [
    r"rm\s", r"shutil\.rmtree", r"os\.remove", r"os\.rmdir",  # 🚨 Block file deletion
    r"\.\./", r"/etc/", r"/var/", r"/home/", r"/root/"  # 🚨 Block access outside `/data/`
]


def get_absolute_path(relative_path: str) -> str:
    return os.path.join(os.getcwd(), relative_path)
import json

def write_cost_response(response):
    """
    Writes the JSON content of an API response to a file.
    """
    try:
        response_json = response.json()  # ✅ Extract JSON data
        with open("cost_response.json", "w") as cost_file:
            json.dump(response_json, cost_file, indent=4)  # ✅ Now it's serializable
    except Exception as e:
        print(f"❌ Failed to write response: {e}")


def is_script_safe(script_code: str) -> bool:
    """Ensure the generated script does NOT contain forbidden operations."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, script_code):
            return False
    return True


async def parse_task_with_llm(task_description: str) -> str:
    """Generate a Python script for a given task, enforcing security constraints."""
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                BASE_URL + "/chat/completions",
                headers=headers,
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

            # Log cost usage (ensure this function exists)
            write_cost_response(response)

            # Extract the response
            script_code = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not script_code:
                raise ValueError("Empty response from LLM")

            # Security check
            if not is_script_safe(script_code):
                raise HTTPException(status_code=400, detail="Generated script contains unsafe operations.")

            return script_code

        except httpx.HTTPStatusError as http_err:
            raise HTTPException(status_code=http_err.response.status_code, detail=f"HTTP error: {http_err}")
        except Exception as e:
            print(f"Error parsing response: {str(e)}")  # Use logging in production
            raise HTTPException(status_code=500, detail="Error parsing LLM response")

def save_script(script_code: str) -> str:
    """Save the generated script securely to /data/."""
    filename = f"/data/task_{uuid.uuid4().hex}.py"
    with open(filename, "w") as script_file:
        script_file.write(script_code)
    return filename


def run_script(filename: str):
    """Execute the generated script safely and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            check=True
        )
        return {"status": "success", "output": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "output": e.stderr.strip()}
    except Exception as e:
        return {"status": "error", "output": str(e)}


def download_and_run_script(script_url: str, user_email: str):
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

@app.post("/run")
async def run_task(task: str = Query(..., description="Task description in plain English")):
    """Detects and runs a task. Calls `download_and_run_script()` if a URL and email are present, otherwise generates a script."""
    
    # ✅ Extract URL from task
    url_match = re.search(r"https?://[^\s]+\.py", task)
    
    # ✅ Extract email from task
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", task)

    if url_match and email_match:
        script_url = url_match.group(0)
        user_email = email_match.group(0)
        return download_and_run_script(script_url, user_email)  # ✅ Run script directly

    # 🚀 Default case: Generate a script for all other tasks
    try:
        script_code = await parse_task_with_llm(task)
        script_filename = save_script(script_code)
        result = run_script(script_filename)

        if result["status"] == "success":
            return result  # ✅ HTTP 200 OK

        raise HTTPException(status_code=400, detail=result["output"])  # ❌ HTTP 400 (Task error)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")  # ❌ HTTP 500 (Agent error)


@app.get("/read")
async def read_file(path: str = Query(..., description="Path to the file to read")):
    """Return the contents of a file for verification, ensuring only /data/ is accessible."""
    if not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Access to files outside /data is not allowed.")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        with open(path, "r") as file:
            content = file.read()
        return {"status": "success", "content": content}  # ✅ HTTP 200 OK
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
