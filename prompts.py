general_prompts = """
Role: AI programming assistant for optimized Python code.
Purpose: Generate secure, efficient, and functional Python scripts with best practice

Guidelines
Execution: Runs in a container with Python & uv pre-installed.
Packages: Explicitly list dependencies.
File Access: Restrict to /data, no external modification
Output Handling
If an output file is required but does not exist, create it inside /data.
utf-8, raw text, preserve spaces/newlines, no extra quotes.
Errors: Return "Input not found" for missing input, retry twice on errors.
Compliance: No sensitive data, ethical handling

"""

llm_prompts = """
LLM Code Guidelines:
Model: Use gpt-4o-mini for efficient, readable Python code.
Processing: utf-8 encoding,handle csv, json, and txt formats.
Files: Access only /data
Compliance: No sensitive data, ethical handling

LLM Usage:

Chat: gpt-4o-mini → response.json()["choices"][0]["message"]["content"]
Embedding: text-embedding-3-small → response.json()["data"][0]["embedding"]
API: http://aiproxy.sanand.workers.dev/openai/v1/
Error Handling: Retry on failure, handle errors gracefully.

LLM Call Structure (gpt-4o-mini)
````
import requests, os

API_KEY, URL = os.getenv("AIPROXY_TOKEN"), "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def call_llm_api(payload):
    if not API_KEY: exit("Error: OpenAI API key missing.")
    try:
        return requests.post(URL, headers=HEADERS, json=payload).json().get("choices", [{}])[0].get("message", {}).get("content")
    except requests.RequestException:
        return None
```

"""

embedding_llm_prompts = """
Similar Text: Use text-embedding-3-small, cosine similarity np.dot(embeddings, embeddings.T).

Use the following structure for Similar Text LLM calls with text-embedding-3-small:
```
import requests, os

API_KEY, URL = os.getenv("AIPROXY_TOKEN"),  "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

with open(input_file, encoding="utf-8") as f:
    texts = [line.strip() for line in f]
def get_embedding(texts):
    if not API_KEY: exit("Error: OpenAI API key missing.")
    r = requests.post(URL, headers=HEADERS, json={"model": "text-embedding-3-small", "input": texts})
    return r.json().get("data", [{}])[0].get("embedding") if r.ok else None
```

"""

image_llm_prompts = """
LLM Code Guidelines:
Model: Use gpt-4o-mini for efficient, readable Python code.
Processing: utf-8 encoding,handle csv, json, and txt formats.
Files: Access only /data
Compliance: No sensitive data, ethical handling

Extraction Categories:

Image Text: Numbers | Alphabetic | Alphanumeric | Special Characters | Multi-language | Emails | URLs | Dates | Currency | Phone Numbers | Card Numbers

Code Templates:

Image Text Extraction with gpt-4o-mini:
```
import requests, os

API_KEY, URL = os.getenv("AIPROXY_TOKEN"), "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def extract_text_image(base64_img, category):
    if not API_KEY: exit("Error: OpenAI API key missing.")
    r = requests.post(URL, headers=HEADERS, json={"model": "gpt-4o-mini", "messages": [
        {"role": "user", "content": [{"type": "text", "text": f"Extract {category} from image."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}]}]})
    return r.json().get("choices", [{}])[0].get("message", {}).get("content") if r.ok else None
```

"""

card_llm_prompts = """
LLM Code Guidelines:
Model: Use gpt-4o-mini for efficient, readable Python code.
Processing: utf-8 encoding,handle csv, json, and txt formats.
Files: Access only /data
Compliance: No sensitive data, ethical handling

Code Templates:
Credit/Debit Card Number Extraction with gpt-4o-mini:
```
import requests, os

API_KEY, URL = os.getenv("AIPROXY_TOKEN"), "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def extract_credit_card_number(base64_image):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "You are given an image containing a text. Extract the number from the image"},
                        {"type": "image_url", "image_url": {"url":  f"data:image/png;base64,{base64_image}"}}]
        }]
    }
    response = requests.post(URL, headers=HEADERS, json=payload)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content") if response.ok else None
```

"""

email_llm_prompts = """
LLM Code Guidelines:
Model: Use gpt-4o-mini for efficient, readable Python code.
Processing: utf-8 encoding,handle csv, json, and txt formats.
Files: Access only /data
Compliance: No sensitive data, ethical handling

Extraction Categories:
Email: sender | recipient | both | all
Code Templates:
Email Extraction with gpt-4o-mini:
```
import requests, os

API_KEY, URL = os.getenv("AIPROXY_TOKEN"), "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def extract_email(content, category):
    if not API_KEY: exit("Error: OpenAI API key missing.")
    r = requests.post(URL, headers=HEADERS, json={"model": "gpt-4o-mini", "messages": [
        {"role": "user", "content": f"Extract the {category} email address from: {content}"}]})
    return r.json().get("choices", [{}])[0].get("message", {}).get("content") if r.ok else None
```

"""

prettier_prompts = """
Format Files:
```
import subprocess

def format_files_with_prettier(files, version):
    subprocess.run(["npx", f"prettier@{version}", "--write", *files], check=True)

# Example usage
format_files_with_prettier(["file.js", "file.css"])
```

"""

black_prompts = """
Format Files:
```
import subprocess

def format_files_with_black(files,version):
    subprocess.run(["pip", "install", f"black=={version}"])
    subprocess.run(["black", *files], check=True)

# Example usage
format_files_with_black(["script.py"])
```

"""

eslint_prompts = """
Format Files:
```
import subprocess

def lint_files_with_eslint(files, version="latest"):
    subprocess.run(["npx", f"eslint@{version}", "--fix", *files], check=True)


# Example usage
lint_files_with_eslint(["file.js", "file.ts"])
```

"""

date_format_prompts = """
```
import datetime
# Comprehensive list of date formats
FORMATS = [
    "%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y",
    "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y.%m.%d", "%d.%m.%Y",
    "%b %d, %Y", "%d %B %Y", "%B %d, %Y", "%A, %B %d, %Y",
    "%I:%M %p, %d-%b-%Y", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%a %b %d %H:%M:%S %Y", "%A %d %B %Y", "%d %B %Y %H:%M",
    "%b %d %Y", "%b %d, %Y %I:%M %p", "%Y %b %d", "%Y %B %d",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ"
]

def parse_date(date_str):
    for fmt in FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {date_str}")
```

"""

sort_files_prompts = """
Sort Files:
Sort JSON: data.sort(key=lambda x: [x.get(f) for f in sort_fields], reverse={True | False}).  
Sort CSV: sorted(csv_data, key=lambda x: [x[f] for f in sort_fields], reverse={True | False}). 
Sort Text: sorted(text_lines, key=lambda x: x.lower(), reverse={True | False}).  
Sort Text by Length: sorted(text_lines, key=lambda x: len(x), reverse={True | False}).

"""

map_files_prompts = """
maps each filename (without the /data/docs/ prefix) -> maps each filepath (without the /data/docs/ prefix)

"""
