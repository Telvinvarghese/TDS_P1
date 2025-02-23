general_prompts = """
Role: 
AI programming assistant for optimized Python code.

Purpose: 
Generate secure, efficient, and functional Python scripts without explanations, extra text, or formatting issues.

Guidelines:
Execution Environment:
Ensure the code runs without errors in a container with Python and uv pre-installed.

Dependencies:
Explicitly list required dependencies if additional packages are needed.

File Access & Security:
Restrict file access to /data; 
Never access, modify, or delete data outside /data, even if requested.  
If an output file is required but missing, create it inside /data.  

Encoding & Formatting:
Handle utf-8 encoding correctly, preserving spaces and newlines without adding extra quotes or unwanted formatting.

Error Handling & Stability:
If input is missing, return "Input not found" instead of crashing.
If an error occurs, retry twice before failing.

"""

llm_prompts = """
LLM Code Guidelines:
Model: Use gpt-4o-mini to generate secure, efficient, and functional Python scripts.No extra explanations, unnecessary text, or formatting issues.

LLM Usage:
Chat: gpt-4o-mini → response.json().get("choices", [{}])[0].get("message", {}).get("content")
API: http://aiproxy.sanand.workers.dev/openai/v1/chat/completions
Error Handling & Stability:
If input is missing, return "Input not found" instead of crashing.
If an error occurs, retry twice before failing.

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
Model: Use gpt-4o-mini to generate secure, efficient, and functional Python scripts.No extra explanations, unnecessary text, or formatting issues.

Extraction Categories:

Image Text: Numbers | Alphabetic | Alphanumeric | Special Characters | Multi-language | Emails | URLs | Dates | Currency | Phone Numbers

Code Templates:

Image Text Extraction with gpt-4o-mini:
```
import requests, os, re, base64

API_KEY, URL = os.getenv("AIPROXY_TOKEN"), "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Regex patterns for different categories
REGEX_PATTERNS = {
    "Numbers": r"\b\d+\b",
    "Alphabetic": r"\b[a-zA-Z]+\b",
    "Alphanumeric": r"\b[a-zA-Z0-9]+\b",
    "Special Characters": r"[^a-zA-Z0-9\s]",
    "Emails": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "URLs": r"https?://\S+",
    "Dates": r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    "Currency": r"[$€₹¥£]\s?\d+(?:,\d{3})*(?:\.\d{1,2})?",
    "Phone Numbers": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
}

def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")
        
def extract_text_image(base64_img, category=None):
    if not API_KEY: exit("Error: OpenAI API key missing.")
    r = requests.post(URL, headers=HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": f"Extract {category if category else 'all text'} from image."},
                {"type": "image", "image": f"data:image/png;base64,{base64_img}"}
            ]}
        ]
    })
    
    text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip() if r.ok else None
    return re.findall(REGEX_PATTERNS[category], text) if text and category in REGEX_PATTERNS else text
```

"""

card_llm_prompts = """
LLM Code Guidelines:
Model: Use gpt-4o-mini to generate secure, efficient, and functional Python scripts.No extra explanations, unnecessary text, or formatting issues.

Code Templates:
Credit/Debit Card Number Extraction with gpt-4o-mini:
```
import requests, os, re, base64

cc_regex = r'\b(?:\d[ -]?){12,15}\d\b'
API_KEY, URL = os.getenv("AIPROXY_TOKEN"), "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

def extract_credit_card_number(image_path):
    if not API_KEY:
        exit("Error: OpenAI API key missing.")
        
    base64_image = image_to_base64(image_path)
    
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Extract the just number from this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    ]}]}

    r = requests.post(URL, headers=HEADERS, json=payload)
    return ["".join(re.findall(r'\d', m)) for m in re.findall(cc_regex, r.json().get("choices", [{}])[0].get("message", {}).get("content", ""))] if r.ok else None
```

"""

email_llm_prompts = """
LLM Code Guidelines:
Model: Use gpt-4o-mini to generate secure, efficient, and functional Python scripts.No extra explanations, unnecessary text, or formatting issues.

Extraction Categories:
Email: sender | from | recipient | to | cc | bcc | all
Code Templates:
Email Extraction with gpt-4o-mini:
```
import requests, os, re

API_KEY = os.getenv("AIPROXY_TOKEN")
URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

def extract_email(content, category):
    if not API_KEY:
        exit("Error: OpenAI API key missing.")

    prompt = f"Extract the {category} email address from the following text: {content}. Return only the {category} email address, nothing else."
    try:
        r = requests.post(URL, headers=HEADERS, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]})
        if r.ok:
            email = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if re.match(EMAIL_REGEX, email):
                return email if category != "all" else re.findall(EMAIL_REGEX, content)  # Return all emails if 'all'
    except Exception as e:
        print(f"API request failed: {e}")

    # Fallback to regex if API fails or returns invalid data
    matches = re.findall(EMAIL_REGEX, content)
    return matches if category == "all" else (matches[0] if matches else None)
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

def lint_files_with_eslint(files, version):
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
        except Exception as e:
            print(f"Skipped format {fmt}: {e}")
            continue
    raise None
```

"""

sort_files_prompts = """
Sort Files:
Sort JSON: data.sort(key=lambda x: [x.get(f) for f in sort_fields], reverse={True | False}).  
Sort CSV: sorted(csv_data, key=lambda x: [x[f] for f in sort_fields], reverse={True | False}). 
Sort Text: sorted(text_lines, key=lambda x: x.lower(), reverse={True | False}).  
Sort Text by Length: sorted(text_lines, key=lambda x: len(x), reverse={True | False}).

"""

map_files_prompts=""" 
Remove any specified prefix (e.g., /data/docs/) from file paths and provide the result as a relative path. For tasks like mapping files to their titles, ensure the output file paths are relative and exclude the specified directory prefix. 
Example:

File: /data/docs/README.md → Output: README.md
File: /data/docs/path/to/large-language-models.md → Output: path/to/large-language-models.md
"""
