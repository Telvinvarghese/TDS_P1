from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.responses import JSONResponse, Response  # type: ignore
from fastapi.middleware.cors import CORSMiddleware
import os

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

# run task
@app.post("/run")
async def run_task(task: str):
    if not task:
        raise HTTPException(
            status_code=400, detail="Task description is missing.")
    try:
        return JSONResponse(content={"message": "run task rendering"},status_code=200)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# read file
@app.get("/read")
async def read_file(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="File path is missing.")
    # Prevent directory traversal attacks
    if not path.startswith("/data"):
        raise HTTPException(
            status_code=400, detail="Invalid file path. Data outside /data can't be accessed or exfiltrated.")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        async with open(path, 'r') as file:
            content = await file.read()
            return JSONResponse(content, media_type="text/plain",status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
