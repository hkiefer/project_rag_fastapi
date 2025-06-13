#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from fastapi import Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

import uvicorn

#import costum llm and helper modules
from src.document_loader import *
from src.embedding import *
from src.llm_rag import *

import warnings
warnings.filterwarnings("ignore")

def clean_result(result_str: str) -> str:
    """
    Helper function to remove markdown code block markers (e.g., ```json and ```) from a result string.
    """
    # Remove starting markdown marker if present
    if result_str.startswith("```json"):
        result_str = result_str[len("```json"):].strip()
    # Remove ending triple backticks if present
    if result_str.endswith("```"):
        result_str = result_str[:-3].strip()
    return result_str

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load environment variables and check API key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

#For test purposes only
#if openai_api_key:
    #print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
#else:
    #print("OpenAI API Key not set")

# Initialize the RAG system
qa_chain = initialize_rag_system()

# Set up the FastAPI app
app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/query")
async def query_endpoint(question: str):
    """
    Endpoint to query the RAG system.
    Example: GET /query?question="Your question here"
    """
    try:
        output = qa_chain(question)
        result_str = output.get("result", "")
        result_str = clean_result(result_str)
        try:
            result_json = json.loads(result_str)
        except Exception:
            result_json = {"answer": result_str}
        return JSONResponse(content=result_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    """
    Render the HTML form for submitting a question to the RAG system.

    This route serves a simple webpage with an input field where users can type a question.
    The submitted question will be processed via the POST request.
    """
    return templates.TemplateResponse("form.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, question: str = Form(...)):
    """
    Handle the submitted question from the HTML form and return the RAG-generated answer.

    Parameters:
    - request (Request): The HTTP request object.
    - question (str): The user-submitted question from the form.

    Returns:
    - HTMLResponse with the generated answer or error message.
    """
    try:
        output = qa_chain(question)
        result_str = output.get("result", "")
        result_str = clean_result(result_str)
        try:
            result_json = json.loads(result_str)
            answer = result_json.get("answer", result_str)
        except Exception:
            answer = result_str
        return templates.TemplateResponse("form.html", {"request": request, "result": answer})
    except Exception as e:
        return templates.TemplateResponse("form.html", {"request": request, "result": f"Error: {str(e)}"})


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
