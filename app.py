#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

#import llm and helper modules
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


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
