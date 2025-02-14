#!/usr/bin/env python3
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#need this to suppress warnings from pypdf for my specific file
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

import sys
sys.path.append('../resources')

def load_documents() -> list:
    """Load documents from a PDF and a JSON file."""
    # Get the directory of the current module
    pdf_loader = PyPDFLoader("test.pdf")
    pdf_documents = pdf_loader.load()

    #For test purposes only
    #print(f"Loaded {len(pdf_documents)} PDF documents")

    json_loader = JSONLoader(file_path="test.json", jq_schema=".", text_content=False)
    json_documents = json_loader.load()

    #For test purposes only
    #print(f"Loaded {len(json_documents)} JSON documents")

    return pdf_documents + json_documents


def split_documents(all_documents: list) -> list:
    """Split documents into chunks using a recursive text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_documents)

    #For test purposes only
    #print(f"Total chunks created: {len(chunks)}")
    return chunks