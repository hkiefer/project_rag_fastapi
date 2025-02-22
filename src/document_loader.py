#!/usr/bin/env python3
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#need this to suppress warnings from pypdf for my specific file
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

import os

def load_documents() -> list:
    """Load documents from a PDF and a JSON file."""

    #needed to load the file from a subfolder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, '..', 'resources', 'test.pdf')
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_documents = pdf_loader.load()

    #For test purposes only
    #print(f"Loaded {len(pdf_documents)} PDF documents")

    #needed to load the file from a subfolder
    json_path = os.path.join(current_dir, '..', 'resources', 'test.json')
    json_loader = JSONLoader(file_path=json_path, jq_schema=".", text_content=False)
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