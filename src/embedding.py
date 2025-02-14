#!/usr/bin/env python3
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

def create_vector_store(chunks: list) -> Chroma:
    """Create a Chroma vector store from document chunks."""
    db_name = "./chroma_db"
    embedding_model = OpenAIEmbeddings()

    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embedding_model).delete_collection()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_name
    )
    vector_db.persist()

    #For test purposes only
    #print(f"Embeddings stored in ChromaDB with {vector_db._collection.count()} documents")

    vector_db = Chroma(
        persist_directory=db_name,
        embedding_function=embedding_model
    )
    return vector_db