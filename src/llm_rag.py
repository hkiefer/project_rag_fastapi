#!/usr/bin/env python3

# Import LangChain and Chroma components (use the community versions, since it gives me DeprecationWarning)
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.document_loader import *
from src.embedding import *

def create_qa_chain(vector_db: Chroma) -> RetrievalQA:
    """Create the RetrievalQA chain with a custom prompt template."""
    
    prompt_template = """
        You are an assistant that can only use the provided context to answer questions.
        Do not use any external or prior knowledge. If the answer is not available in the context, respond with "Information not available in the provided document."

        Context:
        {context}

        Question:
        {question}

        Please provide your answer as a JSON object with the following keys:
        - "answer": The answer extracted from the document.
        - "Page number": The page number(s) where the answer is found.
        - "Additional Metadata": Any other relevant metadata.

        JSON
        """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def initialize_rag_system() -> RetrievalQA:
    """Load documents, split them to chunks, create embeddings, and build the QA chain."""
    all_documents = load_documents()
    chunks = split_documents(all_documents)
    vector_db = create_vector_store(chunks)
    qa_chain = create_qa_chain(vector_db)
    return qa_chain
