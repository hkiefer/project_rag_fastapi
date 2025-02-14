# LLM-RAG-App for Answering Questions from Simple Documents.

This project employs a simple Retrieve-and-Generate (RAG) system that leverages a Large Language Model (LLM), a vector database, and REST API to answer queries based on one given pdf- and json-document (```test.pdf``` and ```test.json``` in resources).

## Installation and Setup

To setup the project, clone the repository and create and launch a clean virtual environment:

```sh
python3.12 -m venv venv
source venv/bin/activate
```

Make sure, Python3 is installed on your local machine.

Install the dependencies using the following command:

```sh
pip3 install --prefer-binary -r requirements.txt
```

Also generate a .env file with the following variable:

```sh
OPENAI_API_KEY='your_OpenAI_api_key'
```

Here, you can create your key:

https://platform.openai.com/api-keys

## Usage

Run the project via this command:

```sh
python3 app.py
```

To submit a query, visit a URL such as:

```sh
http://localhost:8000/query?question="When is the travel starting?"
```
The system should respond with a JSON object structured as follows:

```json
{
  "answer": "Extracted answer from the document",
  "Page number": "Relevant page number from the document",
  "Additional Metadata": "Other relevant information"
}
```
