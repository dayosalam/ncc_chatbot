# NCC Organizational Chatbot (RAG-powered)

This project is an **organizational chatbot** built to intelligently respond to employee and stakeholder queries using internal documents and FAQs as its knowledge base. It leverages the **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, context-aware answers drawn from the organization's proprietary data.

## Overview

Many organizations face challenges in making internal documentation easily accessible. This chatbot addresses that by enabling natural language interactions with company-specific knowledge. It retrieves relevant content from a document store and augments it with a large language model (LLM) to produce responses.

## Tech Stack

- **Backend:** Flask
- **LLM Integration:** Llama models
- **Document Retrieval:** RAG (Retrieval-Augmented Generation)
- **Vector Store & Embedding:** Chroma + Sentence Transformers (or similar)
- **Directory Structure:** Windows-based local environment

## How It Works

1. Internal documents were collected from the organization and used to create the chatbot's knowledge base.
2. These documents were embedded and stored in a vector database.
3. When a user sends a query, the chatbot:
   - Embeds the query.
   - Retrieves the most relevant chunks of data using similarity search.
   - Sends the context along with the query to the language model to generate a final answer.


## Getting Started

### Prerequisites

- Python 3.8+
- `pip` for installing dependencies

### Installation

```bash
pip install -r requirements.txt

cd my-app\app
python server.py


 ## Author
Developed by Dayo as part of the AI initiative at the Nigerian Communications Commission.

## Notes
1. The chatbot was designed for internal use only.

2. Data privacy and document security were maintained throughout.

3. The RAG setup can be adapted to other organizations with similar needs.

## Contact
For questions or collaboration, feel free to reach out.

---

## Project Structure

```plaintext
ncc_chatbot/
│
├── my-app/
│   ├── app/                      ← Flask backend code
│   └── packages/
│       └── pirate-speak/
│           └── pirate_speak/
│               └── nccragv3.py  ← RAG implementation (embedding, vector store logic)

