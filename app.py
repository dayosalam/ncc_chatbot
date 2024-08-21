from flask import Flask, request, jsonify
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel
import ollama
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import os
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import lark
from langchain_community.document_transformers import LongContextReorder
from ncc_rag import rag_chain
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
# Add your RAG setup code here

# This function will be invoked when the chatbot sends a message


@app.route('/')
def index():
    return "Server is running!"


@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        data = request.json
        question = data.get('message')
        # chat_history = data.get('history', [])
        start_time = time.time()
        print(f"Start time is: {start_time}")
        print(f"This is my data coming from the frontend {data}")
        # Invoke your chain with the question and history
        response = rag_chain.invoke(
            {"question": question})

        # Add the question and response to chat history
        # chat_history.append({"role": "user", "content": question})
        # chat_history.append({"role": "assistant", "content": response.content})
        print(f"This is my data sending to the frontend {response}")
        end_time = time.time()
        print(f"Time Taken is: {end_time-start_time}")
        return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
