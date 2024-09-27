# %% Setting up Llama3 using GROQ API
# from cd import txt_to_docx
import tiktoken
import matplotlib.pyplot as plt
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import lark
from langchain_core.documents import Document
import os
import ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
# from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from typing import List, Optional, Any
import time
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
# 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
# GROQ_API_KEY = "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"
# llm = ChatGroq(temperature=0, model="Llama3-8b-8192",
#                groq_api_key=GROQ_API_KEY)
# Add the LLM downloaded from Ollama
ollama_llm = "llama3"
# llm = ChatOllama(model=ollama_llm)

# %% # Directory path where your files are located and adding metadata to the documents

directory = r'C:\Users\i\Desktop\llm\ncc_doc\publish\FAQs'

def load_knowledge_base(directory):
    # List to store Document objects
    documents = []

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Assuming all files are text files
            file_path = os.path.join(directory, filename)

            # Read content from the file
            with open(file_path, 'r', encoding='utf-8') as file:
                page_content = file.read()

            # Create Document object and append to list
            doc = Document(
                page_content=page_content,
            )
            documents.append(doc)

    # print(f"This is the data type of returned value {type(documents)}")
    # print(f"This is the number of document in the list {len(documents)}")
    return documents

documents = load_knowledge_base(directory=directory)


# %% Showing the frequency of token count in the document


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def token_count(func, documents):
    # Doc texts
    docs_texts = [d.page_content for d in documents]

    # Calculate the number of tokens for each document
    counts = [func(d, "cl100k_base") for d in docs_texts]

    # Plotting the histogram of token counts
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
    plt.title("Histogram of Token Counts")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Display the histogram
    # return plt.show

token_count(num_tokens_from_string, documents)
# %% Number of total context count in the document
# Doc texts concat
# d_sorted = sorted(documents, key=lambda x: x.metadata["year"])
# d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in documents]
)

print(
    "Num tokens in all context: %s"
    % num_tokens_from_string(concatenated_content, "cl100k_base")
)
# print(
#     f"{concatenated_content}"
# )
# %% Splitting the text into chunks
def text_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter

text_splitter = text_splitter(3000, 300)
texts_split = text_splitter.create_documents([concatenated_content])
# print(len(texts_split))

# File path where the output will be written
file_path = "output.txt"

# Open the file in write mode ('w')
# If the file doesn't exist, it will be created
# If the file exists, its content will be overwritten
# with open(file_path, 'w', encoding='utf-8') as file:
#     file.write(str(texts_split))

# txt_file = 'output.txt'
# docx_file = 'output.docx'
# txt_to_docx(txt_file, docx_file)
# print(f"Content written to {docx_file}")
#%%
# texts_split[37]
#%%
import pickle
# with open('bge-large-en-v1.5.pkl', 'rb') as file:
#     embedding = pickle.load(file)

embedding = OllamaEmbeddings(model="mxbai-embed-large")

# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyA1exXlH0Mc6dYTBb3Afjo4QmcMLq_PUE0')
#%%
print("Creating vector database")
start_time = time.time()
print(f"Start time is: {start_time}")
# Embed and store the texts
persist_directory = r'C:\Users\i\Desktop\llm\ncc_doc\ncc_chatbot\my-app\packages\pirate-speak\pirate_speak\vertordb_mx'

# Now, use all_texts to build the vectorstore with Chroma
vectorstore = Chroma.from_documents(
    documents=texts_split, embedding=embedding, persist_directory=persist_directory)

end_time = time.time()
print(f"Time Taken is: {end_time - start_time}")
print("Done creating Vector database")

# %%
# Using FAISS vectorDB


