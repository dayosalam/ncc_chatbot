# %% Setting up Llama3 using GROQ API
from pirate_speak.helper_functions import *
import tiktoken
import matplotlib.pyplot as plt
from langchain_core.documents import Document
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import time
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)

# Initialize embedding model with a specified model name
embedding = OllamaEmbeddings(model="mxbai-embed-large")

# Define directories for storing vector database and loading document files
persist_directory = r'C:\Users\i\Desktop\llm\ncc_doc\ncc_chatbot\my-app\packages\pirate-speak\pirate_speak\vertordb_mx'
directory = r'C:\Users\i\Desktop\llm\ncc_doc\publish\FAQs'

# %%
# Function to load all .txt files from a specified directory and create Document objects
def load_knowledge_base(directory):
    """Loads documents from a directory and returns a list of Document objects."""
    
    documents = []  # List to store Document objects

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(directory, filename)
            
            # Attempt to read file content and create a Document object
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    page_content = file.read()
                doc = Document(page_content=page_content)
                documents.append(doc)  # Add Document object to the list
            
            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping this file.")
            except IOError:
                print(f"Error reading file {file_path}. Skipping this file.")
    
    return documents  # Return list of documents

# Function to count tokens in a string using a specified encoding
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to plot a histogram of token counts for documents
def token_count(func, documents):
    """Calculates token counts for each document and plots a histogram of token counts."""
    
    # Extract page content from each document
    docs_texts = [d.page_content for d in documents]
    # Count tokens in each document's text
    counts = [func(d, "cl100k_base") for d in docs_texts]
    
    # Plot the histogram of token counts
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
    plt.title("Histogram of Token Counts")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

# Function to calculate total tokens in concatenated documents
def show_num_tokens(documents):
    """Concatenates all document texts and prints total token count."""
    
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in documents]
    )
    print(
        "Num tokens in all context: %s"
        % num_tokens_from_string(concatenated_content, "cl100k_base")
    )
    return concatenated_content  # Return concatenated text

# Function to split text into chunks based on a specified chunk size and overlap
def text_splitter(chunk_size, chunk_overlap):
    """Creates a RecursiveCharacterTextSplitter with specified chunk size and overlap."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    return text_splitter  # Return text splitter object

# Function to write split text chunks to a .docx file
def write_to_docx(texts_split):
    """Writes split texts to a .txt file and then converts it to .docx format."""
    
    file_path = "output.txt"  # Define output file path
    
    # Attempt to write split texts to a .txt file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(str(texts_split))
    except IOError:
        print(f"Error writing to file {file_path}. Check file permissions or path.")
        return  # Exit function if write fails
    
    # Attempt to convert .txt file to .docx format
    try:
        txt_file = 'output.txt'
        docx_file = 'output.docx'
        txt_to_docx(txt_file, docx_file)
        print(f"Content written to {docx_file}")
    except IOError:
        print(f"Error converting {txt_file} to .docx format.")

# Function to create embeddings for document texts and store them in a vector database
def embedding_document(embedding_model, persist_directory):
    """Embeds documents and stores them in a vector database."""
    
    print("Creating vector database")
    start_time = time.time()
    print(f"Start time is: {start_time}")
    
    # Create and persist vectorstore using Chroma
    try:
        vectorstore = Chroma.from_documents(
            documents=texts_split, embedding=embedding_model, persist_directory=persist_directory)
        end_time = time.time()
        print(f"Time Taken is: {end_time - start_time}")
        print("Done creating Vector database")
        return vectorstore  # Return the created vectorstore
    except IOError:
        print(f"Error creating or saving vector database at {persist_directory}")
        return None

# %%
# Load documents from the specified directory
documents = load_knowledge_base(directory=directory)

# Plot the token count histogram for loaded documents
token_count(num_tokens_from_string, documents)

# Split the text of the loaded documents into chunks with specified parameters
text_splitter = text_splitter(6000, 600)
texts_split = text_splitter.create_documents([show_num_tokens(documents)])
print(f"Number of documents split into: {len(texts_split)}")

# Write the split texts to a .docx file (uncomment if you are debugging)
write_to_docx(texts_split)

# Embedding the documents into the vectorstore (uncomment to execute)
# vectorstore = embedding_document(embedding, persist_directory)

