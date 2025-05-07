#%% Import helper functions from pirate_speak module
from pirate_speak.helper_functions import template_ncc, prompt_ncc, txt_to_docx
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

# Load environment variables from a .env file (e.g., API keys)
load_dotenv()

# Load the GROQ API key from environment variables or hardcode for testing
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Initialize a large language model (LLM) with a specific temperature and model using the GROQ API
llm_groq = ChatGroq(temperature=0, model="llama-3.3-70b-versatile",
                    groq_api_key=GROQ_API_KEY)

# Set up embedding model for vector-based text search
embedding = OllamaEmbeddings(model="mxbai-embed-large")
# Directory path to save and load the vector database
persist_directory = r'C:\Users\i\Desktop\llm\ncc_doc\ncc_chatbot\my-app\packages\pirate-speak\pirate_speak\vertordb_mx'
# Define the initial question or query to process
question = "Non-Commercial/Closed User Radio Networks for Non-Telecoms Companies"

# %%
# Post-processing function to format document contents for readability
def format_docs(docs):
    """Formats document contents into a readable string format."""
    return "\n\n".join(doc.page_content for doc in docs)


# Function to initialize a retriever using the embedding model and vector database
def retriever(embedding_model, persist_directory):
    """
    Loads a vector store from disk and creates a retriever.
    
    Args:
        embedding_model: The model used for embeddings.
        persist_directory: Directory path where the vector store is saved.
        
    Returns:
        A retriever object to search the vector store.
    """
    # Initialize vector store from the specified directory with the embedding model
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embedding)
    # Set up a retriever to return the top 3 results for each search
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    return retriever


# Function to test the retriever's ability to answer questions
def debugging_testing(retriever, question):
    """
    Tests the retriever function by invoking it with a given question.
    
    Args:
        retriever: The retriever object to search the vector store.
        question: The query to retrieve relevant documents.
        
    Returns:
        The result from the retriever.
    """
    # Use the retriever to get the answer for the question
    result = retriever.invoke(question)

    # Print the retrieved result and its length
    print(f"{result}\n This is the length {len(result)}")
    return result


# Function to write the retrieved result to a text file and convert it to .docx format
def write_to_docx(result):
    """
    Writes the retrieval result to a .txt file and converts it to .docx format.
    
    Args:
        result: The retrieval result to be written.
    """
    file_path = "output.txt"  # Define output file path
    
    # Attempt to write split texts to a .txt file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(str(result))
    except IOError:
        # Handle error if writing to file fails
        print(f"Error writing to file {file_path}. Check file permissions or path.")
        return  # Exit function if write fails
    
    # Attempt to convert .txt file to .docx format
    try:
        txt_file = 'output.txt'
        docx_file = 'output.docx'
        txt_to_docx(txt_file, docx_file)
        print(f"Content written to {docx_file}")
    except IOError:
        # Handle error if conversion to .docx fails
        print(f"Error converting {txt_file} to .docx format.")

#%%
# Instantiate the retriever with the embedding model and persistent directory
retriever = retriever(embedding, persist_directory)
# Test the retriever with a question and save the result(uncomment when debugging)
# result = debugging_testing(retriever, question) 
# write_to_docx(result)  # Uncomment when debugging to enable writing to .docx


# %% This section constructs a prompt and retrieves the response

# Use a predefined template for the prompt
template = template_ncc

# Define a conversational prompt structure with message history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_ncc),          # System message template
        MessagesPlaceholder(variable_name="history"),  # Placeholder for conversation history
        ("human", template),             # Human message template
    ]
)

# Initialize memory to store the conversation history
memory = ConversationBufferMemory(return_messages=True)

# Create a chain of runnables to execute the retrieval and LLM response steps
rag_chain = (
    RunnablePassthrough.assign(
        context=retriever,  # Set retriever context
        history=RunnableLambda(
            lambda x: memory.load_memory_variables(x)["history"])  # Load conversation history
    )
    | prompt         # Process prompt
    | llm_groq       # Generate LLM response
    | StrOutputParser()  # Parse the output as a string
)

# Invoke the retrieval and generation process with the initial question
# print(rag_chain.invoke({"question": question}))


