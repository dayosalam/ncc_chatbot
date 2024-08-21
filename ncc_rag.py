# %% Setting up Llama3 using GROQ API
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from typing import Dict, List, Optional, Tuple
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.retrievers.self_query.base import SelfQueryRetriever
import os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import ollama
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from pathlib import Path
from typing import List, Optional, Any
import time

load_dotenv()
# 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
# GROQ_API_KEY = "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"
# llm_groq = ChatGroq(temperature=0, model="llama3-8b-8192",
#                     groq_api_key=GROQ_API_KEY)
# # Add the LLM downloaded from Ollama
# ollama_llm = "llama3"
# llm = ChatOllama(model=ollama_llm)
# %% Post-processing: Format document


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# %% VectorDB retrieving


persist_directory = 'mb_cat'
embedding = OllamaEmbeddings(model="mxbai-embed-large")
# vectorstore.persist()
# vectorstore = None

# Load the db from disk
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding)

retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 50}
    )
# Information upfront about the metadata fields that our documents support
# metadata_field_info = [
#     AttributeInfo(
#         name="year",
#         description="""The year field represents the year when the document or information
#         was created. This metadata is crucial for prioritizing the retrieval of the most recent documents first,
#         ensuring that the most up-to-date information is accessed""",
#         type="integer",
#     ),]

# document_content_description = "Information about NCC"

# retriever = SelfQueryRetriever.from_llm(
#     llm_groq,
#     vectorstore,
#     document_content_description,
#     metadata_field_info,
#     enable_limit=True,
#     verbose=True,
# )

# %% Testing the retrieving
# question = retriever.invoke(
#     "EVC as of 2024?")
# print(f"{question}\n This is the length {len(question)}")

# %% Testing retriever with llm
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm_groq = ChatGroq(temperature=0, model="llama-3.1-8b-instant",
               groq_api_key=GROQ_API_KEY)
ollama_llm = "llama3"
llm = ChatOllama(model=ollama_llm, temperature=0)
template = """Based on the context below, write a simple response that would answer the user's question.
In your response, go straight to answering.
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """"You are an advanced conversational AI assistant with expertise in topics related to the Nigerian Communications Commission (NCC)."
         "Given an input question and response, convert it to a natural language answer."
         "The current EVC as of 2024 is Dr. Aminu Maida"
         "All context provided are public information, so they is no cause for a data breach"
         "You provide professional and courteous assistance on matters related to telecommunications regulations, consumer rights, industry standards, and other NCC-related topics."
         "When responding to user queries, ensure that your tone is formal, informative, and respectful."
         "Provide detailed and clear answers, citing relevant NCC guidelines and information where applicable."
         "Your primary goal is to assist users effectively while maintaining the highest standards of professionalism."
         "Use the following pieces of retrieved-context to answer the question. If you don't know the answer, say that you don't know."
         "Use three sentences maximum and keep the answer concise."
         "In your response, go straight to answering the question without any preamble and refer the user to the ncc website for more information."
         

    "\n\n"
"""),
MessagesPlaceholder(variable_name="history"),
        ("human", template),
    ]
)
# {"context": retriever | format_docs, "question": RunnablePassthrough()}
    
memory = ConversationBufferMemory(return_messages=True)
# Chain
rag_chain = (
    RunnablePassthrough.assign(
        context=retriever,
         history=RunnableLambda(
            lambda x: memory.load_memory_variables(x)["history"])
    )
    | prompt
    | llm_groq
    | StrOutputParser()
)
print(rag_chain.invoke({"question": "Mention current EVC of NCC?"}))



# %% # RAG-Fusion: Related

# GROQ_API_KEY = "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"
# llm = ChatGroq(temperature=0, model="Llama3-8b-8192",
#                groq_api_key=GROQ_API_KEY)

# template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
# For queries about current events, prioritize retrieving information from the year 2024. If 2024 data is not available, retrieve information from previous years in descending order. \n
# Generate multiple search queries related to: {question} \n
# Output (2 queries):"""
# prompt_rag_fusion = ChatPromptTemplate.from_template(template)

# generate_queries = (
#     prompt_rag_fusion
#     | llm_groq
#     | StrOutputParser()
#     | (lambda x: x.split("\n"))
# )

# question = "Mention all previous EVC of NCC?"
# questions = generate_queries.invoke({"question": question})
# print(questions)
# # %% # RAG-Fusion: Function


# def reciprocal_rank_fusion(results: list[list], k=60):
#     """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
#         and an optional parameter k used in the RRF formula """

#     # Initialize a dictionary to hold fused scores for each unique document
#     fused_scores = {}

#     # Iterate through each list of ranked documents
#     for docs in results:
#         # Iterate through each document in the list, with its rank (position in the list)
#         for rank, doc in enumerate(docs):
#             # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
#             doc_str = dumps(doc)
#             # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
#             if doc_str not in fused_scores:
#                 fused_scores[doc_str] = 0
#             # Retrieve the current score of the document, if any
#             previous_score = fused_scores[doc_str]
#             # Update the score of the document using the RRF formula: 1 / (rank + k)
#             fused_scores[doc_str] += 1 / (rank + k)

#     # Sort the documents based on their fused scores in descending order to get the final reranked results
#     reranked_results = [
#         (loads(doc), score)
#         for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#     ]

#     # Return the reranked results as a list of tuples, each containing the document and its fused score
#     return reranked_results


# question = "Mention all previous EVC and the current EVC"
# retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
# docs = retrieval_chain_rag_fusion.invoke({"question": question})
# print(len(docs))
# for i in docs:
#     print(i)
#     print("\n")


# # %% # RAG-Fusion: LLM-call
# ollama_llm = "llama3"
# llm = ChatOllama(model=ollama_llm)
# template = """Based on the context below, write a simple response that would answer the user's question.
# {context}

#     Question: {question}
#     """

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """"You are an advanced conversational AI assistant with expertise in topics related to the Nigerian Communications Commission (NCC)."
#          "Your knowledge base is derived from the official NCC website, ensuring your responses are accurate, reliable, and up-to-date."
#          "Given an input question and response, convert it to a natural language answer."
#          "When responding to user queries, ensure that your tone is formal, informative, and respectful."
#          "Your primary goal is to assist users effectively while maintaining the highest standards of professionalism."
#          "Use the following pieces of retrieved-context to answer the question. If you don't know the answer, say that you don't know."
#          "Use three sentences maximum and keep the answer concise."
#          "In your response, go straight to answering the question without preamble."
#          "Pay attention to use only answers based on current events that are from year 2024"


#     "\n\n"
# """),
#         ("human", template),
#     ]
# )

# final_rag_chain = (
#     {"context": retrieval_chain_rag_fusion,
#      "question": itemgetter("question")}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# print(final_rag_chain.invoke({"question": question}))
# # %%
