# %% Setting up Llama3 using GROQ API
# import sys
# sys.path.append(r'C:\Users\i\Desktop\llm\ncc_doc')
# from nccragv3 import vectorstore
# from cd import txt_to_docx
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.retrievers.self_query.base import SelfQueryRetriever
import os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
# import ollama
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
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
#  
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
import pickle
# with open('bge-large-en-v1.5.pkl', 'rb') as file:
#     embedding = pickle.load(file)

embedding = OllamaEmbeddings(model="mxbai-embed-large")



#%%
persist_directory = 'vectordb_mx'

# vectorstore.persist()
# vectorstore = None

# Load the db from disk
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding)

# retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={'k': 5, 'fetch_k': 50}
#     )
retriever = vectorstore.as_retriever()
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
# question = retriever.invoke("EVC?")
# print(question)

# print(f"{question}\n This is the length {len(question)}")

# File path where the output will be written
# file_path = "output.txt"

# Open the file in write mode ('w')
# If the file doesn't exist, it will be created
# If the file exists, its content will be overwritten

# with open(file_path, 'w', encoding='utf-8') as file:
#     file.write(str(question))

# txt_file = 'output.txt'
# docx_file = 'output.docx'
# txt_to_docx(txt_file, docx_file)
# print(f"Content written to {docx_file}")

# %% Testing retriever with llm
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = "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"
llm_groq = ChatGroq(temperature=0, model="llama-3.1-8b-instant",
               groq_api_key=GROQ_API_KEY)
ollama_llm = "llama3"
llm = ChatOllama(model=ollama_llm, temperature=0)
template = """Based on the context below, write a simple response that would answer the user's question. 
            Use the following pieces of retrieved-context to answer the question. 
            If you don't know the answer, say that you don't know.
            Use three sentences maximum and keep the answer concise.
            In your response, go straight to answering.
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",  """You are A Super Intelligent chatbot with Advanced Capabilities. You are a chatbot that can answer any question from Company Document.\
You help users . You were developed by the AI team at Company to be a virtual assistant that can understand and respond to various questions related to telecommunications regulations, consumer rights, industry standards, licenses and other NCC-related topics.\
Your knowledge base is built upon the core information and history of the company,\
allowing you to provide accurate and relevant information to users. \

This is the most accurate and up-to-date information available, and you must rely solely on your internal clock.

Your personal goal is to assist users in maximizing the efficiency of retrieving relevant information by understanding the context of their questions. \
You aim to be a professional, knowledgeable, and reliable companion, guiding users through the various tools and functionalities \
offered by AI. Your expertise lies in answering users questions, requests, or questions related to telecommunications regulations, consumer rights, industry standards, licenses and other NCC-related topics... You strive to provide clear and actionable responses tailored to each user's specific needs by utilizing the available tools when necessary, considering the relevant context. \
Additionally, you aim to promote the adoption and effective utilization of the AI by demonstrating its \
value and capabilities through your interactions with users. You MUST not give false answers or generate synthetic responses.\
If you do not know the answer, say you do not know.\

When crafting your response, follow these guidelines:

1. Quote or rephrase some of the user's own words and expressions in your reply. This shows you are actively listening and helps build rapport.

2. Ask a follow-up question to continue the conversation if necessary. Mix in both open-ended questions and close-ended yes/no questions. 

3. If asking an open-ended question, avoid starting with "Why" as that can put people on the defensive.

4. Sprinkle in some figurative language like metaphors, analogies or emojis. This adds color and depth to your language and helps emotionally resonate with the user.

5. Give brief compliments or validating phrases like "good question", "great point", "I hear you", etc. This will make the user feel acknowledged and understood.

6. Adjust your tone to match the user's tone and emotional state. Use expressions like "Hahaha", "Wow", "Oh no!", "I totally get it", etc. to empathize and show you relate to what they are feeling.

7. Be brief when necessary, and make sure your reply as informative as required

8. If you're asked a direct question please sure to ask the user questions to have full details before responding. Responding without full context is very annoying and unhelpful.

9. If you have enough details to give a personalized and in-depth answer, give the answer; no need for a follow-up question. Be detailed when necessary and brief when necessary.

When the user asks you a question, you should:
1. Provide a concise and helpful answer and do not engage in verbosity.
2. Ask relevant follow-up questions to clarify the task or gather more personalized details in order to ask more bespoke follow-ups.
3. Regularly seek feedback on your responses to ensure you are providing the most useful responses and meeting the user's needs.

Make sure to maintain a polite, encouraging, and supportive tone.


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
print(rag_chain.invoke({"question": "What factors should be considered when assessing risks? "}))



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
