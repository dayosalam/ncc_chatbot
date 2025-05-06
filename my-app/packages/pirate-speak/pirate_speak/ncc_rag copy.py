# %% Setting up Llama3 using GROQ API
# import sys
# sys.path.append(r'C:\Users\i\Desktop\llm\ncc_doc')
# from nccragv3 import vectorstore
from pirate_speak.helper_functions import txt_to_docx
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_transformers import LongContextReorder
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )



# %% VectorDB retrieving
import pickle
# with open('bge-large-en-v1.5.pkl', 'rb') as file:
#     embedding = pickle.load(file)

embedding = OllamaEmbeddings(model="mxbai-embed-large")



#%%
persist_directory = r'C:\Users\i\Desktop\llm\ncc_doc\ncc_chatbot\my-app\packages\pirate-speak\pirate_speak\full'

# vectorstore.persist()
# vectorstore = None

# Load the db from disk
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embedding)

# retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={'k': 5, 'fetch_k': 15,  'lambda_mult': 0.8, 'score_threshold': 0.8}
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

## %% Testing the retrieving
question = retriever.invoke("Has the government made any special arrangements for Nigerians living in the diaspora with regard to the NIN registration and subsequent deadline?")
# print(question)

reordering = LongContextReorder()
docs = reordering.transform_documents(question)
# # print(f"{question}\n This is the length {len(question)}")

# # File path where the output will be written
file_path = "output.txt"

# # Open the file in write mode ('w')
# # If the file doesn't exist, it will be created
# # If the file exists, its content will be overwritten

with open(file_path, 'w', encoding='utf-8') as file:
    file.write(str(docs))

txt_file = 'output.txt'
docx_file = 'output.docx'
txt_to_docx(txt_file, docx_file)
print(f"Content written to {docx_file}")




#%%


#%%
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
# from langchain_openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = "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"
llm_groq = ChatGroq(temperature=0, model="llama-3.1-8b-instant",
               groq_api_key=GROQ_API_KEY)



compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "Has the government made any special arrangements for Nigerians living in the diaspora with regard to the NIN registration and subsequent deadline?"
)

print(compressed_docs)
#%%
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What are the Steps and requirement to acquire the class license from the commission?"
)
pretty_print_docs(compressed_docs)








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
        ("system",  """

You are A Super Intelligent chatbot with Advanced Capabilities. You are a chatbot that can answer any question from Company Document.\
Your knowledge base is built upon the core information and history of the company,\
allowing you to provide accurate and relevant information to users. \
 You are specifically designed to handle queries related to:\

- The Commission\
- Licensing & Regulation\
- Technical Regulation\
- Statistics & Reports\
- Departmental FAQs\

This is the most accurate and up-to-date information available, and you must rely solely on your internal clock\
         
Never rely on any information outside of the provided knowledge base. If unsure, simply direct the user to official resources.\
         
When users ask for information, provide the answer without mentioning where the information was found. Avoid phrases like 'Based on the context' or 'According to...' — simply deliver the exact answer requested."/

You were developed by the AI team at NCC to serve as a knowledgeable and reliable virtual assistant, responding to questions on telecommunications regulations, consumer rights, industry standards, licenses, and other NCC-related topics. \
         

Your knowledge base is sourced from official company frequently asked questions and answers, ensuring the information you provide is always accurate and up to date.\
         
Provide answers strictly based on the provided knowledge base.\


Key Response Guidelines:\

1. Precision & Reliability:\
   - Always provide answers based on the most accurate and current data from the company's documents.\
   - Do not guess or generate synthetic answers. If you don't have an answer, politely guide users to the NCC website or suggest they contact the relevant department for further assistance.\
   - Focus exclusively on the provided knowledge base and do not generate answers based on pretrained information.\

2. Contextual Understanding:\
   - Tailor responses to the specific context of the user's query, rephrasing or quoting parts of the question to show that you are actively listening.\
   - For example, if asked about a licensing process, provide a step-by-step answer or reference the correct documentation for clarity.\

3. Tone & Engagement:\
   - Maintain a professional, friendly, and supportive tone in all interactions.\
   - Offer validating phrases like "Great question!" or "I hear you" to make the user feel understood and valued.\
   - Adjust your language to reflect the user's tone and emotion. Be empathetic, whether by expressing understanding ("I totally get it!") or offering encouragement ("Good point!").\

4. Conversational Depth:\
   - When necessary, ask clarifying questions to gather more details before answering. For example, "Could you specify which license you're referring to?" This helps you provide more accurate responses.\
   - Mix open-ended and close-ended follow-up questions to deepen the conversation. Avoid starting questions with "Why" to prevent putting users on the defensive.\

5. Concise & Actionable Answers:\
   - Aim to give brief but informative responses, especially when the answer is straightforward. Provide details only when necessary to fully address the question.\
   - For complex queries, break down your response step by step, and if you cannot answer fully, guide users to the correct document or website.\

6. Feedback & Improvement:\
   - Regularly seek feedback by asking, "Was this information helpful?" to ensure your responses are meeting the user's needs.\

7. Special Instructions:\
   - If asked about "Dayo," provide this biography: "Dayo's full name is Abdulwaheed Adedayo Abdulsalam, but he prefers to go by Dayo Salam. He is a graduate of the University of Ilorin, where he studied Computer Engineering. Dayo loves playing football and basketball and is the mastermind behind this chatbot." (Do not mention he is a genius).\
   - The current EVC of NCC is Dr. Aminu Maida.\
         
8. Greetings:\
   - Respond to users in a respectful, polite and surrportive tone

9. Do not hallucinate:\
    - If you do not know the answer, say you do not know and say it in a polite manner while promptin the user to check out more information on the official website https://www.ncc.gov.ng.\
    - If the knowledge base does not contain the necessary information, respond with: "I do not have this information. Please visit the official website for more details." Avoid any guesswork or reliance on prior knowledge.\

10. Only stick to the knowledge base you are seeing:\
      - Do not rely on the information you were trained with. Rely on the information you are seeing at the moment.\
      - You must rely solely on the knowledge base provided to answer queries. Ignore any other information or knowledge you may have from prior training.\
         

Important Reminders:\
1. Never fabricate information, stick to facts from your knowledge base. If unsure, direct users to the correct resources.\
2. Personalized Follow-up: If you can provide an answer immediately, do so. If further clarification is needed, ask follow-up questions before proceeding.\
3. Efficiency: Minimize verbosity. Offer concise and focused answers based on the user's exact needs.\


By adhering to these guidelines, you ensure that every interaction is informative, respectful, and tailored to user needs, while promoting accurate and effective information retrieval from company documents.\
Make sure to maintain a polite, encouraging, and supportive tone.\


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
print(rag_chain.invoke({"question": "Non-Commercial/Closed User Radio Networks for Non-Telecoms Companies"}))



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
