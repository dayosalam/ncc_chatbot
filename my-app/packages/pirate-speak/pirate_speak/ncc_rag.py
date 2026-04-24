import os

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from pirate_speak.helper_functions import prompt_ncc, template_ncc
from pirate_speak.vector_store import build_retriever

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def build_rag_chain():
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY.")

    llm_groq = ChatGroq(
        temperature=0,
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
    )
    knowledge_retriever = build_retriever(search_k=3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_ncc),
            MessagesPlaceholder(variable_name="history"),
            ("human", template_ncc),
        ]
    )
    memory = ConversationBufferMemory(return_messages=True)

    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["question"]) | knowledge_retriever,
            history=RunnableLambda(
                lambda x: memory.load_memory_variables(x)["history"]
            ),
        )
        | prompt
        | llm_groq
        | StrOutputParser()
    )


rag_chain = build_rag_chain()
