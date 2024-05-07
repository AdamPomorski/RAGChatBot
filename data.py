import getpass
import os

OPENAI_API_KEY = ''
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from pprint import pprint
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

PINECONE_API_KEY = ''
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

pc = Pinecone()

loader = PyPDFLoader("https://www.warta.pl/documents/UFK/Warunki_ubezpieczen/WN2/WN2_-_OWU_WARTA_NIERUCHOMOSCI.pdf", extract_images=True)
docs = loader.load()


#separators=["§","\n\n", "\n", " ", ""]

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=10, separators=["\n§"])
docs = text_splitter.split_documents(docs)

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
index_name = "proj2"
pc.delete_index(index_name)
pc.create_index(
        index_name,
        dimension=1536,
        metric='euclidean',
        spec=spec
    )

embeddings = OpenAIEmbeddings()
index_name = "proj2"
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name, namespace="ns1")


for i, d in enumerate(docs):
    print(f"\n## Document {i}\n")
    print(d.page_content)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

query = "Jakie są rodzaje opłat?"

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

qa.run(query)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st


store = {}

window_memory = ConversationBufferWindowMemory(k=2)


contextualize_q_system_prompt = """Mając historię rozmowy oraz najnowsze pytanie użytkownika, \
które może odnosić się do kontekstu w historii rozmowy, sformułuj samodzielne pytanie, które będzie zrozumiałe bez historii rozmowy. \
NIE odpowiadaj na pytanie, po prostu je sformułuj w razie potrzeby i w przeciwnym razie zwróć je takie, jakie jest."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever = docsearch.as_retriever()

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """Jesteś asystentem do zadań związanych z odpowiadaniem na pytania. \
Wykorzystaj następujące fragmenty odzyskanego kontekstu, aby odpowiedzieć na pytanie. \
Jeśli otrzymasz puste pytanie, wypisz to: :) \
Jeśli nie znasz odpowiedzi, po prostu powiedz, że nie wiesz. Odpowiedz maksymalnie trzema zdaniami i zachowaj odpowiedź zwięzłą.\"

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

