from dotenv import load_dotenv
import os
# importing Gemini #step1
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 
# importing Document Loader step 2
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
#splitting

#calling chroma db
from langchain_chroma import Chroma
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import regex as re

#import from prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

#loads key from .env file
load_dotenv()


# Load embedding models with caching
# Load models with caching
@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# getting key through env file
GOOGLE_API = os.getenv('GOOGLE_API_KEY')

import('pysqlite3')
import sys
#for sqlite3 error that arises after we use chroma or crewai library
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


#Initializing Gemini Model
def generate_response(GOOGLE_API):
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.7, google_api_key=GOOGLE_API)
    return gemini_model

def set_reader(document_raw):
    with open(document_raw.name, mode='wb') as w:
        w.write(document_raw.getvalue())
    if document_raw: # check if path is not None
        loader = PyPDFLoader(document_raw.name)
        document = loader.load()
    return document

def cleaning_document(document):
    for i in range(len(document)):
        document[i].page_content = re.sub("[^A-Za-z0-9.\']+", ' ', document[i].page_content)
    
    return document

def split_embed(document):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    splits = text_splitter.split_documents(document)

    embedding_function = load_embedding_model()
    document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])

    return document_embeddings,splits

def get_vectordb(splits):
    collection_name = "my_collection"
    vectorstore = Chroma.from_documents(collection_name=collection_name, documents=splits, embedding=load_embedding_model(), persist_directory="./chroma_db")
    return vectorstore

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_prompt_template(retriever,gemini_model):
    template = """you're a helpful assisstant. you will have some context to answer. You can also use your knowledge to
    assist answering the user's queries as best as possible. Make sure the answer is structured and detailed.
    Don't give wrong asnwers\n
    context:{context}

    Question: {question}

    Answer: """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()}
    | prompt
    | gemini_model
    | StrOutputParser()
    )
    return rag_chain
