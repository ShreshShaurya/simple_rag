from dotenv import load_dotenv
import os
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 

import streamlit as st

#loads key from .env file
load_dotenv()


# Load embedding models with caching
@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# #Initializing Gemini Model
# def generate_response(GOOGLE_API_KEY):
#     gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.7, google_api_key=GOOGLE_API_KEY)
#     return gemini_model

# getting key through env file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
print(gemini_model.invoke("Come up with 10 names for a song about parrots"))