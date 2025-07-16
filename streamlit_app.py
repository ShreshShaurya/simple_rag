import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 


# setting up the page streamlit
st.set_page_config(
    page_title="RAG App ", layout="wide", page_icon="./images/langchain.png"
)

st.title("💬 RAG Chatbot")
st.caption("🚀 A chatbot powered by Gemini")

#loads key from .env file
load_dotenv()

#uploading document
document = st.sidebar.file_uploader("Upload the pdf",key="document_key")
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", key="file_qa_api_key", type="password")

if document is not None:
    st.toast(f'Successfully uploaded 🎉{document.name}')

# Load embedding models with caching
@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# #Initializing Gemini Model
if GOOGLE_API_KEY:
    print("Hi")
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=GOOGLE_API_KEY)

    #model = generate_response(GOOGLE_API_KEY)
    print(gemini_model.invoke("Come up with 10 names for a song about parrots"))

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": f"You have uploaded a pdf file named {document}"}]