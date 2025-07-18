import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 

from pipeline import generate_response, set_reader,cleaning_document,split_embed,get_vectordb,get_prompt_template


# setting up the page streamlit
st.set_page_config(
    page_title="RAG App ", layout="wide", page_icon="./images/langchain.png"
)

st.title("💬 RAG Chatbot")
st.caption("🚀 A chatbot powered by Gemini")

#loads key from .env file
load_dotenv()


#uploading document
document_raw = st.sidebar.file_uploader("Upload the pdf",key="document_key", type="pdf")
#puttin Gemini Key
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", key="file_qa_api_key", type="password")


#creating toggle for search or generating ans
search = st.toggle("Search Document")

#successful upload
if document_raw is not None: 
    st.toast(f'Successfully uploaded 🎉{document_raw.name}')
    
#checking Gemini
if not GOOGLE_API_KEY:
    st.error("No Gemini Key")
    st.stop()
elif len(GOOGLE_API_KEY)<39:
    st.error("API key is invalid, length less than 30")


with st.form("chat"):
    if document_raw:
        text = st.text_input("Write your question down below:", f"Can you summmarize the pdf {document_raw.name}?")
    else:
        text_unnamed = st.text_input("Write your question down below:", f"Come up with 10 famous taylor swift song")

    submitted = st.form_submit_button("Submit")
    #Initializing Gemini Model
    #gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=GOOGLE_API_KEY)

    if submitted and GOOGLE_API_KEY and not document_raw:
        gemini_model = generate_response(GOOGLE_API_KEY)
        response = gemini_model.invoke(text_unnamed)
        output_parser = StrOutputParser() #load o/p parser
        st.info(output_parser.invoke(response))

    if submitted and GOOGLE_API_KEY and document_raw:
        document = set_reader(document_raw)
        st.info(f"Total Pages in the pdf {len(document)}")
        cleaned_document = cleaning_document(document)

        document_embeddings,splits = split_embed(cleaned_document)
        st.info(f"Total embeddings {len(document_embeddings)}")

        vectorstore = get_vectordb(splits)

        if not search:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            gemini_model = generate_response(GOOGLE_API_KEY) #calling the model

            rag_chain = get_prompt_template(retriever,gemini_model)
            
            response = rag_chain.invoke(text)
            st.write(response)

        else:
            search_results = vectorstore.similarity_search(text, k=2)
            for i, result in enumerate(search_results, 1):
                st.write(f"Result {i}:")
                st.write(f"Content: {result.page_content}")
        

    
