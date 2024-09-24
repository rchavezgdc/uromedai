# Libraries
import os
import time
from dotenv import load_dotenv
import yaml
from yaml.loader import SafeLoader
import markdown
import streamlit as st
import streamlit_authenticator as stauth
from PyPDF2 import PdfReader
# Langchain Libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# Google Generative AI
import google.generativeai as genai

# Load environment variables and assign to constants
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
EMBEDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))
PDF_DIR_NAME = os.getenv("PDF_DIR_NAME")
# Setup Google GenerativeAI
genai.configure(api_key=GOOGLE_API_KEY)
# Load YAML configuration
with open('./config/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
# Setup Streamlit Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized'],
    auto_hash=True
)
# List all files in the content folder
def list_files():
    files = ["./"+PDF_DIR_NAME+"/"+f for f in os.listdir(PDF_DIR_NAME)]
    return files

# Read PDF file and return text content
def read_pdf(file_path):
    print("Reading text...")
    extracted_text = ""
    with open(file_path, "rb") as pdf_content:
        try:
            # Process the PDF using PyPDF2
            pdf_reader = PdfReader(pdf_content)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()
        except Exception as e: 
            print(f"Warning: Could not read PDF file '{file_path}' (might be encrypted or corrupted) Exception: {e}")
    return extracted_text

# Get Text Chunks
def get_text_chunks(text):
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    text_chunks = text_splitter.split_text(text)
    print("Splitting done, number of chunks: ", len(text_chunks))
    return text_chunks

# Convert Chunks to Vectors and save to local
def convert_chunks_to_vectors(chunks):
    print("Converting chunks to vectors...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDINGS_MODEL)
    if not os.path.exists("faiss_store"):
        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings_model,
        )
        vector_store.save_local("faiss_store")
    else:
        vector_store = FAISS.load_local(folder_path="faiss_store", 
                                        embeddings=embeddings_model,
                                        allow_dangerous_deserialization=True)
        vector_store.add_texts(chunks)
        vector_store.save_local("faiss_store")
    

# Get Conversational Chain and create the prompt
def get_conversational_chain():
    prompt_template = """
    [System Instructions]: You are a health assistant bot specialiced on urology. 
    You are required to help doctors with their health-related questions.
    Answer the following questions as deatiled as possible to help them to understand
    the patient's health condition better, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the current context", don't provide the wrong answer\n\n
    Context:\n {context} \n
    Question: \n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=LLM_MODEL,
                            temperature=LLM_TEMPERATURE)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    print("Conversational Chain loaded")
    return chain

# Parse the user input question and return the answer
def answer_question(user_question):
    print("Answering question...: ", user_question)
    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDINGS_MODEL)
    vector_store = FAISS.load_local(folder_path="faiss_store", 
                                        embeddings=embeddings_model,
                                        allow_dangerous_deserialization=True)
    
    docs = vector_store.similarity_search(user_question, k=5)

    chain = get_conversational_chain()
    response = chain(
        { "input_documents":docs, "question": user_question }, 
        return_only_outputs=True )
    return response

# Parse all documents and creates vector store
def setup_llm_with_rag():
    with st.spinner("Processing..."):
        # List all files in the content
        files = list_files()
        text_chunks = []
        for file in files:
            print("Processing file: ", file)
            # Read PDF file and return text content
            file_content = read_pdf(file)
            # Get Text Chunks
            text_chunks.extend(get_text_chunks(file_content) )
            # Create vector store
            convert_chunks_to_vectors(text_chunks)
            print("Pausing ...")
            time.sleep(20)
        st.success("Done")


# Streamlit App
def main():
    #st.set_page_config(page_title="Health Assistant Bot", page_icon="🩺")
    name, authentication_status, username = authenticator.login(key='Login', location='main')
    # Setup context
    if not os.path.exists("faiss_store"):
        setup_llm_with_rag()
    
    # Authenticate user
    if st.session_state['authentication_status']:
        authenticator.logout()
        # Setup Streamlit page
        st.header("CHAT with your Health Assistant Bot")
        st.write(f'Welcome *{name}*')
        # User input
        user_input = st.text_input("Enter your question here:")
        if st.button("Ask"):
            response = answer_question(user_input)
            parsed_response = markdown.markdown(response["output_text"])
            st.write(parsed_response, unsafe_allow_html=True)
    elif st.session_state['authentication_status'] is False:
        st.error('Username/password is incorrect')
    elif st.session_state['authentication_status'] is None:
        st.warning('Please enter your username and password')
    


if __name__ == "__main__":
    main()
