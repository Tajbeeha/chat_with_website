import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text and structure from the website
def get_website_text(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch webpage. Status code: {response.status_code}")
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract text from paragraph, heading, and list tags
    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li'])
    text_content = "\n".join([para.get_text() for para in paragraphs])

    # Extract tables as text
    tables = soup.find_all('table')
    table_content = ""
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            table_content += " | ".join([cell.get_text() for cell in cells]) + "\n"

    # Extract image URLs
    images = soup.find_all('img')
    image_urls = [img['src'] for img in images if 'src' in img.attrs]

    # Clean and return the extracted content
    return text_content + "\n" + table_content + "\nImages: " + ", ".join(image_urls)

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create a conversational QA chain
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\nContext:\n{context}?\nQuestion:\n{question}\nAnswer:"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user queries
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load FAISS index with dangerous deserialization enabled (trusted local data)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main Streamlit application
def main():
    st.set_page_config("Chat Website")
    st.header("Chat with Website Content")
    
    # URL input
    url = st.text_input("Enter the URL of the website:")
    
    if url:
        # Extract text from the website
        with st.spinner("Extracting content from the website..."):
            website_text = get_website_text(url)
        
        if website_text:
            text_chunks = get_text_chunks(website_text)
            
            # Generate FAISS vector store
            get_vector_store(text_chunks)
            st.success("Website content processed successfully!")
            
            # User query input
            user_question = st.text_input("Ask a question about the website content:")
            if user_question:
                with st.spinner("Processing your query..."):
                    response = user_input(user_question)
                st.write("Reply:", response)

if __name__ == "__main__":
    main()
