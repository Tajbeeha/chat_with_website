Project Overview : 
This project enables users to interact with content extracted from url website. By using advanced natural language processing techniques,it allows users to ask questions about the content and receive detailed answers.

Packages Involved :
Streamlit: A library for building interactive web applications.
PyMuPDF (fitz): Used to extract text from PDF files.
LangChain: A framework for building applications with language models.
Google Generative AI: A language model for generating embeddings and answers.
FAISS: A library for efficient similarity search.
dotenv: A tool for managing environment variables.
requests: A package for handling HTTP requests (if required for other functionalities like API calls).

Project Process :
Environment Setup
* Install the necessary Python packages:
         pip install streamlit pypdf2 langchain google-generativeai faiss-cpu python-dotenv
  
* Add API Key to .env File
    Create a .env file in the root directory of your project and add your Google API key as follows:
    GOOGLE_API_KEY=your_google_api_key_here
  
* URL Content Extraction
* Embedding Creation
* Vector Store Creation
* Question Retrieval
* Document Retrieval
* Answer Generation
* User Interaction through Streamlit UI

How to Run the Project :
 * Clone this repository to your local machine.
 * Set up a virtual environment (optional but recommended).
 * Install the required packages using pip install -r requirements.txt.
 * Create a .env file and add your GOOGLE_API_KEY.
   
Run the Streamlit app using:
     streamlit run app.py
* Open the provided URL in your browser to interact with the app.
  
Example Workflow
Upload a url.
Ask questions about the content of the website.
The system processes the website content, retrieves relevant sections, and generates answers.
