import os
from flask import Flask, render_template, request, jsonify
import PyPDF2
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import google.generativeai as genai

app = Flask(__name__)
load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Gemini
API_KEY = "AIzaSyDDgDljAhNfMavVCX-xD15X2xAQ447YgU8"
if not API_KEY:
    raise ValueError("Missing Google Gemini API key. Set the GOOGLE_API_KEY environment variable.")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

# Global variable to store vectorstore
vectorstore = None

def process_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

def perform_similarity_search(query, k=1):
    global vectorstore
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)

def generate_response(context, query):
    prompt = f"""Based on the following context from Database Management Systems notes, 
    please answer the question. If the answer isn't in the context, say so.

    Context:
    {context}

    Question: {query}

    Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def initialize_vectorstore():
    global vectorstore
    
    file_path = "Database-Management-Systems.pdf"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return False
        
    try:
        pdf_text = process_pdf(file_path)
        chunks = split_text(pdf_text)

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name="dbms_notes_chunks"
        )
        return True
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        user_query = data.get('question', '')
        
        if not user_query:
            return jsonify({'error': 'No question provided'}), 400

        global vectorstore
        if vectorstore is None:
            if not initialize_vectorstore():
                return jsonify({'error': 'Failed to initialize vector store'}), 500

        results = perform_similarity_search(user_query, k=2)
        
        if not results:
            return jsonify({'error': 'No relevant context found'}), 404
        
        context = "\n".join([doc.page_content for doc in results])
        response = generate_response(context, user_query)
        
        return jsonify({
            'answer': response,
            'context': context
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if initialize_vectorstore():
        app.run(debug=True)
    else:
        print("Failed to initialize vector store. Please check the PDF file and try again.")