import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import os
import tempfile

# Initialize Groq client
client = Groq(
    api_key="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"  # Directly set API key
)

# Initialize the model and Pinecone
model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = "final1"
pinecone_api_key = "f619ef3f-14c1-4348-9072-a575330dd5cf"  # Directly set API key

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def make_chunks(text):
    return text_splitter.split_text(text)

def get_context(ques, tot_chunks):
    index = pc.Index(index_name)
    ques_emb = model.encode(ques)
    DB_response = index.query(
        vector=ques_emb.tolist(),
        top_k=3,
        include_values=True
    )

    if not DB_response or 'matches' not in DB_response:
        st.error("No matches found in the database response.")
        return ""

    st.json(DB_response)  # Debugging information

    cont = ""
    for match in DB_response['matches']:
        try:
            chunk_index = int(match['id'][3:]) - 1
            cont += tot_chunks[chunk_index]
        except (IndexError, ValueError) as e:
            st.error(f"Error accessing chunk: {e}")
            st.error(f"Chunk ID: {match['id']}, Chunk Index: {chunk_index}")
    return cont

def extract_pdf(path):
    reader = PdfReader(path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() or ""
    return extracted_text

# Custom CSS for better visuals
st.markdown("""
    <style>
    .stSidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #007bff;
        color: white;
    }
    .stTextInput>div>div>input {
        width: 100%;
        padding: 8px;
    }
    .spinner-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .spinner {
        border: 6px solid #f3f3f3;
        border-top: 6px solid #007bff;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 2s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .home-container {
        text-align: center;
        padding: 50px;
    }
    .home-container h1 {
        font-size: 3rem;
        color: #007bff;
    }
    .home-container p {
        font-size: 1.2rem;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["DOC SCAN"])

if 'tot5' not in st.session_state:
    st.session_state.tot5 = []


if selection == "DOC SCAN":
    st.title("Document Analyzer")

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Upload and Process"):
        if uploaded_files:
            paths = []
            with tempfile.TemporaryDirectory() as tmpdirname:
                for file in uploaded_files:
                    file_path = os.path.join(tmpdirname, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    paths.append(file_path)
                    st.success(f"Uploaded file: {file.name}")

                extracted = ""
                for path in paths:
                    extracted += extract_pdf(path)
                
                tot_chunks = make_chunks(extracted)
                st.session_state.tot5 = tot_chunks  
                st.write("Total chunks created:", len(tot_chunks))  # Debugging information

                tot_embeddings = model.encode(tot_chunks)
                tot_vectors = [{"id": f"vec{i+1}", "values": vec.tolist()} for i, vec in enumerate(tot_embeddings)]

                # Check if index exists before creating it
                index_names = pc.list_indexes()
                if index_name in index_names:
                    st.info("Index already exists. Skipping creation.")
                else:
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                    st.success("Index created successfully.")

                index = pc.Index(index_name)
                index.upsert(tot_vectors)
                st.success("Documents processed and indexed successfully!")

    query = st.text_input("Enter your query:")
    if st.button("Get Answer"):
        if query:
            context = get_context(query, st.session_state.tot5)  
            if context:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Context: {context}, Analyse and understand the above context completely and answer the below query, Query: {query}",
                        }
                    ],
                    model="llama3-8b-8192",
                )
                response_text = chat_completion.choices[0].message.content
                st.write("Answer:")
                st.write(response_text)

    if st.button("Clear Database"):
        with st.spinner('Clearing database...'):
            try:
                pc.delete_index(index_name)
                st.success("Database cleared successfully!")
            except Exception as e:
                st.warning(f"Error clearing database: {e}")

