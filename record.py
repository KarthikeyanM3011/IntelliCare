import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import google.generativeai as genai
from streamlit_option_menu import option_menu
import requests
from googletrans import Translator
from streamlit_lottie import st_lottie  # Make sure to install and import this if using Lottie animations

languages = [
    'English','Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque', 'Belarusian', 'Bengali', 'Bosnian',
    'Bulgarian', 'Catalan', 'Cebuano', 'Chichewa', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Corsican', 'Croatian',
    'Czech', 'Danish', 'Dutch', 'Esperanto', 'Estonian', 'Filipino', 'Finnish', 'French', 'Frisian', 'Galician',
    'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hmong', 'Hungarian',
    'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean',
    'Kurdish (Kurmanji)', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay',
    'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Myanmar (Burmese)', 'Nepali', 'Norwegian', 'Odia', 'Pashto',
    'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Samoan', 'Scots Gaelic', 'Serbian', 'Sesotho',
    'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik',
    'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yiddish', 'Yoruba','Zulu'
]

LANGUAGES = {
'Afrikaans': 'af',
'Albanian': 'sq',
'Amharic': 'am',
'Arabic': 'ar',
'Armenian': 'hy',
'Azerbaijani': 'az',
'Basque': 'eu',
'Belarusian': 'be',
'Bengali': 'bn',
'Bosnian': 'bs',
'Bulgarian': 'bg',
'Catalan': 'ca',
'Cebuano': 'ceb',
'Chichewa': 'ny',
'Chinese (Simplified)': 'zh-cn',
'Chinese (Traditional)': 'zh-tw',
'Corsican': 'co',
'Croatian': 'hr',
'Czech': 'cs',
'Danish': 'da',
'Dutch': 'nl',
'English': 'en',
'Esperanto': 'eo',
'Estonian': 'et',
'Filipino': 'tl',
'Finnish': 'fi',
'French': 'fr',
'Frisian': 'fy',
'Galician': 'gl',
'Georgian': 'ka',
'German': 'de',
'Greek': 'el',
'Gujarati': 'gu',
'Haitian Creole': 'ht',
'Hausa': 'ha',
'Hawaiian': 'haw',
'Hebrew': 'iw',
'Hindi': 'hi',
'Hmong': 'hmn',
'Hungarian': 'hu',
'Icelandic': 'is',
'Igbo': 'ig',
'Indonesian': 'id',
'Irish': 'ga',
'Italian': 'it',
'Japanese': 'ja',
'Javanese': 'jw',
'Kannada': 'kn',
'Kazakh': 'kk',
'Khmer': 'km',
'Korean': 'ko',
'Kurdish (Kurmanji)': 'ku',
'Kyrgyz': 'ky',
'Lao': 'lo',
'Latin': 'la',
'Latvian': 'lv',
'Lithuanian': 'lt',
'Luxembourgish': 'lb',
'Macedonian': 'mk',
'Malagasy': 'mg',
'Malay': 'ms',
'Malayalam': 'ml',
'Maltese': 'mt',
'Maori': 'mi',
'Marathi': 'mr',
'Mongolian': 'mn',
'Myanmar (Burmese)': 'my',
'Nepali': 'ne',
'Norwegian': 'no',
'Odia': 'or',
'Pashto': 'ps',
'Persian': 'fa',
'Polish': 'pl',
'Portuguese': 'pt',
'Punjabi': 'pa',
'Romanian': 'ro',
'Russian': 'ru',
'Samoan': 'sm',
'Scots Gaelic': 'gd',
'Serbian': 'sr',
'Sesotho': 'st',
'Shona': 'sn',
'Sindhi': 'sd',
'Sinhala': 'si',
'Slovak': 'sk',
'Slovenian': 'sl',
'Somali': 'so',
'Spanish': 'es',
'Sundanese': 'su',
'Swahili': 'sw',
'Swedish': 'sv',
'Tajik': 'tg',
'Tamil': 'ta',
'Telugu': 'te',
'Thai': 'th',
'Turkish': 'tr',
'Ukrainian': 'uk',
'Urdu': 'ur',
'Uyghur': 'ug',
'Uzbek': 'uz',
'Vietnamese': 'vi',
'Welsh': 'cy',
'Xhosa': 'xh',
'Yiddish': 'yi',
'Yoruba': 'yo',
'Zulu': 'zu'
}

GOOGLE_API_KEY = 'AIzaSyAPMmJVLK403FPhDjNCz2f6cbZWVWROGLg'
genai.configure(api_key=GOOGLE_API_KEY)

model_name = 'gemini-1.0-pro'
client = genai.GenerativeModel(model_name)

model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = "index0"
pinecone_api_key = "f619ef3f-14c1-4348-9072-a575330dd5cf"
pc = Pinecone(api_key=pinecone_api_key)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def make_chunks(text):
    return text_splitter.split_text(text)


def get_context(ques, tot_chunks):
    index = pc.Index(index_name)
    ques_emb = model.encode(ques)
    DB_response = index.query(vector=ques_emb.tolist(), top_k=3, include_values=True)

    if not DB_response or 'matches' not in DB_response:
        st.error("No matches found in the database response.")
        return ""

    st.json(DB_response)

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


def record():
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

    st.title("Report Chat")

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

                tot_embeddings = model.encode(tot_chunks)
                tot_vectors = [{"id": f"vec{i+1}", "values": vec.tolist()} for i, vec in enumerate(tot_embeddings)]

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
                    # st.success("Index created successfully.")

                index = pc.Index(index_name)
                index.upsert(tot_vectors)
                st.success("Documents processed successfully!")

    query = st.text_input("Enter your query:")
    if st.button("Get Answer"):
        try:
            if query:
                context = get_context(query, st.session_state.tot5)
                if context:
                    input_text = f"Context: {context}, Analyse and understand the above context completely and answer the below query, Query: {query}"
                    output = client.generate_content(input_text)
                    response_text = output.text
                    st.write("Answer:")
                    st.write(response_text)
        except Exception as e:
            print(e)

    if st.button("Clear Database"):
        with st.spinner('Clearing database...'):
            try:
                pc.delete_index(index_name)
                st.success("Database cleared successfully!")
            except Exception as e:
                st.warning(f"Error clearing database: {e}")


def scanner():
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    def extract_pdf(path):
        reader = PdfReader(path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
        return extracted_text
    
    def translate(info,lang):
            translator=Translator()
            translation = translator.translate(info, dest=LANGUAGES[lang])
            return translation.text

    left_column, right_column = st.columns((2, 5))

    with left_column:
        logo = 'https://lottie.host/49cfa049-139a-498a-954f-7985b2b60086/qvfWaOHQJR.json'
        logo_image = load_lottieurl(logo)  # Ensure load_lottieurl is defined or imported
        st_lottie(logo_image, width=300, height=100, key='logo')

    with right_column:
        st.header("RecordChat")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    res=''
    if pdf_file is not None:
        try:
            prefix = 'You are a helpful medical assistant. From the Patient details given below for your reference, you need to generate a detailed report about it and explain the terms in that and other stuff mentioned in the medical report content. You must give only the detailed explanation of the given content in a complete report structure.Give the response in a prettified manner. **Medical report content**: '
            query = extract_pdf(pdf_file)
            res = client.generate_content(prefix + query)
            res = res.text
        except Exception as e:
            res = 'Error Occured while generating'
            print(e)
        finally:
            print(res)
            st.write("Report Insights")
            answer_placeholder=st.empty()
            answer_placeholder.write(res)
            selected_option = st.selectbox("Select an option",languages )
            if(selected_option != 'Select an option' ):
                translated_answer = translate(res,selected_option)
                answer_placeholder.markdown(translated_answer)
            


def report():
    selected = option_menu(None, ['RecordScan', 'RecordQuery'],
                           icons=['camera', 'search'],
                           menu_icon='cast',
                           default_index=0,
                           orientation='horizontal')

    if selected == 'RecordScan':
        scanner()
    if selected == 'RecordQuery':
        record()
