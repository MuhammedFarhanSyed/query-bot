import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import requests

st.set_page_config(page_title="Python Handbook QA")
st.title("😉 Interview Handbook AI Assistant")
st.write("Ask questions based on the loaded knowledge sources.")

load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.image("images/icon_dcl.png", width=40)
    st.title("Settings")
    model_name = st.selectbox(
        "Choose Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )
    messages = st.session_state.messages
    for message in reversed(st.session_state.messages):

    # for message in messages:
        # st.chat_message(message)
        if message['role']=='user':
            print(message['content'])
            st.write(message['content'])
        



@st.cache_resource
def initialize_retriever():
    all_docs = []

    # PDF
    pdf_loader = PyPDFLoader("data/pythonHandbook.pdf")
    all_docs.extend(pdf_loader.load())

    # TXT
    with open("data/java_data.txt", "r", encoding="utf-8", errors="ignore") as f:
        all_docs.append(Document(page_content=f.read(), metadata={"source": "java_data.txt"}))

    # DOCX
    doc = DocxDocument("data/react_data.docx")
    doc_text = "\n".join(p.text for p in doc.paragraphs)
    all_docs.append(Document(page_content=doc_text, metadata={"source": "react_data.docx"}))

    # Website
    try:
        response = requests.get("https://dheecodinglab.com/", timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        web_text = soup.get_text(separator="\n", strip=True)
        print(web_text)
        all_docs.append(Document(page_content=web_text, metadata={"source": "website"}))
    except Exception:
        pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb.as_retriever(search_kwargs={"k": 5})



try:
    retriever = initialize_retriever()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

llm = ChatGroq(model_name=model_name)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, reply exactly:

Sorry, I could not get the answer.

Context:
{context}

Question:
{question}

Answer:
"""
)


if 'messages_history' not in st.session_state:
    st.session_state.messages_history = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask your question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            final_prompt = prompt.format(
                context=context,
                question=question
            )

            response = llm.invoke(final_prompt)

            answer = response.content

            st.markdown(answer)

            with st.expander("Retrieved Context"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(d.page_content[:1000])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}

    )
    # st.session_state.messages_history.append()


