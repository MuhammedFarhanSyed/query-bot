# Interview Handbook AI Assistant (Query Bot) 🤖

An interactive, multi-source Retrieval-Augmented Generation (RAG) assistant built using **LangChain**, **FAISS**, **HuggingFace Embeddings**, **Groq LLMs**, and **Streamlit**.

The assistant ingests documents across multiple formats (PDF, TXT, DOCX) as well as live web data, creates dense vector representations, and enables factual, context-grounded conversational search.

---

## 🌟 Features

- **Multi-Source Knowledge Ingestion**:
  - 📄 **PDF**: Python Interview Handbook (`data/pythonHandbook.pdf`)
  - 📝 **TXT**: Java concepts and interview notes (`data/java_data.txt`)
  - 📑 **DOCX**: React interview notes (`data/react_data.docx`)
  - 🌐 **Web Scraping**: Live website text extraction from [Dhee Coding Lab](https://dheecodinglab.com/)
- **Text Chunking & Preprocessing**: Smart document chunking using LangChain's `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 100).
- **Dense Vector Search**: Powered by `sentence-transformers/all-MiniLM-L6-v2` embeddings and indexed via **FAISS** for fast top-$k$ similarity search.
- **Ultra-Fast LLM Inference**: Integrated with **Groq** for high-throughput, low-latency completions.
- **Interactive Streamlit UI**:
  - Model selection dropdown in sidebar.
  - Conversational chat interface with persistent message history in session state.
  - Collapsible **Retrieved Context** expander showing the exact document source chunks retrieved for each response.

---

## 📁 Project Structure

```text
├── .streamlit/
│   └── logo                     # Streamlit app logo assets
├── data/
│   ├── java_data.txt            # Java notes knowledge source
│   ├── pythonHandbook.pdf       # Python Handbook PDF knowledge source
│   └── react_data.docx          # React notes knowledge source
├── images/
│   └── icon_dcl.png             # UI branding icon
├── .env.example                 # Environment variables template
├── .gitignore                   # Ignored files (secrets, caches, venv)
├── chatbot.py                   # Main Streamlit RAG application
├── demo_model_practice.py       # Standalone test/practice script
├── req.txt                      # Project package dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or 3.11
- A [Groq API Key](https://console.groq.com/keys)

### 2. Clone the Repository

```bash
git clone https://github.com/MuhammedFarhanSyed/query-bot.git
cd query-bot
```

### 3. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r req.txt
```

> **Note on Compatibility**: `req.txt` pins `faiss-cpu==1.8.0.post1` and `langchain-huggingface>=0.1.0,<0.2.0` to ensure complete binary compatibility with NumPy 1.x and LangChain 0.3.x.

### 5. Configure Environment Variables

Create a `.env` file in the root directory (or copy from `.env.example`):

```bash
cp .env.example .env
```

Add your Groq API key inside `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🖥️ Running the Application

Start the Streamlit application:

```bash
streamlit run chatbot.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

---

## ⚙️ How It Works (RAG Pipeline)

1. **Ingest**: `PyPDFLoader`, `python-docx`, raw text readers, and `BeautifulSoup` read raw data across files and URLs.
2. **Chunk**: Documents are divided into 500-character segments with 100-character overlaps.
3. **Embed**: Each chunk is mapped to dense 384-dimensional vector space using `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`).
4. **Index**: Embeddings are stored in a local in-memory FAISS vector index, cached using `@st.cache_resource`.
5. **Retrieve**: When a user inputs a query, FAISS retrieves the top 5 most semantically relevant text chunks.
6. **Generate**: The context and user query are injected into a strict prompt template and passed to Groq to generate a fact-grounded response.

---

## 📄 License

This project is licensed under the MIT License.
