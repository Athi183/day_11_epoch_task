# 📄 PDF Question Answering Assistant (RAG Pipeline)

**Participant Name**: Athira V  
**MUID**: athirav-3@mulearn  
**Project Repository**: https://github.com/Athi183/day_11_epoch_task  

---

## 🌟 Project Overview

The **PDF Question Answering Assistant** is a production-ready Retrieval-Augmented Generation (RAG) system that allows users to upload any PDF document and ask unlimited context-grounded questions. The application combines **LangChain 0.3.x**, **ChromaDB**, **HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)**, and **Groq (`llama-3.3-70b-versatile`)** with an interactive **Gradio** web interface.

---

## ✨ Features

- **Document Processing**: Automatic parsing with `PyPDFLoader` and text chunking using `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
- **Progress Tracking**: Real-time status updates reporting document filename, total pages loaded, and text chunks generated.
- **High-Performance Embeddings**: Semantic embeddings generated locally using `sentence-transformers/all-MiniLM-L6-v2`.
- **In-Memory Vector Search**: Rapid similarity retrieval (`k=4`) powered by ChromaDB.
- **History-Aware Retrieval**: Multi-turn dialogue support using `create_history_aware_retriever` to reformulate follow-up questions in context.
- **Strict Non-Hallucination Guardrails**: Powered by Groq's `llama-3.3-70b-versatile` at zero temperature. If context does not contain the answer, returns:
  > *"I couldn't find that information in the uploaded PDF."*
- **Conversational Memory Lifecycle**: Managed per session using `RunnableWithMessageHistory`. Uploading a new PDF automatically resets memory and rebuilds the vector database.
- **Hugging Face Spaces Ready**: Pre-configured for seamless deployment on Hugging Face Spaces.

---

## 🛠️ Technologies Used

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Web Interface** | Gradio |
| **RAG Orchestration** | LangChain 0.3.x (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| **LLM Provider** | Groq API (`llama-3.3-70b-versatile`) |
| **Embeddings** | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB (`langchain-chroma` / `chromadb`) |
| **PDF Loader** | `pypdf` (`PyPDFLoader`) |
| **Environment Management** | `python-dotenv` |

---

## 📂 Folder Structure

```text
pdf-rag-chatbot/
│
├── app.py              # Core application entry point & RAG pipeline
├── requirements.txt    # Production dependencies with pinned versions
├── README.md           # Project documentation & execution guide
├── .env.example        # Environment variable template
└── .gitignore          # Git exclusion rules
```

---

## ⚙️ Installation & Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Athi183/day_11_epoch_task.git
   cd day_11_epoch_task
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔑 Environment Variables

1. Get a free API Key from [console.groq.com](https://console.groq.com).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and set your key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   ```

---

## 🚀 Running Locally

Launch the application:

```bash
python app.py
```

Open your browser and navigate to: **`http://localhost:7860`**

---

## 🌐 How to Deploy to Hugging Face Spaces

Deploying this application to **Hugging Face Spaces** takes less than 3 minutes:

### Step 1: Create a New Space on Hugging Face
1. Log in to [Hugging Face](https://huggingface.co) (or create an account).
2. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**.
3. Fill in the Space details:
   - **Space Name**: `pdf-rag-assistant` (or your preferred name)
   - **License**: Choose `mit` or leave default
   - **Select the Space SDK**: Choose **Gradio**
   - **Space Hardware**: Choose **CPU basic - Free**
   - **Visibility**: Public or Private
4. Click **Create Space**.

### Step 2: Push Your Code to the Hugging Face Space
You can deploy using Git or directly via the web interface:

#### Option A: Via Git CLI
```bash
# Add Hugging Face Space as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME.git

# Push code to Hugging Face Space
git push hf main
```

#### Option B: Drag and Drop Files
Upload the following files directly in the Space files tab:
- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`

### Step 3: Add Your Groq API Key Secret
1. Inside your Hugging Face Space, click on **Settings** (top right tab).
2. Scroll down to **Variables and secrets**.
3. Click **New secret**.
4. Set:
   - **Name**: `GROQ_API_KEY`
   - **Value**: `gsk_...` (your actual Groq API key)
5. Click **Save**.

Hugging Face will automatically build the environment, install dependencies from `requirements.txt`, and launch `app.py` on port 7860!

---

## 🧠 Memory Implementation & Lifecycle

The application utilizes **`RunnableWithMessageHistory`** combined with **`ChatMessageHistory`** to preserve multi-turn context:

```mermaid
graph TD
    A[User Question] --> B{History Aware Retriever}
    B -->|Contextualized Query| C[ChromaDB Vector Store]
    C -->|Top-4 Chunks| D[QA Chain]
    D --> E[Groq Llama 3.3 70B]
    E --> F[Answer + Append to Session Memory]
```

- **Session Isolation**: Chat history is tracked per session using `get_session_history()`.
- **Upload Reset**: Uploading a new PDF invokes `session_store.clear()` and disposes of the previous `vectorstore` instance to prevent cross-document memory leaks.
- **Manual Reset**: Clicking **Clear Chat** resets conversation history without needing to re-process the PDF.

---

## ⚡ Challenges Faced & Solutions

1. **Gradio 5/6 Migration & Message Formats**:
   - *Challenge*: Newer Gradio versions transitioned `gr.Chatbot` from tuple pairs to structured dictionary messages (`{"role": "...", "content": "..."}`).
   - *Solution*: Refactored chat handlers to output standardized role-content dictionaries.

2. **Hallucination Prevention**:
   - *Challenge*: Open-domain LLMs tend to extrapolate beyond document context.
   - *Solution*: Enforced strict system prompt guardrails with `temperature=0` instructing the model to respond *exactly* with `"I couldn't find that information in the uploaded PDF."` when out-of-context.

---

## 🔮 Future Improvements

- **Multi-File PDF Processing**: Support simultaneous uploading and merging of multiple PDFs into a single vector database.
- **Source Citation & Highlights**: Display page numbers and extracted text snippets alongside answers.
- **Hybrid Retrieval**: Combine BM25 keyword search with dense vector retrieval for improved accuracy on complex terms.
