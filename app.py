"""
📄 PDF Question Answering Assistant (RAG Pipeline)
=================================================
A production-ready PDF Question Answering application built with:
- Python & Gradio
- LangChain 0.3.x Ecosystem APIs
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- ChromaDB Vector Store
- Groq API (llama-3.3-70b-versatile)
- History-Aware Conversational Retrieval Chain with RunnableWithMessageHistory
"""

import os
from typing import Dict, List, Tuple, Any, Optional

import gradio as gr
from dotenv import load_dotenv

# LangChain 0.3.x modern imports with fallback safety
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Global state management
embedding_model: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[Chroma] = None
conversational_rag_chain: Optional[RunnableWithMessageHistory] = None
session_store: Dict[str, BaseChatMessageHistory] = {}
SESSION_ID: str = "pdf_rag_session"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Lazy load and return the HuggingFace embeddings model instance.

    Returns:
        HuggingFaceEmbeddings: Pretrained sentence-transformers embedding model.
    """
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return embedding_model


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or initialize in-memory conversation message history for a session.

    Args:
        session_id (str): Unique session identifier.

    Returns:
        BaseChatMessageHistory: Active chat message history.
    """
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def build_chain(retriever: VectorStoreRetriever, api_key: str) -> RunnableWithMessageHistory:
    """Construct a History-Aware RAG Retrieval Chain with strict prompt guardrails.

    Args:
        retriever (VectorStoreRetriever): Document retriever from ChromaDB.
        api_key (str): Groq API key.

    Returns:
        RunnableWithMessageHistory: Fully wrapped conversational RAG pipeline.
    """
    # 1. Initialize Groq LLM (llama-3.3-70b-versatile, temperature=0 for deterministic output)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

    # 2. History-aware retriever prompt to contextualize follow-up questions
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. Question Answering prompt with strict non-hallucination guardrails
    qa_system_prompt = (
        "You are an expert AI assistant answering questions strictly based on the uploaded PDF document.\n"
        "Use ONLY the following retrieved pieces of context to answer the user's question.\n"
        "If the answer is not found in the provided document context, respond EXACTLY with:\n"
        "\"I couldn't find that information in the uploaded PDF.\"\n"
        "Do NOT invent, extrapolate, or hallucinate answers under any circumstances.\n"
        "Keep your answer clear, accurate, and directly grounded in the text.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 4. Create document combination QA chain & full retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 5. Attach session history runner
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def process_pdf(file_obj):
    """Process uploaded PDF: load pages, split into chunks, embed, index in Chroma, and build RAG chain.

    Args:
        file_obj: File object uploaded via Gradio.

    Returns:
        Tuple[str, list]: Status feedback message and cleared chat history.
    """
    global vectorstore, conversational_rag_chain, session_store
    progress = gr.Progress()

    # 1. Error handling: Check Groq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or not groq_api_key.strip():
        return (
            "⚠️ Error: GROQ_API_KEY is missing.\n"
            "Please add your GROQ_API_KEY to the .env file and restart the app.",
            []
        )

    # 2. Error handling: Check if PDF file is uploaded
    if file_obj is None:
        return "⚠️ Error: No PDF file selected. Please upload a PDF document.", []

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    if not file_path or not os.path.exists(file_path):
        return "⚠️ Error: The specified PDF file path could not be found or read.", []

    filename = os.path.basename(file_path)

    try:
        progress(0.1, desc="📄 Reading PDF document...")
        # Reset memory state and previous vectorstore reference for fresh upload
        session_store.clear()
        vectorstore = None

        # 3. Load PDF via PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            return f"⚠️ Error: '{filename}' appears to be empty or contains no extractable text.", []

        num_pages = len(documents)

        # 4. Text Chunking via RecursiveCharacterTextSplitter
        progress(0.3, desc="🧩 Chunking document text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        if not splits:
            return f"⚠️ Error: Failed to split text in '{filename}'. Check if the PDF is image-only or corrupt.", []

        num_chunks = len(splits)

        # 5. Generate embeddings
        progress(0.6, desc="🧠 Generating HuggingFace embeddings...")
        embeddings = get_embedding_model()

        # 6. Store in ChromaDB vector store
        progress(0.8, desc="⚡ Indexing vector store in ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        # 7. Create retriever with top-k similarity search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # 8. Build History-Aware Conversational Chain
        progress(0.95, desc="⚙️ Finalizing RAG chain...")
        conversational_rag_chain = build_chain(retriever, groq_api_key.strip())

        progress(1.0, desc="✅ Complete!")
        status_msg = (
            f"✅ PDF Processed Successfully!\n\n"
            f"📄 Filename: {filename}\n"
            f"📑 Pages Loaded: {num_pages}\n"
            f"🧩 Text Chunks Created: {num_chunks}\n\n"
            f"You can now ask unlimited questions about this document."
        )
        return status_msg, []

    except Exception as e:
        return f"⚠️ Error processing PDF '{filename}': {str(e)}", []


def chat(user_message, history):
    """Process user query through conversational RAG chain and append response to chat history.

    Args:
        user_message: Question entered by user.
        history: Current chat history.

    Returns:
        Tuple[str, list]: Reset input box text and updated history list.
    """
    global conversational_rag_chain

    if history is None:
        history = []

    if not user_message or not str(user_message).strip():
        return "", history

    # Error handling: Check if PDF has been processed
    if conversational_rag_chain is None:
        history.append({"role": "user", "content": user_message})
        history.append({
            "role": "assistant",
            "content": "⚠️ Please upload and process a PDF document first before asking questions."
        })
        return "", history

    try:
        # Invoke conversational RAG chain with session context
        response = conversational_rag_chain.invoke(
            {"input": str(user_message).strip()},
            config={"configurable": {"session_id": SESSION_ID}}
        )
        answer = response.get("answer", "I couldn't find that information in the uploaded PDF.")

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
        return "", history

    except Exception as e:
        history.append({"role": "user", "content": user_message})
        history.append({
            "role": "assistant",
            "content": f"⚠️ An error occurred while generating the answer: {str(e)}"
        })
        return "", history


def clear_chat():
    """Reset chat history window and conversation memory store.

    Returns:
        Tuple[list, str]: Empty chat messages list and updated status notice.
    """
    global session_store
    session_store.clear()
    return [], "Conversation history cleared."


# ==============================================================================
# Gradio Modern User Interface Layout
# ==============================================================================

custom_css = """
/* Clean UI styling */
.container { max-width: 1200px; margin: 0 auto; }
.status-box textarea { font-family: monospace; font-size: 0.95rem; font-weight: 500; }
"""

with gr.Blocks(title="📄 PDF Question Answering Assistant", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 📄 PDF Question Answering Assistant
        Upload any PDF document, process its contents into an embedded vector database, and ask questions to receive context-grounded answers powered by **LangChain 0.3.x**, **ChromaDB**, **HuggingFace Embeddings**, and **Groq (Llama 3.3 70B)**.
        """
    )

    with gr.Row():
        # Left Panel: PDF Management & Metadata
        with gr.Column(scale=1):
            gr.Markdown("### 📥 1. Document Upload & Indexing")
            pdf_file = gr.File(
                label="Select PDF File",
                file_types=[".pdf"],
                file_count="single"
            )
            process_btn = gr.Button("⚡ Process PDF", variant="primary")
            status_output = gr.Textbox(
                label="Processing Status & Metadata",
                value="Waiting for PDF upload...",
                interactive=False,
                lines=8,
                elem_classes=["status-box"]
            )

        # Right Panel: Interactive Q&A Chatbot
        with gr.Column(scale=2):
            gr.Markdown("### 💬 2. Contextual Question Answering")
            try:
                chatbot = gr.Chatbot(
                    label="Conversation History",
                    height=450,
                    type="messages"
                )
            except TypeError:
                chatbot = gr.Chatbot(
                    label="Conversation History",
                    height=450
                )
            question_input = gr.Textbox(
                label="Ask a question about the uploaded document",
                placeholder="e.g., What are the key findings or takeaways described in this PDF?",
                lines=2
            )

            with gr.Row():
                send_btn = gr.Button("🚀 Send Question", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    # Event Wiring
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_file],
        outputs=[status_output, chatbot],
        api_name=False
    )

    send_btn.click(
        fn=chat,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot],
        api_name=False
    )

    question_input.submit(
        fn=chat,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot],
        api_name=False
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, status_output],
        api_name=False
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

