import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
#from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# page config
st.set_page_config(page_title="Ask PDF", page_icon="📚",
                   layout="centered", initial_sidebar_state="auto", menu_items=None)

# main app
st.subheader(
    "💬 :orange[GROQ] :orange-background[RAG-PDF] :orange[CHATBOT] :books:", divider='orange')


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=st.secrets['NOMIC_API'])
    #vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    #vectordb = Chroma.from_documents(splits, embeddings, persist_directory="vectorstore") # TO PERSIST VECTORSTORE IN LOCAL FOLDER
    vectordb = Chroma.from_documents(splits, embeddings) # WITHOUT PERSISTING VECTORSTORE IN LOCAL FOLDER

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


# Get an Groq API Key before continuing
with st.sidebar:
    st.text("Ask PDF - Retrieval Augmented Generation (RAG) App built using LangChain, Nomic embeddings, and Groq Llama3")
    "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/0xZee/AskPDF-bot/)"
    st.code("github.com/0xZee/AskPDF-bot")
    st.subheader("🔐 GROQ INFERENCE API", divider="grey")
    if "GROQ_API" in st.secrets:
        GROQ_API = st.secrets['GROQ_API']
        st.success("✅ GROQ API Key Set")
    else:
        GROQ_API = st.text_input("Groq API Key", type="password")
    if not GROQ_API:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()

    st.subheader("📚 PDF DOCUMENTS", divider="grey")
    uploaded_files = st.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload PDF documents to continue.")
        st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain - model "llama3-8b-8192" , "mixtral-8x7b-32768"
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, groq_api_key=GROQ_API, streaming=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear Chat History", use_container_width=True):
    msgs.clear()
    msgs.add_ai_message("👋 Hello, I'm here to help, How can I assist you with your PDF 📚 ? :sparkles:")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything about your PDF"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
