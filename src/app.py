import os

from dotenv import load_dotenv

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Website Chatbot", page_icon="🪢")
st.title("Website Chatbot")


def create_vector_store(url: str):
    loader = WebBaseLoader(url)
    document = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(documents=document)

    # Creating embeddings and storing in Qdrant
    # embed_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OPENAI_API_KEY
    )

    qdrant = Qdrant.from_documents(
        documents=docs,
        embedding=embed_model,
        url="localhost",
        collection_name="website-chat-bot",
    )

    return qdrant


def get_context_retriever_chain(vector_store):
    llm = ChatGroq(temperature=0, model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the conversation above, generate a query to find the most relevant content.",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_rag_chain(retriever_chain):
    llm = ChatGroq(temperature=0, model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Using only the context between the two `, answer the users questions without referncing that you are using the context. `{context}`",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_docuements_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_docuements_chain)


def get_response(question: HumanMessage, chat_history: list) -> str:

    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_rag_chain(retriever_chain=retriever_chain)

    response = rag_chain.invoke({"chat_history": chat_history, "input": question})

    return response["answer"], response["context"]


with st.sidebar:
    st.header("Website URL")
    website_url = st.text_input("Enter the URL:")

if website_url is None or website_url == "":
    st.info("Enter a website URL to chat with.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_vector_store(website_url)

    user_question = st.chat_input("Ask a question.")

    if user_question is not None and user_question != "":
        st.session_state.chat_history.append(HumanMessage(user_question))
        response, context = get_response(user_question, st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(response))

        with st.sidebar:
            context

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)
