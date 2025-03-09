import pandas as pd
from src.analysis.qualitative import extract_conversation_text
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os


# Function to create and save vector embeddings
def create_vector_embeddings(conversations_df: pd.DataFrame, persist_directory: str = "../data/chroma_db"):
    """Create vector embeddings for each conversation and save to a vector database."""
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings()

    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    # Check if database already exists
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        print(f"Loading existing vector database from {persist_directory}")
        # Load existing database
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore

    # Extract conversation texts and metadata
    texts = []
    metadatas = []

    for idx, row in conversations_df.iterrows():
        conversation_text = extract_conversation_text(row)
        texts.append(conversation_text)
        metadatas.append({"conversation_id": idx})

    print(f"Creating new vector database in {persist_directory}")
    # Create and persist the vector store
    vectorstore = Chroma.from_texts(
        texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory=persist_directory
    )

    # Persist the database
    vectorstore.persist()

    return vectorstore
