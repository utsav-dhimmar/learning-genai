from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


load_dotenv()
base = Path("/")
model = ChatGoogleGenerativeAI(
    # model="gemini-3-pro-preview",
    model="gemini-3-flash-preview",
    # model="gemini-2.5-flash-lite",
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=32,
    task_type="SEMANTIC_SIMILARITY",
)

location = base.cwd()

docs: list[Document] = [
    Document(
        page_content="Virat kohli was a caption of RCB",
        metadata={"team": "RCB"},
    ),
    Document(
        page_content="Rohit Sharam is very successful caption of Mumbai Indians",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="MS Dhoni known as caption cool, he was a caption of CSK",
        metadata={"team": "CSK"},
    ),
    Document(
        page_content="Jasprit Bumrah is one the best bowler of india and he plays from Mumbai Indians ",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="Ravindra Jadeja is all rounder, every batsman think before running he play from CSK",
        metadata={"team": "CSK"},
    ),
]


vector_store = Chroma(
    collection_name="player_info",
    embedding_function=embeddings,
    persist_directory="chroma_db_data",
)

vectors = vector_store.add_documents(documents=docs)

print(vectors[0:3])
