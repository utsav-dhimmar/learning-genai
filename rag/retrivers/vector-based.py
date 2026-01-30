from pathlib import Path

from dotenv import load_dotenv

# from langchain_core.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.retrievers import MultiQueryRetriever

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


vector_store = Chroma(
    collection_name="player_info",
    embedding_function=embeddings,
    persist_directory="chroma_db_data",
)

# TODO LEARN ABOUT other way to find relevane tdocs

# print(res)
retrivers = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.7}
)


# mutli_retrivers = MultiQueryRetriever.from_llm(
#     search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.7}
# )
# res = vector_store.get(include=["embeddings", "documents", "metadatas"])
# vector_store =VectorStore
# vector_store.
# vector_store.add_documents()
# vector_store.delete(ids=[''])

res = retrivers.invoke("virat kohli")
# etc function
print(res)
