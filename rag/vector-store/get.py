from langchain_core.documents import Document
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma


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


vector_store = Chroma(
    collection_name="player_info",
    embedding_function=embeddings,
    persist_directory="chroma_db_data",
)

# res = vector_store.get(include=["embeddings", "documents", "metadatas"])

# print(res)

search_res = vector_store.similarity_search(query="who is caption cool", k=1)
print(f"who is caption cool- {search_res}")
print(
    "with score ",
    vector_store.similarity_search_with_score(
        query="who is caption cool", k=2, filter={"team": "CSK"}
    ),
)
vector_store.update_document(document_id="", document=Document(page_content=""))
# vector_store.add_documents()
# vector_store.delete(ids=[''])


# etc function
