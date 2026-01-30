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
