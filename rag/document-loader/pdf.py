from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

base = Path("/")
pdf_path = base.cwd() / "learn-langchain" / "data.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"{docs=}")
print(f"{len(docs)=}")
print(f"{docs[0].page_content=}")
print(f"{docs[0].metadata=}")
