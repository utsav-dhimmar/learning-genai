from langchain_community.document_loaders import TextLoader
from pathlib import Path

base = Path
file_path = base.cwd() / "rag" / "document-loader" / "data.txt"

loader = TextLoader(file_path=file_path, encoding="utf-8")
docs = loader.load()  # load in to memory as Document()

print(docs)
print(f"{len(docs)=}")
print(f"{type(docs)=}")
print(f"{docs[0]}")
print(f"{docs[0].page_content=}")
print(f"{docs[0].metadata=}")
