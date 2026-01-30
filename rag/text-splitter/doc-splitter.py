from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader

text = """
from pathlib import Path

location = Path("/").cwd() / "learn-langchain" / "data.pdf"
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="")

# res = splitter.split_text(text=text)

loader = PyPDFLoader(file_path=location)
res = splitter.split_documents(loader.load())

print(f"{len(res)=}")
print(f"{res[1].page_content=}")
"""
# text = "My name is utsav \n\n he lived in navsari"
from pathlib import Path

location = Path("/").cwd() / "learn-langchain" / "data.pdf"
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=100
)

res = splitter.split_text(text=text)

# loader = PyPDFLoader(file_path=location)
# res = splitter.split_documents(loader.load())

print(f"{len(res)=}")
print(f"{res[1]=}")
print(f"{res=}")
