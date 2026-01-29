from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

location = Path("/").cwd() / "learn-langchain"

loader = DirectoryLoader(path=str(location), glob="*.py", loader_cls=TextLoader)
docs = loader.lazy_load()

for f in docs:
    print(f, f.metadata)
