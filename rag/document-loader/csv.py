from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
)
from pathlib import Path

location = Path("/").cwd() / "rag" / "document-loader" / "data.csv"

loader = CSVLoader(file_path=location)

doc = loader.load()
print(doc)
