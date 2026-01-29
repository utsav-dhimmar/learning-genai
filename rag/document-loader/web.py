from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader

loader = WebBaseLoader(
    web_paths=["https://portfolio.utsav-dev.workers.dev/", "https://google.com/"],
    encoding="utf-8",
)

docs = loader.load()

print(docs)
