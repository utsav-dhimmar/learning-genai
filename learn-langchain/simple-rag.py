
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

load_dotenv()

base = Path("/")
pdf_path = base.cwd() / "learn-langchain" / "data.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
# print(f"{loader=}, {docs=}")

split_data = docs
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = DocArrayInMemorySearch.from_documents(split_data, embeddings)
retriver = vector_store.as_retriever()

# print(f"{vector_store=}")

llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    # max_tokens=100,
    max_retries=2,
    streaming=True,
)
system_prompt = (
    "Use the following pieces of retrieved context to answer "
    "the user's question. If you don't know the answer, say that you "
    "don't know. Keep the answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

q_a_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriver, q_a_chain)

response = rag_chain.invoke({"input": "Summaries it"})
print(response["answer"])
