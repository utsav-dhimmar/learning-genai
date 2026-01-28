from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=32,
    task_type="SEMANTIC_SIMILARITY",
)


def generate_embedding_for_single_text(txt: str) -> list[int | float]:
    return embeddings.embed_query("I am Utsav")


def generate_embedding_for_multiple_statements(
    texts: list[str],
) -> list[list[int | float]]:
    return embeddings.embed_documents(texts=texts)


vector = generate_embedding_for_multiple_statements(["I am Utsav", "hello"])
print(vector)
