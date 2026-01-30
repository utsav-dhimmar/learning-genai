from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")
# its a Retriever not a search engine

q = "Python,JavaScript, AI"

docs = retriever.invoke(q)
for index, doc in enumerate(docs):
    print(f"--- doc {index + 1} ---- ")
    print(doc)
