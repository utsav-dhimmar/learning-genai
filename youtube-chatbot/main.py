import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

load_dotenv()


def get_youtube_id_from_url(url: str) -> str:
    """Get Youtube video id from the youtube url"""
    match = re.match(
        r"^https?://.*(?:youtu.be/|v/|u/\w/|embed/|watch\?v=)([^#&?]*).*$", url
    )
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")


def get_transcribe_as_text(video_id) -> str:
    """takes youtube video id as input and return its transcribe as text"""
    ytt_api = YouTubeTranscriptApi()

    fetch_transcribe = ytt_api.fetch(video_id, languages=("en", "hi"))
    formatter = TextFormatter()
    transcribe = formatter.format_transcript(fetch_transcribe)
    return transcribe


# https://www.youtube.com/watch?v=30JGQcRB3Nw
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def split_text(text: str) -> list[str]:
    return text_splitter.split_text(text)


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)


chroma_instace = Chroma(
    collection_name="youtube_video_info",
    embedding_function=embeddings,
    persist_directory="chroma_db_data",
)


output_prase = StrOutputParser()
# now it procecc will got propmt -> it got to llm and -> output

# setup model
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    # model="gemini-3-pro-preview",
    model="gemini-3-flash-preview",
    # model="gemini-2.5-flash-lite",
    # max_tokens=100,
    max_retries=2,
    streaming=True,
)

prompt = PromptTemplate(
    template="""
    You are very smart ai chat bot that only give relevenat answer in plain text format from the give transcribe context,
    if context do not have Enough information. Then you can Politely Say that the context do not have information regarding this query .
    {context}
    query:{query}
    """,
    input_variables=["context", "query"],
)

chain = prompt | llm | output_prase


video_url = input(
    "enter video url(make sure video has transcibe enable Hindi or English): "
)
video_id = get_youtube_id_from_url(video_url)
transcribe = get_transcribe_as_text(video_id)
chucks = text_splitter.create_documents(split_text(transcribe))
vector_store = chroma_instace.from_documents(documents=chucks)
retriveral = vector_store.as_retriever(search_type="similarity")


def get_relevant_documents(q: str) -> list[Document]:
    return retriveral.invoke(q)


def get_context_as_text(docs: list[Document]) -> str:
    return "\n\n\n\n".join(doc.page_content for doc in docs)


print("Simple AI bot built using langchain")
while True:
    user_input = input("You:").strip()
    docs = get_relevant_documents(user_input)
    context = get_context_as_text(docs)
    if user_input.lower() == "exit" or user_input.lower() == "bye":
        print("BYE BYE!!!")
        break

    res = chain.stream({"context": context, "query": user_input})
    print(f"{"--"*20} AI Response{"--"*20}")
    for r in res:
        print(r)

    print(f"{"--"*20} AI Response End{"--"*20}")
