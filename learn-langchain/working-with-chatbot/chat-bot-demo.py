import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from helper_functions import parser_res
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.2,  # how creative answer is
)

res = llm.invoke("what is ai explain like i am 5 in 100 words?")


print(parser_res(res.content[0]))
