from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()


llm = GoogleGenerativeAI(model="gemini-3-flash-preview")

res = llm.invoke("what is ai explain like i am 5 in 100 words?")

print(res)
