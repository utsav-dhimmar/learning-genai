from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from pathlib import Path

load_dotenv()

BASE_PATH = Path("/")
# setup model
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    # max_tokens=100,
    max_retries=2,
    streaming=True,
)


def get_sys_prompt() -> str:
    propmt_file_location = BASE_PATH.cwd() / "learn-langchain" / "prompt.txt"
    if not propmt_file_location.exists():
        raise FileNotFoundError("Prompt file not found")

    with propmt_file_location.open("r") as file:
        return file.read()


sys_message = SystemMessage(get_sys_prompt())
propmt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [sys_message, HumanMessage("{input}")]
)


output_prase = StrOutputParser()
# how it procecc will got propmt -> it got to llm and -> output


print("AI ChatBot that help for learning from first principle")

while True:
    user_input = input("You:")

    if user_input.lower() == "exit" or user_input.lower == "bye":
        print("BYE BYE!!!")
        break
    message = [sys_message, HumanMessage(user_input)]
    res = model.invoke(message)
    human_readable_answer = HumanMessage(content_blocks=res.content_blocks).content
    print(f"AI:{human_readable_answer}")
