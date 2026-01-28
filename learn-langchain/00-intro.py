from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
# setup model
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    # max_tokens=100,
    max_retries=2,
    streaming=True,
)
propmt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are highly intelligance chat bot",
        ),
        ("human", "{input}"),
    ]
)


output_prase = StrOutputParser()
# how it procecc will got propmt -> it got to llm and -> output
chain = propmt | llm | output_prase


print("Simple AI bot built using langchain")

while True:
    user_input = input("You:")

    if user_input.lower() == "exit" or user_input.lower == "bye":
        print("BYE BYE!!!")
        break

    res = chain.invoke({"input": user_input})
    print(f"AI:{res}")
