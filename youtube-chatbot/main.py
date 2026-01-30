from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
# setup model
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    # model="gemini-3-pro-preview",
    model="gemini-3-flash-preview",
    # model="gemini-2.5-flash-lite",
    # max_tokens=100,
    max_retries=2,
    streaming=True,
)


chat_history: list[HumanMessage | AIMessage] = []


prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful and accurate AI assistant. When performing calculations, think step-by-step and verify your arithmetic carefully. Always provide correct mathematical answers.",
        ),  # init messages
        MessagesPlaceholder(variable_name="chat_history"),  # for chat history
        ("human", "{input}"),
    ]
)


output_prase = StrOutputParser()
# now it procecc will got propmt -> it got to llm and -> output


chain = prompt | llm | output_prase

print("Simple AI bot built using langchain")

while True:
    user_input = input("You:").strip()
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == "exit" or user_input.lower == "bye":
        print(chat_history)
        print("BYE BYE!!!")
        break

    res: str = chain.invoke({"chat_history": chat_history, "input": user_input})

    # add in chat history
    chat_history.append(AIMessage(content=res))
    print(f"AI:{res}")
