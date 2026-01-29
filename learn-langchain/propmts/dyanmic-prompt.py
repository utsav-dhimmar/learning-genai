from langchain_core.prompts import ChatPromptTemplate

# chat_template = ChatPromptTemplate(
#     [
#         ("system", "You are export of {domain}"),
#         ("human", "Explain {topic} topic in simple terms"),
#     ]
#     # [
#     #     SystemMessage("You are export of {domain}"),
#     #     HumanMessage("Explain {topic} topic in simple terms"),
#     # ]
#     # I don't know why not working
# )

chat_template = ChatPromptTemplate.from_messages(
    # [
    #     SystemMessage("You are export of {domain}"),
    #     HumanMessage("Explain {topic} topic in simple terms"),
    # ] # Not working
    [
        ("system", "You are export of {domain}"),
        ("human", "Explain {topic} topic in simple terms"),
    ]
)

prompt = chat_template.invoke(
    {"domain": "langchain", "topic": "ChatPromptTemplate"}
)

print(prompt)
