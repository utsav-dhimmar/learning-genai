from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_history = [
    "HumanMessage(content='hi', additional_kwargs={}, response_metadata={}), AIMessage(content='Hello! How can I help you today? If you have any questions or need help with a calculation, feel free to ask!', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='pi-2', additional_kwargs={}, response_metadata={}), AIMessage(content='To calculate the value of the expression **$\\pi - 2\\pi - 2$**, we follow these steps:\n\n### 1. Combine the like terms (the terms involving $\\pi$):\n$$\\pi - 2\\pi = -1\\pi \\text{ (or simply } -\\pi)$$\n\n### 2. Rewrite the expression:\n$$-\\pi - 2$$\n\nThis is the **exact form** of the result.\n\n### 3. Calculate the approximate decimal value:\nUsing the approximation $\\pi \\approx 3.14159$:\n$$-3.14159 - 2 = -5.14159$$\n\n**Final Answer:**\nThe exact value is **$-\\pi - 2$**, which is approximately **$-5.14159$**.', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='exit', additional_kwargs={}, response_metadata={})"
]

chat_template = ChatPromptTemplate(
    [
        ("system", "You are conversation analizer"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)
p = chat_template.invoke({"chat_history": chat_history, "query": "hi 222"})
print(p)
