from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
)
# propmt = PromptTemplate(
#     template="Generate 5 facts about {topic}", input_variables=["topic"]
# )

# chain = propmt | model
# res = chain.invoke({"topic": "javascript"})
# print(res.content[0])


prompt1 = PromptTemplate(
    template="Generate report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give summary of {report}", input_variables=["report"]
)


parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
res = chain.invoke({"topic": "impact of ai in india"})
print(res)
chain.get_graph().print_ascii()
