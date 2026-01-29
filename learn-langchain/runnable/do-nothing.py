from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
model = ChatGoogleGenerativeAI(
    # model="gemini-3-flash-preview",
    model="gemini-2.5-flash-lite",
)
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="write funny joke on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the joke \n joke: {joke}", input_variables=["joke"]
)

q = "Python"
# chain_1 = prompt1 | model | parser
# chain_2 = prompt2| model| parser

jock_gen_chain = RunnableSequence(prompt1, model, parser)
p_chain = RunnableParallel(
    {
        "explain": RunnableSequence(prompt2, model, parser),
        "smart": RunnablePassthrough(),
    }
)
final_chain = RunnableSequence(jock_gen_chain, p_chain)
res = final_chain.invoke({"topic": q})

print(res)
