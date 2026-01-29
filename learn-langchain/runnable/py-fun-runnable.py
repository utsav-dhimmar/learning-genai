from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
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


def my_word_counter(text) -> int:
    return len(text.split())


word_counter = RunnableLambda(my_word_counter)
p_chain = RunnableParallel(
    {
        "length": word_counter,  # RunnableLambda(lambda x: len(x.split()))
        "smart": RunnablePassthrough(),
    }
)
final_chain = RunnableSequence(jock_gen_chain, p_chain)
res = final_chain.invoke({"topic": q})

print(res)
