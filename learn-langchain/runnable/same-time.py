from langchain_core.runnables import RunnableSequence, RunnableParallel
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
    template="Generate post on {topic} for facebook", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate post on {topic} for instagram", input_variables=["topic"]
)

q = "JavaScript"

chain = RunnableParallel(
    {
        "fb": RunnableSequence(prompt1, model, parser),
        "insta": RunnableSequence(prompt2, model, parser),
    }
)

#  RunnableSequence(prompt1, model, parser, prompt2, model, parser)

res = chain.invoke({"topic": q})

print("fb", res["fb"])
print("insta", res["insta"])
