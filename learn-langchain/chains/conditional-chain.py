from pydantic import Field
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import TypedDict, Literal, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

model = ChatGoogleGenerativeAI(
    # model="gemini-3-flash-preview",
    model="gemini-2.5-flash-lite",
)


class ReviewSentimentModel(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


py_parser = PydanticOutputParser(pydantic_object=ReviewSentimentModel)

parser = StrOutputParser()

feedback_analyze = PromptTemplate(
    template="You have to analyze the sentiment of review and classify into 'positive' or 'negative' in one word \n review-'{review}'",
    input_variables=["review"],
    partial_variables={"format_instrcution": py_parser.get_format_instructions()},
)

positive_feedback_chain = (
    PromptTemplate(
        template="Since review is positive thank full to customer and greet them {review}",
        input_variables=["review"],
    )
    | model
    | parser
)
negative_feedback_chain = (
    PromptTemplate(
        template="Since review is negative first say sorry to customer and then ask what wrong they feel about product {review}",
        input_variables=["review"],
    )
    | model
    | parser
)


default_chain = RunnableLambda(lambda x: "Sorry, no info available for this role.")
branches = RunnableBranch(
    # (condition,which_chain_execute aka runnable)
    (lambda x: x == "positive", positive_feedback_chain),
    (lambda x: x == "negative", negative_feedback_chain),
    default_chain,
)
feeadback_classifier_chain = feedback_analyze | model | parser

chain = feeadback_classifier_chain | branches
# feedback_res = feeadback_classifier_chain.invoke(
#     {"review": "useless software old version work better then current(v11) one"}
# )
# print(f"{feedback_res=}")

res = chain.invoke(
    {"review": "useless software old version work better then current(v11) one"}
)

print(res)

chain.get_graph().print_ascii()
