from pydantic import Field, BaseModel
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI


# class Person(TypedDict):
#     name: str
#     age: int


# me: Person = {"name": "utsav", "age": 19}
# print(me)

# set_debug(True)
# set_verbose(True)

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
)


# TypedDict
class Reviews(TypedDict):
    # summary: str
    # sentiment: str
    summary: Annotated[str, "Brief summary of the review"]
    sentiment: Annotated[
        str, "return the sentiment of review Positive,negative,Neutral"
    ]
    prons: Annotated[Optional[list[str]], "Write All prons"]
    cons: Annotated[Optional[list[str]], "Write All cons"]


# Todo Complete after learning pydantic
# class Reviews(BaseModel):

#     summary: str = Field(
#         ...,
#     )
#     sentiment: Annotated[
#         str, "return the sentiment of review Positive,negative,Neutral"
#     ]
#     prons: Annotated[Optional[list[str]], "Write All prons"]
#     cons: Annotated[Optional[list[str]], "Write All cons"]


# JSON Schema

structure_model = model.with_structured_output(Reviews)

example = {
    "1": """
     The mobile perfromance is good but software and looks suck still v shape Notch in 2026 looks very old, software is decent thier are some preinstall app which i wont able to delete but thank full no full of ads like redmi
    """,
    "2": """
This ASUS TUF laptop is literally tough and durable, the build quality feels solid and premium. Performance is excellent, apps open fast, gaming and multitasking are smooth, and the overall speed is very impressive. Battery backup and display are also good for daily work and entertainment. Only one thing to note, Microsoft Office is not lifetime free, it’s a limited license. Apart from that, totally satisfied. Worth buying""",
    "3": """
    I've been using the ASUS TUF Gaming A15 with AMD Ryzen 7 7435HS for a few weeks now, and I'm impressed by its raw power and features. Here's my detailed review:

Pros:

- Lightning-Fast Performance: The Ryzen 7 7435HS processor handles demanding games and tasks with ease, providing seamless gaming and content creation experiences.
- Immersive Gaming: The NVIDIA GeForce RTX 3060 graphics card delivers stunning visuals, and the 144Hz display ensures smooth gameplay.
- Long-Lasting Battery: The laptop lasts up to 6 hours on a single charge (Regular use), making it perfect for long gaming sessions or work trips.
- Ergonomic Design: The TUF Gaming A15's rugged design and comfortable keyboard make it ideal for extended use.
- Affordable: Considering its specs and performance, this laptop is a great value for money.

Cons:

- Thermal Management: The laptop can get hot during intense gaming sessions, but the dual-fan design helps keep temperatures in check.
- Webcam Quality: The webcam could be better, but it's not a deal-breaker for me.

Overall, the ASUS TUF Gaming A15 is an excellent choice for gamers and content creators seeking a powerful, feature-packed laptop without breaking the bank. Its performance, display, and battery life make it a top contender in its class.

Recommendation: If you're looking for a reliable gaming laptop with impressive specs, look no further than the ASUS TUF Gaming A15.

Rating Breakdown:

- Performance: 5/5
- Display: 5/5
- Battery Life: 4.5/5
- Design: 4.5/5
- Value: 5/5""",
}


res = structure_model.invoke(input=example.get("3", "1"))

print(res)
