from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from template_generator import TEMPLATE_PATH


def parser_res(res_input: str | dict[str, Any]) -> str:
    """
    - Take `AIMessage` as input and parser it return only response text if it is fail then it return whole answer
    """
    if isinstance(res_input, str):
        return res_input
    elif isinstance(res_input, dict):
        text_content = res_input.get("text")
        if text_content is not None:
            return str(text_content)
    return f"unable to parser default {res_input}"


load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
)


st.header("Static and Dynamic Propmts")

book_input = st.selectbox(
    "Select book name", ["Rich Dad Poor Dad", "Python in Nutshell"]
)

tone_input = st.selectbox(
    "Select tone of Explanation",
    ["Beginner-Friendly", "Code example", "Techinal", "Serious"],
)
length_input = st.selectbox(
    "Select Length of input",
    [
        "Short (max 2 paragraph)",
        "Medium (3-5 paragraph)",
        "long (max 8 paragraph)",
    ],
)


propmt = load_prompt(TEMPLATE_PATH)

if st.button("Summries"):
    chain = propmt | model
    res = chain.invoke(
        {
            "book_name": book_input,
            "tone": tone_input,
            "word_length": length_input,
        }
    )
    print(res.content)
    st.write(parser_res(res.content[0]))
