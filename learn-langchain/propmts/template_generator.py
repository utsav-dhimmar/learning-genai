from pathlib import Path

from langchain_core.prompts import PromptTemplate

base = Path("/")
TEMPLATE_PATH = base.cwd() / "learn-langchain" / "propmts" / "template.json"

template = PromptTemplate(
    template="""You are professional book summary write who can write summary of any book in various tones
now write summary of book named {book_name} in {tone} tone in {word_length}.
if you didn't have any info regrading the book then you can directly say no""",
    input_variables=["book_name", "tone", "word_length"],
    validate_template=True,
)
template.save(TEMPLATE_PATH)
